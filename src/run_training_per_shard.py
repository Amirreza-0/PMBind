#!/usr/bin/env python
"""
Training script for separate models per shard with balanced sampling.

This script trains individual models on each balanced shard separately,
allowing for model diversity and potentially better handling of imbalanced data.
Each shard gets its own model trained with balanced sampling and saved separately.
"""
import csv
import sys
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pyarrow.parquet as pq
import json
import os
from tqdm import tqdm
import argparse
import random

# Local imports
from utils import (get_embed_key, NORM_TOKEN, MASK_TOKEN, PAD_TOKEN, PAD_VALUE, MASK_VALUE,
                   clean_key, masked_categorical_crossentropy, seq_to_indices,
                   AMINO_ACID_VOCAB, PAD_INDEX, BLOSUM62, AA, PAD_INDEX_OHE, BinaryMCC,
                   AsymmetricPenaltyBinaryCrossentropy)

from models import pmbind_multitask_modified as pmbind
from visualizations import visualize_training_history

mixed_precision = True  # Enable mixed precision for significant speedup

if mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f'Mixed precision enabled: {policy}')

# --- Globals for the tf.data pipeline ---
MHC_EMBEDDING_TABLE = None
MAX_PEP_LEN = 0
MAX_MHC_LEN = 0
ESM_DIM = 0
MHC_CLASS = 1

# --- Create a BLOSUM62 lookup table for on-the-fly feature creation ---
_blosum_vectors = [BLOSUM62[aa] for aa in AMINO_ACID_VOCAB]
_blosum_vectors.append([PAD_VALUE] * len(_blosum_vectors[0]))
BLOSUM62_TABLE = tf.constant(np.array(_blosum_vectors), dtype=tf.float32)


def load_embedding_table(lookup_path):
    """Loads the NPZ lookup file into a TensorFlow constant for fast GPU access."""
    global MHC_EMBEDDING_TABLE
    with np.load(lookup_path) as data:
        num_embeddings = len(data.files)
        table = np.zeros((num_embeddings, MAX_MHC_LEN, ESM_DIM), dtype=np.float16)
        for i in range(num_embeddings):
            table[i] = data[str(i)]

    MHC_EMBEDDING_TABLE = tf.constant(table)
    print(f"✓ Loaded MHC embedding table into a tf.constant with shape: {MHC_EMBEDDING_TABLE.shape}")


def _parse_tf_example(example_proto):
    """Parses a lightweight TFRecord example and performs on-the-fly data reconstruction."""
    feature_description = {
        'pep_indices': tf.io.FixedLenFeature([], tf.string),
        'pep_ohe_indices': tf.io.FixedLenFeature([], tf.string),
        'mhc_indices': tf.io.FixedLenFeature([], tf.string),
        'mhc_ohe_indices': tf.io.FixedLenFeature([], tf.string),
        'embedding_id': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    pep_indices = tf.io.parse_tensor(parsed['pep_indices'], out_type=tf.int8)
    mhc_indices = tf.io.parse_tensor(parsed['mhc_indices'], out_type=tf.int8)
    pep_ohe_indices = tf.io.parse_tensor(parsed['pep_ohe_indices'], out_type=tf.int8)
    mhc_ohe_indices = tf.io.parse_tensor(parsed['mhc_ohe_indices'], out_type=tf.int8)

    embedding_id = tf.cast(parsed['embedding_id'], tf.int32)
    mhc_emb = tf.gather(MHC_EMBEDDING_TABLE, embedding_id)

    compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    mhc_emb = tf.cast(mhc_emb, compute_dtype)

    pep_blossom62_input = tf.gather(BLOSUM62_TABLE, tf.cast(pep_indices, tf.int32))
    pep_blossom62_input = tf.cast(pep_blossom62_input, compute_dtype)

    vocab_size_ohe = len(AA)
    pep_ohe_target = tf.one_hot(pep_ohe_indices, depth=vocab_size_ohe, dtype=tf.float32)
    mhc_ohe_target = tf.one_hot(mhc_ohe_indices, depth=vocab_size_ohe, dtype=tf.float32)

    pep_mask = tf.where(pep_indices == PAD_INDEX, PAD_TOKEN, NORM_TOKEN)
    mhc_mask = tf.where(mhc_indices == PAD_INDEX, PAD_TOKEN, NORM_TOKEN)

    labels = tf.cast(parsed['label'], tf.int32)
    labels = tf.expand_dims(labels, axis=-1)

    return {
        "pep_blossom62": pep_blossom62_input,
        "pep_mask": pep_mask,
        "mhc_emb": mhc_emb,
        "mhc_mask": mhc_mask,
        "pep_ohe_target": pep_ohe_target,
        "mhc_ohe_target": mhc_ohe_target,
        "labels": labels
    }


def apply_dynamic_masking(features, emd_mask_d2=True):
    """Applies random masking for training augmentation inside the tf.data pipeline."""
    # Peptide Masking
    valid_pep_positions = tf.where(tf.equal(features["pep_mask"], NORM_TOKEN))
    num_valid_pep = tf.shape(valid_pep_positions)[0]

    num_to_mask_pep = tf.maximum(2, tf.cast(tf.cast(num_valid_pep, tf.float32) * 0.15, tf.int32))
    shuffled_pep_indices = tf.random.shuffle(valid_pep_positions)[:num_to_mask_pep]

    if tf.shape(shuffled_pep_indices)[0] > 0:
        features["pep_mask"] = tf.tensor_scatter_nd_update(features["pep_mask"], shuffled_pep_indices,
                                                           tf.repeat(MASK_TOKEN, num_to_mask_pep))
        feat_dtype = features["pep_blossom62"].dtype
        mask_updates_pep = tf.fill([num_to_mask_pep, tf.shape(features["pep_blossom62"])[-1]],
                                   tf.cast(MASK_VALUE, feat_dtype))
        features["pep_blossom62"] = tf.tensor_scatter_nd_update(features["pep_blossom62"], shuffled_pep_indices,
                                                                mask_updates_pep)

    # MHC Masking
    valid_mhc_positions = tf.where(tf.equal(features["mhc_mask"], NORM_TOKEN))
    num_valid_mhc = tf.shape(valid_mhc_positions)[0]
    num_to_mask_mhc = tf.maximum(5, tf.cast(tf.cast(num_valid_mhc, tf.float32) * 0.30, tf.int32))
    shuffled_mhc_indices = tf.random.shuffle(valid_mhc_positions)[:num_to_mask_mhc]

    if tf.shape(shuffled_mhc_indices)[0] > 0:
        features["mhc_mask"] = tf.tensor_scatter_nd_update(features["mhc_mask"], shuffled_mhc_indices,
                                                           tf.repeat(MASK_TOKEN, num_to_mask_mhc))
        mhc_dtype = features["mhc_emb"].dtype
        mask_updates_mhc = tf.fill([num_to_mask_mhc, tf.shape(features["mhc_emb"])[-1]], tf.cast(MASK_VALUE, mhc_dtype))
        features["mhc_emb"] = tf.tensor_scatter_nd_update(features["mhc_emb"], shuffled_mhc_indices, mask_updates_mhc)

    if emd_mask_d2:
        remaining_valid_mhc = tf.where(tf.equal(features["mhc_mask"], NORM_TOKEN))

        if tf.shape(remaining_valid_mhc)[0] > 0:
            valid_embeddings = tf.gather_nd(features["mhc_emb"], remaining_valid_mhc)
            dim_mask = tf.random.uniform(shape=tf.shape(valid_embeddings), dtype=features["mhc_emb"].dtype) < tf.cast(
                0.15, features["mhc_emb"].dtype)
            masked_embeddings = valid_embeddings * tf.cast(~dim_mask, features["mhc_emb"].dtype)
            features["mhc_emb"] = tf.tensor_scatter_nd_update(features["mhc_emb"], remaining_valid_mhc,
                                                              masked_embeddings)

    return features


def create_dataset(file_list, batch_size, is_training=True, apply_masking=True, subset_steps=None):
    """Creates a tf.data.Dataset from a specific list of files."""
    if not file_list:
        raise ValueError("File list provided to create_dataset is empty.")

    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP", num_parallel_reads=tf.data.AUTOTUNE),
        cycle_length=len(file_list),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not is_training
    )

    if is_training:
        dataset = dataset.shuffle(buffer_size=100_000, reshuffle_each_iteration=True)

    dataset = dataset.map(_parse_tf_example, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training and apply_masking:
        dataset = dataset.map(apply_dynamic_masking, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=is_training)

    # Apply subset if specified
    if subset_steps is not None:
        dataset = dataset.take(subset_steps)

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


@tf.function()
def train_step(model, batch_data, loss_fn, optimizer, metrics, class_weights, run_conf):
    """Compiled training step with proper mixed precision handling."""
    with tf.GradientTape() as tape:
        outputs = model(batch_data, training=True)
        raw_cls_loss = loss_fn(batch_data["labels"], tf.cast(outputs["cls_ypred"], tf.float32))

        labels_flat = tf.reshape(batch_data["labels"], [-1])
        sample_weights = tf.gather(class_weights, tf.cast(labels_flat, tf.int32))
        weighted_cls_loss = tf.reduce_mean(raw_cls_loss * sample_weights)

        raw_recon_loss_pep = masked_categorical_crossentropy(outputs["pep_ytrue_ypred"], batch_data["pep_mask"])
        raw_recon_loss_mhc = masked_categorical_crossentropy(outputs["mhc_ytrue_ypred"], batch_data["mhc_mask"])

        weighted_cls_loss = tf.where(tf.math.is_finite(weighted_cls_loss), weighted_cls_loss, 0.0)
        raw_recon_loss_pep = tf.where(tf.math.is_finite(raw_recon_loss_pep), raw_recon_loss_pep, 0.0)
        raw_recon_loss_mhc = tf.where(tf.math.is_finite(raw_recon_loss_mhc), raw_recon_loss_mhc, 0.0)

        total_loss_weighted = (run_conf["CLS_LOSS_WEIGHT"] * weighted_cls_loss) + \
                             (run_conf["PEP_RECON_LOSS_WEIGHT"] * raw_recon_loss_pep) + \
                             (run_conf["MHC_RECON_LOSS_WEIGHT"] * raw_recon_loss_mhc)
        total_loss_weighted = tf.clip_by_value(total_loss_weighted, 0.0, 10.0)

        if mixed_precision:
            scaled_loss = optimizer.scale_loss(total_loss_weighted)
            grads = tape.gradient(scaled_loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        else:
            grads = tape.gradient(total_loss_weighted, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    labels_flat = tf.reshape(batch_data["labels"], [-1])
    preds_flat = tf.reshape(outputs["cls_ypred"], [-1])

    metrics['train_acc'].update_state(labels_flat, tf.cast(preds_flat, tf.float32))
    metrics['train_auc'].update_state(labels_flat, tf.cast(preds_flat, tf.float32))
    metrics['train_precision'].update_state(labels_flat, tf.cast(preds_flat, tf.float32))
    metrics['train_recall'].update_state(labels_flat, tf.cast(preds_flat, tf.float32))
    metrics['train_mcc'].update_state(labels_flat, tf.cast(preds_flat, tf.float32))
    metrics['train_loss'].update_state(total_loss_weighted)
    metrics['pep_loss'].update_state(raw_recon_loss_pep)
    metrics['mhc_loss'].update_state(raw_recon_loss_mhc)
    metrics['cls_loss'].update_state(weighted_cls_loss)


@tf.function()
def eval_step(model, batch_data, loss_fn, metrics, run_conf):
    """Compiled evaluation step."""
    outputs = model(batch_data, training=False)
    labels_flat = tf.reshape(batch_data["labels"], [-1])
    preds_flat = tf.reshape(outputs["cls_ypred"], [-1])

    cls_loss = loss_fn(batch_data["labels"], tf.cast(outputs["cls_ypred"], tf.float32))
    recon_loss_pep = masked_categorical_crossentropy(outputs["pep_ytrue_ypred"], batch_data["pep_mask"])
    recon_loss_mhc = masked_categorical_crossentropy(outputs["mhc_ytrue_ypred"], batch_data["mhc_mask"])

    total_loss_weighted = (run_conf["CLS_LOSS_WEIGHT"] * cls_loss) + \
                         (run_conf["PEP_RECON_LOSS_WEIGHT"] * recon_loss_pep) + \
                         (run_conf["MHC_RECON_LOSS_WEIGHT"] * recon_loss_mhc)

    metrics['val_acc'].update_state(labels_flat, tf.cast(preds_flat, tf.float32))
    metrics['val_auc'].update_state(labels_flat, tf.cast(preds_flat, tf.float32))
    metrics['val_precision'].update_state(labels_flat, tf.cast(preds_flat, tf.float32))
    metrics['val_recall'].update_state(labels_flat, tf.cast(preds_flat, tf.float32))
    metrics['val_mcc'].update_state(labels_flat, tf.cast(preds_flat, tf.float32))
    metrics['val_loss'].update_state(total_loss_weighted)


def train_single_shard(shard_idx, positive_train_file, negative_train_file, val_ds,
                      lookup_path_train, lookup_path_val, out_dir, run_config,
                      epochs, batch_size, lr, embed_dim, heads, noise_std, enable_masking=True,
                      train_samples=None, num_shards=1):
    """Train a single model on one balanced shard."""

    print(f"\n{'='*80}")
    print(f"TRAINING MODEL FOR SHARD {shard_idx}")
    print(f"{'='*80}")
    print(f"Negative shard: {os.path.basename(negative_train_file)}")
    print(f"Output directory: {out_dir}")

    # Load training embedding table
    load_embedding_table(lookup_path_train)

    # Create training dataset for this shard
    epoch_train_files = [positive_train_file, negative_train_file]

    # Calculate training steps for subset if applicable
    train_steps = None
    if train_samples:
        train_steps = train_samples // (num_shards * batch_size)  # Divide by number of shards

    train_ds = create_dataset(epoch_train_files, batch_size, is_training=True,
                             apply_masking=enable_masking, subset_steps=train_steps)

    # Build model
    model = pmbind(max_pep_len=MAX_PEP_LEN, max_mhc_len=MAX_MHC_LEN, emb_dim=embed_dim,
                   heads=heads, noise_std=noise_std, latent_dim=embed_dim * 2,
                   ESM_dim=ESM_DIM, drop_out_rate=0.4, l2_reg=0.03)
    model.build(train_ds.element_spec)

    # Setup optimizer and loss
    base_optimizer = keras.optimizers.Lion(learning_rate=lr, weight_decay=1e-4)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer) if mixed_precision else base_optimizer
    loss_fn = AsymmetricPenaltyBinaryCrossentropy(
        label_smoothing=run_config["LABEL_SMOOTHING"],
        asymmetry_strength=run_config["ASYMMETRIC_LOSS_SCALE"]
    )

    # Class weights for balanced training
    class_weight_dict = {0: 1.0, 1: 1.0}  # Since we're using balanced shards
    class_weights_tensor = tf.constant([class_weight_dict[0], class_weight_dict[1]], dtype=tf.float32)

    # Metrics
    metrics = {
        'train_loss': tf.keras.metrics.Mean(name='train_loss'),
        'train_auc': tf.keras.metrics.AUC(name='train_auc'),
        'train_acc': tf.keras.metrics.BinaryAccuracy(name='train_accuracy'),
        'train_precision': tf.keras.metrics.Precision(name='train_precision'),
        'train_recall': tf.keras.metrics.Recall(name='train_recall'),
        'train_mcc': BinaryMCC(name='train_mcc'),
        'val_auc': tf.keras.metrics.AUC(name='val_auc'),
        'val_acc': tf.keras.metrics.BinaryAccuracy(name='val_accuracy'),
        'val_precision': tf.keras.metrics.Precision(name='val_precision'),
        'val_recall': tf.keras.metrics.Recall(name='val_recall'),
        'val_mcc': BinaryMCC(name='val_mcc'),
        'val_loss': tf.keras.metrics.Mean(name='val_loss'),
        'pep_loss': tf.keras.metrics.Mean(name='pep_loss'),
        'mhc_loss': tf.keras.metrics.Mean(name='mhc_loss'),
        'cls_loss': tf.keras.metrics.Mean(name='cls_loss'),
    }

    history = {k: [] for k in
               ["train_loss", "train_auc", "train_acc", "train_precision", "train_recall", "train_f1", "train_mcc",
                "val_auc", "val_acc", "val_precision", "val_recall", "val_f1", "val_mcc", "val_loss",
                "cls_loss", "pep_loss", "mhc_loss"]}

    best_val_mcc = -1.0
    patience, patience_counter, min_improvement = 5, 0, 0.001
    lr_patience, lr_patience_counter, lr_reduction_factor, min_lr = 3, 0, 0.5, 1e-7

    # Training loop
    for epoch in range(epochs):
        print(f"\nShard {shard_idx} - Epoch {epoch + 1}/{epochs}")
        for m in metrics.values():
            m.reset_state()

        # Training
        pbar = tqdm(train_ds, desc=f"Training Shard {shard_idx}", unit="batch")
        for batch_data in pbar:
            train_step(model, batch_data, loss_fn, optimizer, metrics, class_weights_tensor, run_config)
            pbar.set_postfix({
                'Loss': f"{metrics['train_loss'].result():.4f}",
                'AUC': f"{metrics['train_auc'].result():.4f}",
                'MCC': f"{metrics['train_mcc'].result():.4f}",
            })

        # Validation - no need to reload embedding table since we're using training table consistently
        for batch_data in tqdm(val_ds, desc=f"Validating Shard {shard_idx}", unit="batch"):
            eval_step(model, batch_data, loss_fn, metrics, run_config)

        # Calculate F1 scores
        train_prec, train_recall = metrics['train_precision'].result(), metrics['train_recall'].result()
        train_f1 = tf.where(tf.equal(train_prec + train_recall, 0.0), 0.0,
                            (2.0 * train_prec * train_recall) / (train_prec + train_recall))

        val_prec, val_recall = metrics['val_precision'].result(), metrics['val_recall'].result()
        val_f1 = tf.where(tf.equal(val_prec + val_recall, 0.0), 0.0,
                          (2.0 * val_prec * val_recall) / (val_prec + val_recall))

        print(f"  Train: Loss={metrics['train_loss'].result():.4f}, MCC={metrics['train_mcc'].result():.4f}")
        print(f"  Val:   Loss={metrics['val_loss'].result():.4f}, MCC={metrics['val_mcc'].result():.4f}")

        # Log history
        history['train_loss'].append(float(metrics['train_loss'].result()))
        history['train_auc'].append(float(metrics['train_auc'].result()))
        history['train_acc'].append(float(metrics['train_acc'].result()))
        history['train_precision'].append(float(train_prec))
        history['train_recall'].append(float(train_recall))
        history['train_f1'].append(float(train_f1))
        history['train_mcc'].append(float(metrics['train_mcc'].result()))
        history['val_auc'].append(float(metrics['val_auc'].result()))
        history['val_acc'].append(float(metrics['val_acc'].result()))
        history['val_precision'].append(float(val_prec))
        history['val_recall'].append(float(val_recall))
        history['val_f1'].append(float(val_f1))
        history['val_mcc'].append(float(metrics['val_mcc'].result()))
        history['val_loss'].append(float(metrics['val_loss'].result()))
        history['cls_loss'].append(float(metrics['cls_loss'].result()))
        history['pep_loss'].append(float(metrics['pep_loss'].result()))
        history['mhc_loss'].append(float(metrics['mhc_loss'].result()))

        # Early stopping and checkpointing
        current_val_mcc = metrics['val_mcc'].result()
        if current_val_mcc > best_val_mcc + min_improvement:
            best_val_mcc = current_val_mcc
            patience_counter, lr_patience_counter = 0, 0
            model.save_weights(os.path.join(out_dir, f"best_model_shard_{shard_idx}.weights.h5"))
            print(f"  -> New best model saved with Val MCC: {best_val_mcc:.4f}")
        else:
            patience_counter += 1
            lr_patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

        # Learning rate reduction
        if lr_patience_counter >= lr_patience:
            current_lr = (optimizer.inner_optimizer.learning_rate.numpy()
                         if hasattr(optimizer, "inner_optimizer")
                         else optimizer.learning_rate.numpy())
            if current_lr > min_lr:
                new_lr = max(current_lr * lr_reduction_factor, min_lr)
                if hasattr(optimizer, 'inner_optimizer'):
                    optimizer.inner_optimizer.learning_rate.assign(new_lr)
                else:
                    optimizer.learning_rate.assign(new_lr)
                print(f"  -> Reducing learning rate to {new_lr:.2e}")
                lr_patience_counter = 0

        if patience_counter >= patience:
            print(f"  -> Early stopping. Best MCC: {best_val_mcc:.4f}")
            break

        # Training embedding table is already loaded consistently, no need to reload

    # Save final model and history
    model.save_weights(os.path.join(out_dir, f"final_model_shard_{shard_idx}.weights.h5"))
    with open(os.path.join(out_dir, f"training_history_shard_{shard_idx}.json"), "w") as f:
        json.dump(history, f, indent=4)

    print(f"✓ Shard {shard_idx} training complete. Best Val MCC: {best_val_mcc:.4f}")
    return best_val_mcc


def train_per_shard(tfrecord_dir, out_dir, mhc_class, epochs, batch_size, lr, embed_dim, heads, noise_std,
                   run_config, enable_masking=True, subset=1.0):
    """Train separate models for each balanced shard."""

    global MAX_PEP_LEN, MAX_MHC_LEN, ESM_DIM, MHC_CLASS
    MHC_CLASS = mhc_class

    # Load metadata
    with open(os.path.join(tfrecord_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    MAX_PEP_LEN, MAX_MHC_LEN, ESM_DIM, MHC_CLASS = (
        metadata['MAX_PEP_LEN'], metadata['MAX_MHC_LEN'],
        metadata['ESM_DIM'], metadata['MHC_CLASS']
    )

    # Get sample counts and apply subset
    train_samples = metadata.get('train_samples', None)
    val_samples = metadata.get('val_samples', None)

    if train_samples and subset < 1.0:
        original_train_samples = train_samples
        train_samples = int(train_samples * subset)
        val_samples = int(val_samples * subset) if val_samples else None
        print(f"Using {subset * 100:.1f}% subset of data: {train_samples:,} / {original_train_samples:,} train samples")
        if val_samples:
            print(f"Validation samples adjusted to: {val_samples:,}")

    # Define file paths
    positive_train_file = os.path.join(tfrecord_dir, "train", "positive_samples.tfrecord")
    negative_train_files = sorted(tf.io.gfile.glob(os.path.join(tfrecord_dir, "train", "negative_samples_*.tfrecord")))

    positive_val_file = os.path.join(tfrecord_dir, "validation", "positive_samples.tfrecord")
    validation_neg_files = sorted(
        tf.io.gfile.glob(os.path.join(tfrecord_dir, "validation", "negative_samples_*.tfrecord")))

    lookup_path_train = os.path.join(tfrecord_dir, "train_mhc_embedding_lookup.npz")
    lookup_path_val = os.path.join(tfrecord_dir, "validation_mhc_embedding_lookup.npz")

    # Verify files exist
    if not all(map(os.path.exists, [positive_train_file, positive_val_file, lookup_path_train, lookup_path_val])):
        raise FileNotFoundError("Could not find all required training/validation files.")

    if not negative_train_files or not validation_neg_files:
        raise FileNotFoundError("Could not find negative training or validation shards.")

    print(f"✓ Found 1 positive training file and {len(negative_train_files)} negative training shards.")
    print(f"✓ Found 1 positive validation file and {len(validation_neg_files)} negative validation shards.")

    # Create validation dataset (same for all shards) - use training embedding table for consistency
    load_embedding_table(lookup_path_train)
    val_files = [positive_val_file] + validation_neg_files[:1]  # Use first validation shard
    val_ds = create_dataset(val_files, batch_size, is_training=False, apply_masking=False)

    # Apply subset to validation dataset if needed
    if subset < 1.0 and val_samples:
        val_steps = val_samples // batch_size
        val_ds = val_ds.take(val_steps)
        print(f"Applied subset to validation: taking {val_steps} batches")

    # Save run configuration
    run_config['training_subset'] = subset
    run_config['validation_subset'] = subset
    run_config['MAX_PEP_LEN'] = MAX_PEP_LEN
    run_config['MAX_MHC_LEN'] = MAX_MHC_LEN
    run_config['ESM_DIM'] = ESM_DIM
    run_config['MHC_CLASS'] = MHC_CLASS
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=4)

    # Train models for each shard
    shard_results = {}
    num_shards = len(negative_train_files)

    for shard_idx, negative_train_file in enumerate(negative_train_files):
        print(f"\n{'='*100}")
        print(f"PROCESSING SHARD {shard_idx + 1}/{num_shards}")
        print(f"{'='*100}")

        best_mcc = train_single_shard(
            shard_idx=shard_idx,
            positive_train_file=positive_train_file,
            negative_train_file=negative_train_file,
            val_ds=val_ds,
            lookup_path_train=lookup_path_train,
            lookup_path_val=lookup_path_val,
            out_dir=out_dir,
            run_config=run_config,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            embed_dim=embed_dim,
            heads=heads,
            noise_std=noise_std,
            enable_masking=enable_masking,
            train_samples=train_samples,
            num_shards=num_shards
        )

        shard_results[f"shard_{shard_idx}"] = {
            "best_val_mcc": float(best_mcc),
            "negative_file": os.path.basename(negative_train_file)
        }

    # Save summary of all shard results
    with open(os.path.join(out_dir, "shard_results_summary.json"), "w") as f:
        json.dump(shard_results, f, indent=4)

    # Print summary
    print(f"\n{'='*100}")
    print("TRAINING SUMMARY")
    print(f"{'='*100}")
    for shard_name, results in shard_results.items():
        print(f"{shard_name}: Best Val MCC = {results['best_val_mcc']:.4f} ({results['negative_file']})")

    avg_mcc = np.mean([results['best_val_mcc'] for results in shard_results.values()])
    print(f"\nAverage MCC across all shards: {avg_mcc:.4f}")
    print(f"✓ All models saved in {out_dir}")


def main(args):
    """Main function to run the per-shard training pipeline."""
    RUN_CONFIG = {
        "MHC_CLASS": 1, "EPOCHS": 15, "BATCH_SIZE": 1024, "LEARNING_RATE": 1e-3,
        "EMBED_DIM": 16, "HEADS": 2, "NOISE_STD": 0.5,
        "LABEL_SMOOTHING": args.ls_param, "ASYMMETRIC_LOSS_SCALE": args.as_param,
        "CLS_LOSS_WEIGHT": 1.0, "PEP_RECON_LOSS_WEIGHT": 0.2, "MHC_RECON_LOSS_WEIGHT": 0.2,
        "description": "Per-shard training with balanced sampling for each individual model"
    }

    base_output_folder = "../results/PMBind_runs_per_shard/"

    fold_to_run = args.fold
    run_name = f"fold_{fold_to_run}_per_shard_mhc{RUN_CONFIG['MHC_CLASS']}_dim{RUN_CONFIG['EMBED_DIM']}_h{RUN_CONFIG['HEADS']}"
    out_dir = os.path.join(base_output_folder, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Starting per-shard training: {run_name}\nOutput directory: {out_dir}")

    tfrecord_dir = f"/media/amirreza/Crucial-500/PMBind_dataset/cross_validation_dataset/mhc{RUN_CONFIG['MHC_CLASS']}/tfrecords/fold_{fold_to_run:02d}_split/"

    if not os.path.exists(tfrecord_dir) or not os.path.exists(os.path.join(tfrecord_dir, 'metadata.json')):
        print(f"Error: TFRecord directory not found or is incomplete: {tfrecord_dir}")
        print("Please run the data splitting script first.")
        sys.exit(1)

    train_per_shard(
        tfrecord_dir=tfrecord_dir,
        out_dir=out_dir,
        mhc_class=RUN_CONFIG["MHC_CLASS"],
        epochs=RUN_CONFIG["EPOCHS"],
        batch_size=RUN_CONFIG["BATCH_SIZE"],
        lr=RUN_CONFIG["LEARNING_RATE"],
        embed_dim=RUN_CONFIG["EMBED_DIM"],
        heads=RUN_CONFIG["HEADS"],
        noise_std=RUN_CONFIG["NOISE_STD"],
        enable_masking=True,
        subset=args.subset,
        run_config=RUN_CONFIG
    )


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(e)
    try:
        tf.config.experimental.enable_tensor_float_32_execution(True)
    except:
        pass

    parser = argparse.ArgumentParser(description="Train separate models per shard with balanced sampling.")
    parser.add_argument("--fold", type=int, required=True, help="Fold number to run (e.g., 0-4).")
    parser.add_argument("--subset", type=float, default=1.0, help="Subset percentage of training data to use.")
    parser.add_argument("--ls_param", type=float, default=0.2, help="Label smoothing parameter.")
    parser.add_argument("--as_param", type=float, default=5.0, help="Asymmetric loss scaling parameter.")
    args = parser.parse_args()
    main(args)