#!/usr/bin/env python
"""
Fully optimized training and inference script for pmbind_multitask.

This script uses a highly efficient TFRecord-based data pipeline:
1.  An offline script (`create_tfrecords.py`) prepares all data.
2.  A central 'mhc_embedding_lookup.npz' file stores unique embeddings once.
3.  Lightweight TFRecord files store only pointers (integer IDs) to the embeddings.
4.  This script loads the lookup table into a tf.constant for fast GPU access.
5.  The tf.data pipeline reconstructs full batches on-the-fly, maximizing performance.
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

# Local imports
from utils import (get_embed_key, NORM_TOKEN, MASK_TOKEN, PAD_TOKEN, PAD_VALUE, MASK_VALUE,
                   clean_key, masked_categorical_crossentropy, seq_to_indices,
                   AMINO_ACID_VOCAB, PAD_INDEX, BLOSUM62, AA, PAD_INDEX_OHE)
from models import pmbind_multitask as pmbind
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
# The order of vectors must match the integer mapping from AMINO_ACID_VOCAB
_blosum_vectors = [BLOSUM62[aa] for aa in AMINO_ACID_VOCAB]
# Append a zero-vector to correspond to the PAD_INDEX
_blosum_vectors.append([PAD_VALUE] * len(_blosum_vectors[0]))
BLOSUM62_TABLE = tf.constant(np.array(_blosum_vectors), dtype=tf.float32)


# ──────────────────────────────────────────────────────────────────────
# TF.DATA PIPELINE
# ----------------------------------------------------------------------

def load_embedding_table(lookup_path):
    """Loads the NPZ lookup file into a TensorFlow constant for fast GPU access."""
    global MHC_EMBEDDING_TABLE
    with np.load(lookup_path) as data:
        num_embeddings = len(data.files)
        # store as float16 to save memory/bandwidth; compute will cast as needed
        table = np.zeros((num_embeddings, MAX_MHC_LEN, ESM_DIM), dtype=np.float16)
        for i in range(num_embeddings):
            table[i] = data[str(i)]

    MHC_EMBEDDING_TABLE = tf.constant(table)
    print(f"✓ Loaded MHC embedding table into a tf.constant with shape: {MHC_EMBEDDING_TABLE.shape}")


def _parse_tf_example(example_proto):
    """Parses a lightweight TFRecord example and performs on-the-fly data reconstruction."""
    # --- UPDATE THE FEATURE DESCRIPTION ---
    feature_description = {
        'pep_indices': tf.io.FixedLenFeature([], tf.string),
        'pep_ohe_indices': tf.io.FixedLenFeature([], tf.string),
        'mhc_indices': tf.io.FixedLenFeature([], tf.string),
        'mhc_ohe_indices': tf.io.FixedLenFeature([], tf.string),  # Add this
        'embedding_id': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    # Indices for BLOSUM62 input (values 0-23)
    pep_indices = tf.io.parse_tensor(parsed['pep_indices'], out_type=tf.int8)
    mhc_indices = tf.io.parse_tensor(parsed['mhc_indices'], out_type=tf.int8)
    # Indices for OHE target (values 0-21)
    pep_ohe_indices = tf.io.parse_tensor(parsed['pep_ohe_indices'], out_type=tf.int8)
    # --- PARSE THE NEW MHC INDICES ---
    mhc_ohe_indices = tf.io.parse_tensor(parsed['mhc_ohe_indices'], out_type=tf.int8)
    # ... (MHC embedding lookup remains the same) ...
    embedding_id = tf.cast(parsed['embedding_id'], tf.int32)
    mhc_emb = tf.gather(MHC_EMBEDDING_TABLE, embedding_id)
    # cast to compute dtype for faster kernels
    compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    mhc_emb = tf.cast(mhc_emb, compute_dtype)
    # Create BLOSUM62 input features for the peptide
    pep_blossom62_input = tf.gather(BLOSUM62_TABLE, tf.cast(pep_indices, tf.int32))
    pep_blossom62_input = tf.cast(pep_blossom62_input, compute_dtype)
    vocab_size_ohe = len(AA)  # This will be 21
    # Use the ohe_indices to create 21-dimensional one-hot vectors for both targets
    # keep targets in float32 for numerically stable CE
    pep_ohe_target = tf.one_hot(pep_ohe_indices, depth=vocab_size_ohe, dtype=tf.float32)
    mhc_ohe_target = tf.one_hot(mhc_ohe_indices, depth=vocab_size_ohe, dtype=tf.float32)
    # Masks are based on the original indices which correctly identify padding
    pep_mask = tf.where(pep_indices == PAD_INDEX, PAD_TOKEN, NORM_TOKEN)
    mhc_mask = tf.where(tf.reduce_all(tf.cast(mhc_emb, tf.float32) == PAD_VALUE, axis=-1), PAD_TOKEN, NORM_TOKEN)
    labels = tf.cast(parsed['label'], tf.int32)
    labels = tf.expand_dims(labels, axis=-1)
    return {
        "pep_blossom62": pep_blossom62_input,  # Shape: (..., 23)
        "pep_mask": pep_mask,
        "mhc_emb": mhc_emb,
        "mhc_mask": mhc_mask,
        "pep_ohe_target": pep_ohe_target,  # Shape: (..., 21)
        "mhc_ohe_target": mhc_ohe_target,  # Shape: (..., 21)
        "labels": labels
    }


def apply_dynamic_masking(features, emd_mask_d2=True):  # Added optional flag
    """
    Applies random masking for training augmentation inside the tf.data pipeline.
    This version is corrected to match the original DataGenerator logic.
    """
    # Peptide Masking
    valid_pep_positions = tf.where(tf.equal(features["pep_mask"], NORM_TOKEN))
    num_valid_pep = tf.shape(valid_pep_positions)[0]
    # --- CORRECTED MASK COUNT ---
    # At least 2 positions, or 15% of the valid sequence length
    num_to_mask_pep = tf.maximum(2, tf.cast(tf.cast(num_valid_pep, tf.float32) * 0.15, tf.int32))
    shuffled_pep_indices = tf.random.shuffle(valid_pep_positions)[:num_to_mask_pep]
    if tf.shape(shuffled_pep_indices)[0] > 0:
        # Update the mask to MASK_TOKEN (-1.0)
        features["pep_mask"] = tf.tensor_scatter_nd_update(features["pep_mask"], shuffled_pep_indices,
                                                           tf.repeat(MASK_TOKEN, num_to_mask_pep))
        # Zero out the feature values for the masked positions
        feat_dtype = features["pep_blossom62"].dtype
        mask_updates_pep = tf.fill([num_to_mask_pep, tf.shape(features["pep_blossom62"])[-1]],
                                   tf.cast(MASK_VALUE, feat_dtype))
        features["pep_blossom62"] = tf.tensor_scatter_nd_update(features["pep_blossom62"], shuffled_pep_indices,
                                                                mask_updates_pep)
    # MHC Masking
    valid_mhc_positions = tf.where(tf.equal(features["mhc_mask"], NORM_TOKEN))
    num_valid_mhc = tf.shape(valid_mhc_positions)[0]
    # --- CORRECTED MASK COUNT ---
    # At least 5 positions, or 15% of the valid sequence length
    num_to_mask_mhc = tf.maximum(10, tf.cast(tf.cast(num_valid_mhc, tf.float32) * 0.30, tf.int32))
    shuffled_mhc_indices = tf.random.shuffle(valid_mhc_positions)[:num_to_mask_mhc]
    if tf.shape(shuffled_mhc_indices)[0] > 0:
        # Update the mask to MASK_TOKEN (-1.0)
        features["mhc_mask"] = tf.tensor_scatter_nd_update(features["mhc_mask"], shuffled_mhc_indices,
                                                           tf.repeat(MASK_TOKEN, num_to_mask_mhc))
        # Zero out the feature values for the masked positions
        mhc_dtype = features["mhc_emb"].dtype
        mask_updates_mhc = tf.fill([num_to_mask_mhc, tf.shape(features["mhc_emb"])[-1]], tf.cast(MASK_VALUE, mhc_dtype))
        features["mhc_emb"] = tf.tensor_scatter_nd_update(features["mhc_emb"], shuffled_mhc_indices, mask_updates_mhc)
    # --- OPTIONAL: IMPLEMENTATION OF FEATURE-DIMENSION MASKING ---
    # This logic was in the original generator but is a distinct augmentation step.
    # It can be enabled if you find it improves model robustness.
    if emd_mask_d2:
        # Find positions that are STILL valid (not padded and not positionally masked)
        remaining_valid_mhc = tf.where(tf.equal(features["mhc_mask"], NORM_TOKEN))
        if tf.shape(remaining_valid_mhc)[0] > 0:
            # Get the embeddings at these remaining valid positions
            valid_embeddings = tf.gather_nd(features["mhc_emb"], remaining_valid_mhc)
            # Create a random mask for the feature dimensions
            dim_mask = tf.random.uniform(shape=tf.shape(valid_embeddings), dtype=features["mhc_emb"].dtype) < tf.cast(
                0.30, features["mhc_emb"].dtype)
            # Apply the mask (multiply by 0 where True, 1 where False)
            masked_embeddings = valid_embeddings * tf.cast(~dim_mask, features["mhc_emb"].dtype)
            # Scatter the modified embeddings back into the original tensor
            features["mhc_emb"] = tf.tensor_scatter_nd_update(features["mhc_emb"], remaining_valid_mhc,
                                                              masked_embeddings)
    return features

# You would then call this in your create_dataset function:
# dataset = dataset.map(lambda x: apply_dynamic_masking(x, emd_mask_d2=False), num_parallel_calls=tf.data.AUTOTUNE)

def create_dataset(filepath_pattern, batch_size, is_training=True, apply_masking=True):
    """Creates a tf.data.Dataset from a set of sharded, compressed TFRecord files."""
    file_list = tf.io.gfile.glob(filepath_pattern)
    if not file_list:
        raise ValueError(f"No files found for pattern: {filepath_pattern}")
    print(f"Creating dataset from {len(file_list)} TFRecord files for pattern: {filepath_pattern}")
    dataset = tf.data.TFRecordDataset(file_list, compression_type="GZIP", num_parallel_reads=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(buffer_size=50000, reshuffle_each_iteration=True)
    dataset = dataset.map(_parse_tf_example, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training and apply_masking:
        dataset = dataset.map(apply_dynamic_masking, num_parallel_calls=tf.data.AUTOTUNE)
    # Correct: `batch` has no num_parallel_calls
    dataset = dataset.batch(
        batch_size,
        drop_remainder=is_training
    )
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# ──────────────────────────────────────────────────────────────────────
# Training & Evaluation Steps
# ----------------------------------------------------------------------
@tf.function()
def train_step(model, batch_data, focal_loss_fn, optimizer, metrics, class_weights):
    """Compiled training step with proper mixed precision handling."""
    with tf.GradientTape() as tape:
        outputs = model(batch_data, training=True)
        # Compute individual losses
        raw_cls_loss = focal_loss_fn(batch_data["labels"], tf.cast(outputs["cls_ypred"], tf.float32))

        # Apply class weights manually
        labels_flat = tf.reshape(batch_data["labels"], [-1])
        sample_weights = tf.gather(class_weights, tf.cast(labels_flat, tf.int32))
        weighted_cls_loss = tf.reduce_mean(raw_cls_loss * sample_weights)

        raw_recon_loss_pep = masked_categorical_crossentropy(outputs["pep_ytrue_ypred"], batch_data["pep_mask"])
        raw_recon_loss_mhc = masked_categorical_crossentropy(outputs["mhc_ytrue_ypred"], batch_data["mhc_mask"])
        # Check for NaN/Inf in individual losses
        weighted_cls_loss = tf.where(tf.math.is_finite(weighted_cls_loss), weighted_cls_loss, 0.0)
        raw_recon_loss_pep = tf.where(tf.math.is_finite(raw_recon_loss_pep), raw_recon_loss_pep, 0.0)
        raw_recon_loss_mhc = tf.where(tf.math.is_finite(raw_recon_loss_mhc), raw_recon_loss_mhc, 0.0)
        # Balanced loss weighting for stability
        total_loss_weighted = (1.0 * weighted_cls_loss) + (0.2 * raw_recon_loss_pep) + (0.2 * raw_recon_loss_mhc)
        total_loss_weighted = tf.clip_by_value(total_loss_weighted, 0.0, 10.0)

        # Use proper LossScaleOptimizer methods for mixed precision
        if mixed_precision:
            scaled_loss = optimizer.scale_loss(total_loss_weighted)
            grads = tape.gradient(scaled_loss, model.trainable_variables)
            # Gradient unscaling is handled automatically by apply_gradients
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
    metrics['train_loss'].update_state(total_loss_weighted)
    metrics['pep_loss'].update_state(raw_recon_loss_pep)
    metrics['mhc_loss'].update_state(raw_recon_loss_mhc)
    metrics['cls_loss'].update_state(weighted_cls_loss)


@tf.function()
def eval_step(model, batch_data, focal_loss_fn, metrics):
    """Compiled evaluation step."""
    outputs = model(batch_data, training=False)
    labels_flat = tf.reshape(batch_data["labels"], [-1])
    preds_flat = tf.reshape(outputs["cls_ypred"], [-1])

    # Simple binary crossentropy for classification (no class weighting for validation)
    cls_loss = focal_loss_fn(batch_data["labels"], tf.cast(outputs["cls_ypred"], tf.float32))

    # Reconstruction losses using masked_categorical_crossentropy
    recon_loss_pep = masked_categorical_crossentropy(outputs["pep_ytrue_ypred"], batch_data["pep_mask"])
    recon_loss_mhc = masked_categorical_crossentropy(outputs["mhc_ytrue_ypred"], batch_data["mhc_mask"])

    # Total validation loss (same weighting as training but no class weights)
    total_val_loss = (1.0 * cls_loss) + (0.2 * recon_loss_pep) + (0.2 * recon_loss_mhc)

    metrics['val_acc'].update_state(labels_flat, tf.cast(preds_flat, tf.float32))
    metrics['val_auc'].update_state(labels_flat, tf.cast(preds_flat, tf.float32))
    metrics['val_precision'].update_state(labels_flat, tf.cast(preds_flat, tf.float32))
    metrics['val_loss'].update_state(total_val_loss)


# ──────────────────────────────────────────────────────────────────────
# Main Training Function
# ----------------------------------------------------------------------
def train(tfrecord_dir, out_dir, mhc_class, epochs, batch_size, lr, embed_dim, heads, noise_std, run_config,
          resume_from_weights=None, enable_masking=True, subset=1.0):
    """Fully optimized training function using the new data pipeline."""

    global MAX_PEP_LEN, MAX_MHC_LEN, ESM_DIM, MHC_CLASS
    MHC_CLASS = mhc_class

    with open(os.path.join(tfrecord_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    MAX_PEP_LEN, MAX_MHC_LEN, ESM_DIM, MHC_CLASS, train_samples, val_samples = metadata['MAX_PEP_LEN'], metadata[
        'MAX_MHC_LEN'], metadata['ESM_DIM'], metadata['MHC_CLASS'], metadata.get('train_samples',
                                                                                 None), metadata.get(
        'val_samples', None)

    # Apply subset to training samples
    if train_samples:
        original_train_samples = train_samples
        train_samples = int(train_samples * subset)
        val_samples = int(val_samples * subset) if val_samples else None
        if subset < 1.0:
            print(
                f"Using {subset * 100:.1f}% subset of training data: {train_samples:,} / {original_train_samples:,} samples")

    if train_samples and val_samples:
        train_steps = train_samples // batch_size
        val_steps = val_samples // batch_size
        print(f"Dataset size: {train_samples:,} train samples, {val_samples:,} val samples")
        print(f"Batches per epoch: {train_steps:,} train, {val_steps:,} val (batch_size={batch_size})")
        total_train_batches = train_steps * epochs
        print(f"Total training batches across {epochs} epochs: {total_train_batches:,}")
    else:
        print("Warning: Dataset sizes not found in metadata. Progress bar won't show totals.")
        train_steps = None
        val_steps = None

    lookup_path = os.path.join(tfrecord_dir, "train_mhc_embedding_lookup.npz")
    load_embedding_table(lookup_path)

    train_pattern = os.path.join(tfrecord_dir, "train_shard_*.tfrecord")
    val_pattern = os.path.join(tfrecord_dir, "validation_shard_*.tfrecord")

    # Create dataset with subset applied via take() if needed
    train_ds = create_dataset(train_pattern, batch_size, is_training=True, apply_masking=enable_masking)
    val_ds = create_dataset(val_pattern, batch_size, is_training=False, apply_masking=False)

    if subset < 1.0 and train_steps:
        train_ds = train_ds.take(train_steps)
        val_ds = val_ds.take(val_steps)

    model = pmbind(max_pep_len=MAX_PEP_LEN, max_mhc_len=MAX_MHC_LEN, emb_dim=embed_dim,
                   heads=heads, noise_std=noise_std, latent_dim=embed_dim * 2,
                   ESM_dim=ESM_DIM, drop_out_rate=0.3)

    model.build(train_ds.element_spec)

    if resume_from_weights:
        if os.path.exists(resume_from_weights):
            print(f"\nLoading weights from {resume_from_weights} to resume training...")
            model.load_weights(resume_from_weights)
            print("✓ Weights loaded successfully.")
        else:
            print(f"\nWarning: Weight file not found at {resume_from_weights}. Starting from scratch.")
    model.summary()

    # save model config
    config_data = model.get_config()
    run_config['training_subset'] = subset  # Add subset info to config
    run_config['validation_subset'] = subset
    run_config['MAX_PEP_LEN'] = MAX_PEP_LEN  # Add sequence length info
    run_config['MAX_MHC_LEN'] = MAX_MHC_LEN
    run_config['ESM_DIM'] = ESM_DIM
    run_config['MHC_CLASS'] = MHC_CLASS
    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        json.dump(config_data, f, indent=4)
    with open(os.path.join(out_dir, "run_config.json"), "a") as f:
        json.dump(run_config, f, indent=4)

    # Create cosine decay learning rate schedule that resets every epoch
    cosine_decay_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=lr,
        first_decay_steps=train_steps,  # Reset every epoch
        alpha=0.01,  # Minimum learning rate = 0.01 * initial_lr
        t_mul=1.0,  # Keep same cycle length
        m_mul=1.0   # Keep same initial learning rate
    )
    print(f"✓ Using CosineDecayRestarts schedule: initial_lr={lr}, first_decay_steps={train_steps}")

    # Create Lion optimizer with cosine decay schedule
    base_optimizer = keras.optimizers.Lion(learning_rate=cosine_decay_schedule)
    if mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            base_optimizer,
            #initial_scale=32768.0,  # Higher initial scale
            #dynamic_growth_steps=2000  # Adapt every 2000 steps
        )
        print("✓ Using LossScaleOptimizer wrapper for Lion with mixed precision")
    else:
        optimizer = base_optimizer

    # Best practice for class weights: inverse frequency weighting
    # pos_ratio = 0.02, neg_ratio = 0.98
    # n_samples / (n_classes * class_frequency)
    pos_weight = 1.0 / (2 * 0.02)  # = 25.0
    neg_weight = 1.0 / (2 * 0.98)  # = 0.51

    # Normalize so negative class weight = 1.0
    class_weight_dict = {0: 1.0, 1: pos_weight / neg_weight}  # {0: 1.0, 1: 49.0}
    print(f"✓ Using class weights: {class_weight_dict}")

    focal_loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
        from_logits=False,
        reduction="none",  # Use 'none' to apply manual weighting
        label_smoothing=0.1,
        gamma=2.0,
        apply_class_balancing=False
        )

    # Convert class weights to tensor for tf.function compatibility
    class_weights_tensor = tf.constant([class_weight_dict[0], class_weight_dict[1]], dtype=tf.float32)
    metrics = {
        'train_loss': tf.keras.metrics.Mean(name='train_loss'),
        'train_auc': tf.keras.metrics.AUC(name='train_auc'),
        'train_acc': tf.keras.metrics.BinaryAccuracy(name='train_accuracy'),
        'train_precision': tf.keras.metrics.Precision(name='train_precision'),
        'val_auc': tf.keras.metrics.AUC(name='val_auc'),
        'val_acc': tf.keras.metrics.BinaryAccuracy(name='val_accuracy'),
        'val_precision': tf.keras.metrics.Precision(name='val_precision'),
        'val_loss': tf.keras.metrics.Mean(name='val_loss'),
        'pep_loss': tf.keras.metrics.Mean(name='pep_loss'),
        'mhc_loss': tf.keras.metrics.Mean(name='mhc_loss'),
        'cls_loss': tf.keras.metrics.Mean(name='cls_loss'),
    }

    history = {k: [] for k in ["train_loss", "train_auc", "train_acc", "train_precision",
                               "val_auc", "val_acc", "val_precision", "val_loss",
                               "cls_loss", "pep_loss", "mhc_loss"]}
    history['subset'] = subset  # Store subset info in history
    best_val_auc = 0.0

    for epoch in range(epochs):
        print(f"\n{'=' * 60}\nEpoch {epoch + 1}/{epochs}\n{'=' * 60}")
        for m in metrics.values(): m.reset_state()

        # Use dataset length divided by batch size for progress bar
        pbar = tqdm(train_ds, desc="Training", unit="batch", total=train_steps)
        for batch_data in pbar:
            train_step(model, batch_data, focal_loss_fn, optimizer, metrics, class_weights_tensor)

            # Update progress bar with current metrics
            pbar.set_postfix({
                'Loss': f"{metrics['train_loss'].result():.4f}",
                'AUC': f"{metrics['train_auc'].result():.4f}",
                'Acc': f"{metrics['train_acc'].result():.4f}",
                'Prec': f"{metrics['train_precision'].result():.4f}",
                'PepL': f"{metrics['pep_loss'].result():.4f}",
                'MhcL': f"{metrics['mhc_loss'].result():.4f}"
            })

        # Validation
        for batch_data in tqdm(val_ds, desc="Validating", unit="batch", total=val_steps):
            eval_step(model, batch_data, metrics)

        print(f"\n  Epoch Summary -> "
              f"Train Loss: {metrics['train_loss'].result():.4f}, "
              f"Train AUC: {metrics['train_auc'].result():.4f}, "
              f"Train Prec: {metrics['train_precision'].result():.4f}, "
              f"Val Loss: {metrics['val_loss'].result():.4f}, "
              f"Val AUC: {metrics['val_auc'].result():.4f}, "
              f"Val Prec: {metrics['val_precision'].result():.4f}")

        # Log history
        history['train_loss'].append(float(metrics['train_loss'].result()))
        history['train_auc'].append(float(metrics['train_auc'].result()))
        history['train_acc'].append(float(metrics['train_acc'].result()))
        history['train_precision'].append(float(metrics['train_precision'].result()))
        history['val_auc'].append(float(metrics['val_auc'].result()))
        history['val_acc'].append(float(metrics['val_acc'].result()))
        history['val_precision'].append(float(metrics['val_precision'].result()))
        history['val_loss'].append(float(metrics['val_loss'].result()))
        history['cls_loss'].append(float(metrics['cls_loss'].result()))
        history['pep_loss'].append(float(metrics['pep_loss'].result()))
        history['mhc_loss'].append(float(metrics['mhc_loss'].result()))

        current_val_auc = metrics['val_auc'].result()
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            model.save_weights(os.path.join(out_dir, "best_model.weights.h5"))
            print(f"  -> New best model saved with Val AUC: {best_val_auc:.4f}")

    print("\nTraining finished. Saving final model and generating visualizations...")
    model.save_weights(os.path.join(out_dir, "final_model.weights.h5"))
    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    visualize_training_history(history, out_dir)
    print(f"✓ Training complete. Artifacts saved in {out_dir}")


# ──────────────────────────────────────────────────────────────────────
# Main Execution
# ----------------------------------------------------------------------
def main(args):
    """Main function to run the training pipeline."""
    RUN_CONFIG = {
        "MHC_CLASS": 1, "EPOCHS": 3, "BATCH_SIZE": 512, "LEARNING_RATE": 1e-3,
        "EMBED_DIM": 32, "HEADS": 2, "NOISE_STD": 0.4,
        "description": "Optimized run with tf.data pipeline and mixed precision"
    }

    base_output_folder = "../results/PMBind_runs_optimized/"
    run_id_base = 0

    fold_to_run = args.fold  # Get fold from args
    run_id = run_id_base + fold_to_run
    run_name = f"run_{run_id}_mhc{RUN_CONFIG['MHC_CLASS']}_dim{RUN_CONFIG['EMBED_DIM']}_h{RUN_CONFIG['HEADS']}_fold{fold_to_run}"
    out_dir = os.path.join(base_output_folder, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Starting run: {run_name}\nOutput directory: {out_dir}")

    tfrecord_dir = f"../data/cross_validation_dataset/mhc{MHC_CLASS}/tfrecords/fold_{fold_to_run:02d}/"

    if not os.path.exists(tfrecord_dir) or not os.path.exists(os.path.join(tfrecord_dir, 'metadata.json')):
        print(f"Error: TFRecord directory not found or is incomplete: {tfrecord_dir}")
        print("Please run the `create_tfrecords.py` script first.")
        sys.exit(1)

    with open(os.path.join(out_dir, "config.json"), 'w') as f:
        json.dump(RUN_CONFIG, f, indent=4)

    train(
        tfrecord_dir=tfrecord_dir,
        out_dir=out_dir,
        mhc_class=RUN_CONFIG["MHC_CLASS"],
        epochs=RUN_CONFIG["EPOCHS"],
        batch_size=RUN_CONFIG["BATCH_SIZE"],
        lr=RUN_CONFIG["LEARNING_RATE"],
        embed_dim=RUN_CONFIG["EMBED_DIM"],
        heads=RUN_CONFIG["HEADS"],
        noise_std=RUN_CONFIG["NOISE_STD"],
        resume_from_weights=args.resume_from,
        enable_masking=True,
        subset=args.subset,
        run_config=RUN_CONFIG
    )


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth for each GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    try:
        tf.config.experimental.enable_tensor_float_32_execution(True)
    except:
        pass
    parser = argparse.ArgumentParser(description="Run training for specified folds.")
    parser.add_argument("--fold", type=int, required=True, help="Fold number to run (e.g., 0-4).")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to model weights (.h5 file) to resume training from.")
    parser.add_argument("--subset", type=float, default=1.0, help="Subset percentage of training data to use.")
    args = parser.parse_args()
    main(args)