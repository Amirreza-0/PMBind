#!/usr/bin/env python
"""
Optimized GradientTape training and inference script for pmbind_multitask model.

Key optimizations:
- Fixed test_auc display issue in progress bar
- Added @tf.function compilation for training step
- Optimized DataGenerator with caching and vectorized operations
- Removed redundant computations
- Improved memory efficiency
- Cleaner code structure
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
from functools import lru_cache

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score)

# Local imports
from utils import (seq_to_onehot, get_embed_key, NORM_TOKEN, MASK_TOKEN, PAD_TOKEN, PAD_VALUE, MASK_VALUE,
                   clean_key, masked_categorical_crossentropy, seq_to_blossom62, split_y_true_y_pred, OHE_to_seq_single,
                   peptide_properties_biopython, cn_terminal_amino_acids, reduced_anchor_pair)
from models import pmbind_multitask as pmbind
from visualizations import _analyze_latents

# ──────────────────────────────────────────────────────────────────────
# Globals & Configuration
# ----------------------------------------------------------------------
EMB_DB: np.lib.npyio.NpzFile | None = None
MHC_CLASS = 1
ESM_DIM = 1536


def load_embedding_db(npz_path: str):
    """Load embedding database with memory mapping."""
    return np.load(npz_path, mmap_mode="r")


class OptimizedDataGenerator(keras.utils.Sequence):
    """Optimized data generator with caching and vectorized operations."""

    def __init__(self, df, seq_map, embed_map, max_pep_len, max_mhc_len, batch_size,
                 is_training=True, shuffle=True, negs_pool=None, pos_df=None):
        super().__init__()
        self.df = df
        self.seq_map = seq_map
        self.embed_map = embed_map
        self.max_pep_len = max_pep_len
        self.max_mhc_len = max_mhc_len
        self.batch_size = batch_size
        self.is_training = is_training
        self.shuffle = shuffle
        self.negs_pool = negs_pool
        self.pos_df = pos_df

        # Pre-process and cache frequently used data
        self._preprocess_data()
        self.on_epoch_end()

    def _preprocess_data(self):
        """Pre-process data for faster access."""
        # Cache cleaned keys
        self.df['_cleaned_key'] = self.df.apply(
            lambda r: r.get('mhc_embedding_key', r['allele'].replace(' ', '').replace('*', '').replace(':', '')),
            axis=1
        )

        # Pre-compute embedding keys
        self.df['_emb_key'] = self.df['_cleaned_key'].apply(
            lambda k: get_embed_key(clean_key(k), self.embed_map)
        )

        # Cache MHC sequences
        if MHC_CLASS == 2:
            self.df['_mhc_seq'] = self.df['_cleaned_key'].apply(self._get_mhc_seq_class2)
        else:
            self.df['_mhc_seq'] = self.df['_emb_key'].apply(
                lambda k: self.seq_map.get(get_embed_key(clean_key(k), self.seq_map), '')
            )

    def _get_mhc_seq_class2(self, key):
        """Get MHC sequence for class 2."""
        key_parts = key.split('_')
        if len(key_parts) >= 2:
            key1 = get_embed_key(clean_key(key_parts[0]), self.seq_map)
            key2 = get_embed_key(clean_key(key_parts[1]), self.seq_map)
            return self.seq_map.get(key1, '') + self.seq_map.get(key2, '')
        return ''

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.df))
        batch_indices = self.indices[start_idx:end_idx]
        batch_df = self.df.iloc[batch_indices]
        return self._generate_batch(batch_df)

    def on_epoch_end(self):
        """Resample negatives and shuffle data."""
        if self.is_training and self.negs_pool is not None and self.pos_df is not None:
            n_neg_samples = len(self.df) - len(self.pos_df)
            df_neg_resampled = self.negs_pool.sample(n=n_neg_samples, replace=True)
            # Pre-process resampled negatives
            df_neg_resampled = df_neg_resampled.copy()
            df_neg_resampled['_cleaned_key'] = df_neg_resampled.apply(
                lambda r: r.get('mhc_embedding_key', r['allele'].replace(' ', '').replace('*', '').replace(':', '')),
                axis=1
            )
            df_neg_resampled['_emb_key'] = df_neg_resampled['_cleaned_key'].apply(
                lambda k: get_embed_key(clean_key(k), self.embed_map)
            )
            if MHC_CLASS == 2:
                df_neg_resampled['_mhc_seq'] = df_neg_resampled['_cleaned_key'].apply(self._get_mhc_seq_class2)
            else:
                df_neg_resampled['_mhc_seq'] = df_neg_resampled['_emb_key'].apply(
                    lambda k: self.seq_map.get(get_embed_key(clean_key(k), self.seq_map), '')
                )
            self.df = pd.concat([self.pos_df, df_neg_resampled], ignore_index=True)

        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    @lru_cache(maxsize=128)
    def _get_embedding(self, emb_key, cleaned_key):
        """Cached embedding retrieval."""
        if MHC_CLASS == 2:
            key_parts = cleaned_key.split('_')
            if len(key_parts) >= 2:
                embd_key1 = get_embed_key(clean_key(key_parts[0]), self.embed_map)
                embd_key2 = get_embed_key(clean_key(key_parts[1]), self.embed_map)
                emb1, emb2 = EMB_DB[embd_key1], EMB_DB[embd_key2]
                return np.concatenate([emb1, emb2], axis=0)
        return EMB_DB[emb_key]

    def _generate_batch(self, batch_df):
        """Generate batch with vectorized operations."""
        n = len(batch_df)

        # Initialize batch arrays
        batch_data = {
            "pep_blossom62": np.zeros((n, self.max_pep_len, 23), np.float32),
            "pep_mask": np.full((n, self.max_pep_len), PAD_TOKEN, dtype=np.float32),
            "mhc_emb": np.zeros((n, self.max_mhc_len, ESM_DIM), np.float32),
            "mhc_mask": np.full((n, self.max_mhc_len), PAD_TOKEN, dtype=np.float32),
            "pep_ohe_target": np.zeros((n, self.max_pep_len, 21), np.float32),
            "mhc_ohe_target": np.zeros((n, self.max_mhc_len, 21), np.float32),
            "labels": np.zeros((n, 1), np.int32),
        }

        # Process batch in vectorized manner where possible
        for i, (_, row) in enumerate(batch_df.iterrows()):
            # Peptide processing
            pep_seq = row['long_mer'].upper()
            pep_len = len(pep_seq)

            batch_data["pep_blossom62"][i] = seq_to_blossom62(pep_seq, max_seq_len=self.max_pep_len)
            batch_data["pep_ohe_target"][i] = seq_to_onehot(pep_seq, max_seq_len=self.max_pep_len)
            batch_data["pep_mask"][i, :pep_len] = NORM_TOKEN

            # Apply masking for training
            if self.is_training and pep_len > 0:
                mask_fraction = 0.15
                n_mask = max(1, int(mask_fraction * pep_len))
                mask_indices = np.random.choice(pep_len, size=n_mask, replace=False)
                batch_data["pep_mask"][i, mask_indices] = MASK_TOKEN
                batch_data["pep_blossom62"][i, mask_indices, :] = MASK_VALUE

            # MHC processing
            emb = self._get_embedding(row['_emb_key'], row['_cleaned_key'])
            L = emb.shape[0]
            batch_data["mhc_emb"][i, :L] = emb
            batch_data["mhc_mask"][i, :L] = NORM_TOKEN

            # Apply MHC masking for training
            if self.is_training and L > 0:
                mask_fraction = 0.30
                n_mask = max(1, int(mask_fraction * L))
                mask_indices = np.random.choice(L, size=n_mask, replace=False)
                batch_data["mhc_mask"][i, mask_indices] = MASK_TOKEN
                batch_data["mhc_emb"][i, mask_indices, :] = MASK_VALUE

            # MHC sequence target
            mhc_seq = row['_mhc_seq']
            batch_data["mhc_ohe_target"][i] = seq_to_onehot(mhc_seq, max_seq_len=self.max_mhc_len)
            batch_data["labels"][i, 0] = int(row['assigned_label'])

        return {k: tf.convert_to_tensor(v) for k, v in batch_data.items()}


# ──────────────────────────────────────────────────────────────────────
# Training Functions
# ----------------------------------------------------------------------
@tf.function
def train_step(model, batch_data, focal_loss_fn, log_vars, optimizer):
    """Compiled training step for better performance."""
    log_var_cls, log_var_pep, log_var_mhc = log_vars

    with tf.GradientTape() as tape:
        outputs = model(batch_data, training=True)

        # Classification loss with focal loss
        raw_cls_loss = focal_loss_fn(batch_data["labels"], outputs["cls_ypred"])
        precision_cls = tf.exp(-log_var_cls)
        # By applying softplus to log_var_cls, we ensure the regularization
        # term is always non-negative, thus preventing the total cls_loss
        # from becoming negative.
        # softplus(x) = log(exp(x) + 1)
        cls_loss = precision_cls * raw_cls_loss + tf.nn.softplus(log_var_cls)

        # Reconstruction losses
        raw_recon_loss_pep = masked_categorical_crossentropy(outputs["pep_ytrue_ypred"], batch_data["pep_mask"])
        precision_pep = tf.exp(-log_var_pep)
        recon_loss_pep = precision_pep * raw_recon_loss_pep + 2*tf.nn.softplus(log_var_pep)

        raw_recon_loss_mhc = masked_categorical_crossentropy(outputs["mhc_ytrue_ypred"], batch_data["mhc_mask"])
        precision_mhc = tf.exp(-log_var_mhc)
        recon_loss_mhc = precision_mhc * raw_recon_loss_mhc + tf.nn.softplus(log_var_mhc)

        total_loss = cls_loss + recon_loss_pep + recon_loss_mhc

    trainable_vars = model.trainable_variables + list(log_vars)
    grads = tape.gradient(total_loss, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars))

    return {
        'total_loss': total_loss,
        'cls_loss': raw_cls_loss,
        'recon_loss_pep': raw_recon_loss_pep,
        'recon_loss_mhc': raw_recon_loss_mhc
    }


def evaluate_model_batched(model, generator):
    """Evaluate model over full dataset with batched processing."""
    y_true, y_pred = [], []

    for batch in generator:
        outputs = model(batch, training=False)
        y_true.append(batch["labels"].numpy())
        y_pred.append(outputs["cls_ypred"].numpy())

    y_true = np.concatenate(y_true).ravel()
    y_pred = np.concatenate(y_pred).ravel()

    return roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float('nan')


def train(train_path, validation_path, test_path, negs_pool, embed_npz, seq_csv, embd_key_path, out_dir,
          mhc_class, epochs, batch_size, lr, embed_dim, heads, noise_std):
    """Optimized training function."""
    global EMB_DB, MHC_CLASS, ESM_DIM
    EMB_DB = load_embedding_db(embed_npz)
    MHC_CLASS = mhc_class
    ESM_DIM = int(next(iter(EMB_DB.values())).shape[1])

    # Load data
    seq_map = pd.read_csv(seq_csv, index_col="allele")["mhc_sequence"].to_dict()
    embed_map = pd.read_csv(embd_key_path, index_col="key")["mhc_sequence"].to_dict()
    seq_map = {clean_key(k): v for k, v in seq_map.items()}

    df_train = pq.ParquetFile(train_path).read().to_pandas()
    df_val = pq.ParquetFile(validation_path).read().to_pandas()
    df_test = pq.ParquetFile(test_path).read().to_pandas()

    # Load negative samples pool
    if not os.path.exists(negs_pool):
        raise FileNotFoundError(f"Negative samples pool not found at: {negs_pool}")
    df_neg_pool = pq.ParquetFile(negs_pool).read().to_pandas()

    # Separate positive samples
    df_train_pos = df_train[df_train['assigned_label'] == 1].copy()
    n_neg_samples = len(df_train) - len(df_train_pos)

    print(f"Loaded {len(df_train)} training, {len(df_val)} validation, {len(df_test)} test samples.")
    print(f"Training has {len(df_train_pos)} positive and {n_neg_samples} negative samples.")

    # Calculate max lengths
    max_pep_len = int(pd.concat([df_train["long_mer"], df_val["long_mer"], df_test["long_mer"]]).str.len().max())
    max_mhc_len = 500 if mhc_class == 2 else int(next(iter(EMB_DB.values())).shape[0])
    print(f"Max peptide length: {max_pep_len}, Max MHC length: {max_mhc_len}")

    # Initialize model
    model = pmbind(
        max_pep_len=max_pep_len, max_mhc_len=max_mhc_len, emb_dim=embed_dim,
        heads=heads, noise_std=noise_std, transformer_layers=2,
        latent_dim=embed_dim * 2, ESM_dim=ESM_DIM
    )

    # Learnable loss weights
    log_var_cls = tf.Variable(0.0, trainable=True, name='log_var_cls')
    log_var_pep = tf.Variable(0.0, trainable=True, name='log_var_pep')
    log_var_mhc = tf.Variable(0.0, trainable=True, name='log_var_mhc')
    log_vars = [log_var_cls, log_var_pep, log_var_mhc]

    # Optimizer with cosine decay
    num_train_steps = (len(df_train) // batch_size) * epochs
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr, decay_steps=num_train_steps, alpha=1e-6
    )
    optimizer = keras.optimizers.Lion(learning_rate=lr_schedule)

    # Loss function
    focal_loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
        from_logits=False, reduction="sum_over_batch_size", alpha=0.25, gamma=2.0
    )

    # Build model
    dummy_gen = OptimizedDataGenerator(
        df=df_train.head(1), seq_map=seq_map, embed_map=embed_map,
        max_pep_len=max_pep_len, max_mhc_len=max_mhc_len,
        batch_size=1, is_training=False, shuffle=False
    )
    dummy_data = dummy_gen[0]
    model(dummy_data)
    model.summary()

    # Create validation generators
    val_gen = OptimizedDataGenerator(
        df=df_val, seq_map=seq_map, embed_map=embed_map,
        max_pep_len=max_pep_len, max_mhc_len=max_mhc_len,
        batch_size=batch_size, is_training=False, shuffle=False
    )

    test_gen = OptimizedDataGenerator(
        df=df_test, seq_map=seq_map, embed_map=embed_map,
        max_pep_len=max_pep_len, max_mhc_len=max_mhc_len,
        batch_size=batch_size, is_training=False, shuffle=False
    )

    # Pre-process positive training samples
    df_train_pos = df_train_pos.copy()
    df_train_pos['_cleaned_key'] = df_train_pos.apply(
        lambda r: r.get('mhc_embedding_key', r['allele'].replace(' ', '').replace('*', '').replace(':', '')),
        axis=1
    )
    df_train_pos['_emb_key'] = df_train_pos['_cleaned_key'].apply(
        lambda k: get_embed_key(clean_key(k), embed_map)
    )
    if MHC_CLASS == 2:
        df_train_pos['_mhc_seq'] = df_train_pos['_cleaned_key'].apply(
            lambda k: OptimizedDataGenerator(None, seq_map, embed_map, 0, 0, 0)._get_mhc_seq_class2(k)
        )
    else:
        df_train_pos['_mhc_seq'] = df_train_pos['_emb_key'].apply(
            lambda k: seq_map.get(get_embed_key(clean_key(k), seq_map), '')
        )

    # Training history
    history = {
        "train_loss": [], "val_loss": [], "test_loss": [],
        "train_auc": [], "val_auc": [], "test_auc": [],
        "train_cls_loss": [], "val_cls_loss": [],
        "train_recon_loss": [], "val_recon_loss": []
    }

    best_val_auc = 0.0
    os.makedirs(out_dir, exist_ok=True)

    # Training metrics
    train_loss_metric = tf.keras.metrics.Mean()
    train_auc_metric = tf.keras.metrics.AUC()
    pep_loss = tf.keras.metrics.Mean()
    mhc_loss = tf.keras.metrics.Mean()
    cls_loss = tf.keras.metrics.Mean()

    # Training loop
    for epoch in range(epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'=' * 60}")

        # Create training generator with resampled negatives
        train_gen = OptimizedDataGenerator(
            df=df_train, seq_map=seq_map, embed_map=embed_map,
            max_pep_len=max_pep_len, max_mhc_len=max_mhc_len,
            batch_size=batch_size, is_training=True, shuffle=True,
            negs_pool=df_neg_pool, pos_df=df_train_pos
        )

        # Reset metrics
        train_loss_metric.reset_state()
        train_auc_metric.reset_state()
        pep_loss.reset_state()
        mhc_loss.reset_state()
        cls_loss.reset_state()


        # Get previous epoch's validation AUC for display
        prev_val_auc = history['val_auc'][-1] if history['val_auc'] else 0.0
        prev_test_auc = history['test_auc'][-1] if history['test_auc'] else 0.0

        # Training progress bar
        pbar = tqdm(range(len(train_gen)), desc=f"Training", file=sys.stdout)
        pbar.set_postfix({'prev_val_auc': f"{prev_val_auc:.4f}", 'prev_test_auc': f"{prev_test_auc:.4f}"})

        for batch_idx in pbar:
            batch_data = train_gen[batch_idx]

            # Execute training step
            losses = train_step(model, batch_data, focal_loss_fn, log_vars, optimizer)

            # Update metrics
            train_loss_metric(losses['total_loss'])
            train_auc_metric(batch_data["labels"], model(batch_data, training=False)["cls_ypred"])
            pep_loss(losses['recon_loss_pep'])
            mhc_loss(losses['recon_loss_mhc'])
            cls_loss(losses['cls_loss'])

            # Update progress bar
            pbar.set_postfix({
                'tot_loss': f"{train_loss_metric.result():.4f}",
                'train_auc': f"{train_auc_metric.result():.4f}",
                'prev_val_auc': f"{prev_val_auc:.4f}",
                'prev_test_auc': f"{prev_test_auc:.4f}",
                'pep_rec': f"{pep_loss.result():.4f}",
                'mhc_rec': f"{mhc_loss.result():.4f}",
                'cls_loss': f"{cls_loss.result():.4f}"
            })

        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        val_auc = evaluate_model_batched(model, val_gen)

        # Evaluate on test set
        print("Evaluating on test set...")
        test_auc = evaluate_model_batched(model, test_gen)

        # Record metrics
        history['train_loss'].append(float(train_loss_metric.result()))
        history['train_auc'].append(float(train_auc_metric.result()))
        history['val_auc'].append(float(val_auc))
        history['test_auc'].append(float(test_auc))
        history['train_cls_loss'].append(float(cls_loss.result()))
        history["train_pep_recon_loss"] = history.get("train_pep_recon_loss", []) + [float(pep_loss.result())]
        history["train_mhc_recon_loss"] = history.get("train_mhc_recon_loss", []) + [float(mhc_loss.result())]

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {history['train_loss'][-1]:.4f}, Train AUC: {history['train_auc'][-1]:.4f}, Cls Loss: {history['train_cls_loss'][-1]:.4f}, Pep Recon Loss: {history['train_pep_recon_loss'][-1]:.4f}, MHC Recon Loss: {history['train_mhc_recon_loss'][-1]:.4f}")
        print(f"  Val AUC: {val_auc:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        print(
            f"  Log Variances - Cls: {log_var_cls.numpy():.4f}, Pep: {log_var_pep.numpy():.4f}, MHC: {log_var_mhc.numpy():.4f}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            model.save_weights(os.path.join(out_dir, "best_model.weights.h5"))
            print(f"  -> Saved best model with Val AUC: {best_val_auc:.4f}")

    # Save final model and history
    model.save_weights(os.path.join(out_dir, "final_model.weights.h5"))
    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    print("\nTraining complete!")
    visualize_training_history(history, out_dir)

    return max_pep_len, max_mhc_len, seq_map, embed_map


# ──────────────────────────────────────────────────────────────────────
# Inference Functions
# ----------------------------------------------------------------------
def infer(model_weights_path, data_path, out_dir, name,
          max_pep_len, max_mhc_len, seq_map, embed_map,
          mhc_class, embed_dim, heads, noise_std, batch_size, source_col=None):
    """Optimized inference function."""
    global MHC_CLASS, ESM_DIM
    MHC_CLASS = mhc_class
    ESM_DIM = int(next(iter(EMB_DB.values())).shape[1])

    df_infer = pq.ParquetFile(data_path).read().to_pandas()
    print(f"\nRunning inference on {name} set ({len(df_infer)} samples)...")
    os.makedirs(out_dir, exist_ok=True)

    # Initialize model
    model = pmbind(
        max_pep_len=max_pep_len, max_mhc_len=max_mhc_len, emb_dim=embed_dim,
        heads=heads, noise_std=noise_std, transformer_layers=2,
        latent_dim=embed_dim * 2, ESM_dim=ESM_DIM
    )

    # Build and load weights
    dummy_gen = OptimizedDataGenerator(
        df=df_infer.head(1), seq_map=seq_map, embed_map=embed_map,
        max_pep_len=max_pep_len, max_mhc_len=max_mhc_len,
        batch_size=1, is_training=False, shuffle=False
    )
    model(dummy_gen[0], training=False)
    model.load_weights(model_weights_path)

    # Check if latents already exist
    latents_seq_path = os.path.join(out_dir, f"latents_seq_{name}.mmap")
    latents_pooled_path = os.path.join(out_dir, f"latents_pooled_{name}.mmap")

    if os.path.exists(latents_pooled_path) and os.path.exists(latents_seq_path):
        print(f"Latents already exist, loading from disk...")
        latents_seq = np.memmap(latents_seq_path, dtype='float32', mode='r',
                                shape=(len(df_infer), max_pep_len + max_mhc_len, embed_dim))
        latents_pooled = np.memmap(latents_pooled_path, dtype='float32', mode='r',
                                   shape=(len(df_infer), max_pep_len + max_mhc_len + embed_dim))
        # Load predictions from saved CSV
        output_path = os.path.join(out_dir, f"inference_results_{name}.csv")
        if os.path.exists(output_path):
            df_infer = pd.read_csv(output_path)
            all_predictions = df_infer["prediction_score"].values
            all_labels = df_infer["assigned_label"].values if "assigned_label" in df_infer.columns else None
    else:
        # Create memory-mapped arrays for latents
        latents_seq = np.memmap(latents_seq_path, dtype='float32', mode='w+',
                                shape=(len(df_infer), max_pep_len + max_mhc_len, embed_dim))
        latents_pooled = np.memmap(latents_pooled_path, dtype='float32', mode='w+',
                                   shape=(len(df_infer), max_pep_len + max_mhc_len + embed_dim))

        # Create inference generator
        infer_gen = OptimizedDataGenerator(
            df=df_infer, seq_map=seq_map, embed_map=embed_map,
            max_pep_len=max_pep_len, max_mhc_len=max_mhc_len,
            batch_size=batch_size, is_training=False, shuffle=False
        )

        # Run inference
        all_predictions, all_labels = [], []
        pbar = tqdm(range(len(infer_gen)), desc=f"Inference on {name}", file=sys.stdout)

        for batch_idx in pbar:
            batch_data = infer_gen[batch_idx]
            outputs = model(batch_data, training=False)

            # Collect predictions
            all_predictions.append(outputs["cls_ypred"].numpy())
            all_labels.append(batch_data["labels"].numpy())

            # Store latents
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_data["labels"].shape[0]
            latents_seq[start_idx:end_idx] = outputs["latent_seq"].numpy()
            latents_pooled[start_idx:end_idx] = outputs["latent_vector"].numpy()

        # Flush memory-mapped arrays
        latents_seq.flush()
        latents_pooled.flush()

        # Process results
        all_predictions = np.concatenate(all_predictions, axis=0).squeeze()
        all_labels = np.concatenate(all_labels, axis=0).squeeze()

        # Save predictions
        df_infer["prediction_score"] = all_predictions
        df_infer["prediction_label"] = (all_predictions >= 0.5).astype(int)

        output_path = os.path.join(out_dir, f"inference_results_{name}.csv")
        df_infer.to_csv(output_path, index=False)
        print(f"✓ Inference results saved to {output_path}")

    # Generate visualizations if labels are available
    if "assigned_label" in df_infer.columns and all_labels is not None:
        visualize_inference_results(df_infer, all_labels, all_predictions, out_dir, name)

    # Run latent space visualizations
    vis_out_dir = os.path.join(out_dir, "visualizations")
    highlight_mask = None
    if source_col and source_col in df_infer.columns:
        highlight_mask = (df_infer[source_col] == 'test').values

    run_visualizations(
        df=df_infer, latents_seq=latents_seq, latents_pooled=latents_pooled,
        enc_dec=model, max_pep_len=max_pep_len, max_mhc_len=max_mhc_len,
        seq_map=seq_map, embed_map=embed_map, out_dir=vis_out_dir,
        dataset_name=name, figsize=(30, 15), point_size=3,
        highlight_mask=highlight_mask
    )

    # Append to summary
    if "assigned_label" in df_infer.columns:
        auc = roc_auc_score(all_labels, all_predictions) if len(np.unique(all_labels)) > 1 else float('nan')
        ap = average_precision_score(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, df_infer["prediction_label"])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        summary_path = os.path.join(os.path.dirname(out_dir), "inference_summary.csv")
        with open(summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['dataset', 'num_samples', 'num_positives', 'num_negatives',
                                 'AUC', 'AP', 'TP', 'TN', 'FP', 'FN'])
            writer.writerow([
                name, len(df_infer),
                int((df_infer['assigned_label'] == 1).sum()),
                int((df_infer['assigned_label'] == 0).sum()),
                f"{auc:.4f}", f"{ap:.4f}", tp, tn, fp, fn
            ])


# ──────────────────────────────────────────────────────────────────────
# Visualization Functions (kept same as original)
# ----------------------------------------------------------------------
def visualize_training_history(history, out_dir):
    """Generate training history plots."""
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        plt.plot(history["val_loss"], label="Val Loss")
    if "test_loss" in history and history["test_loss"]:
        plt.plot(history["test_loss"], label="Test Loss", linestyle='--', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training, Validation, and Test Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot AUC
    plt.subplot(1, 2, 2)
    plt.plot(history["train_auc"], label="Train AUC")
    plt.plot(history["val_auc"], label="Val AUC")
    if "test_auc" in history and history["test_auc"]:
        plt.plot(history["test_auc"], label="Test AUC", linestyle='--', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Training, Validation, and Test AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_history.png"))
    plt.close()


def visualize_inference_results(df_with_results, true_labels, prediction_scores, out_dir, name):
    """Generate comprehensive evaluation plots."""
    print(f"Generating visualizations for {name} set...")

    # Confusion Matrix
    cm = confusion_matrix(true_labels, df_with_results["prediction_label"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"Confusion Matrix on {name} Set")
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{name}.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, prediction_scores)
    auc = roc_auc_score(true_labels, prediction_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve on {name} Set')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(out_dir, f"roc_curve_{name}.png"))
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(true_labels, prediction_scores)
    ap = average_precision_score(true_labels, prediction_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve on {name} Set')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(out_dir, f"precision_recall_curve_{name}.png"))
    plt.close()

    # Score Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_with_results, x='prediction_score', hue='assigned_label', kde=True, bins=50)
    plt.title(f'Prediction Score Distribution on {name} Set')
    plt.savefig(os.path.join(out_dir, f"score_distribution_{name}.png"))
    plt.close()


def run_visualizations(df, latents_seq, latents_pooled, enc_dec, max_pep_len, max_mhc_len,
                       seq_map, embed_map, out_dir, dataset_name, highlight_mask=None,
                       figsize=(40, 15), point_size=2):
    """Generate latent space visualizations."""
    print("\nGenerating latent space visualizations...")
    os.makedirs(out_dir, exist_ok=True)

    alleles = df['allele'].apply(clean_key).astype('category')
    unique_alleles = alleles.cat.categories
    num_to_highlight = min(5, len(unique_alleles))
    np.random.seed(999)
    random_alleles_to_highlight = np.random.choice(unique_alleles, num_to_highlight, replace=False).tolist()

    if len(unique_alleles) <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_alleles)))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_alleles)))
    allele_color_map = {allele: color for allele, color in zip(unique_alleles, colors)}

    # Analyze sequential latents
    df_seq = _analyze_latents(
        latents=latents_seq, df=df, alleles=alleles, allele_color_map=allele_color_map,
        random_alleles_to_highlight=random_alleles_to_highlight, latent_type="seq",
        out_dir=out_dir, dataset_name=dataset_name, highlight_mask=highlight_mask,
        figsize=figsize, point_size=point_size
    )

    # Analyze pooled latents
    df_pooled = _analyze_latents(
        latents=latents_pooled, df=df, alleles=alleles, allele_color_map=allele_color_map,
        random_alleles_to_highlight=random_alleles_to_highlight, latent_type="pooled",
        out_dir=out_dir, dataset_name=dataset_name, highlight_mask=highlight_mask,
        figsize=figsize, point_size=point_size
    )

    # Save results
    output_parquet_path = os.path.join(out_dir, f"{dataset_name}_with_clusters.parquet")
    df_seq.to_parquet(output_parquet_path, index=False)
    print(f"✓ Saved dataset with cluster IDs to {output_parquet_path}")


# ──────────────────────────────────────────────────────────────────────
# Main Execution
# ----------------------------------------------------------------------
class Tee:
    """Helper class to redirect stdout to both console and file."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except Exception:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass


def run_fold(fold, config, run_id_base, base_output_folder, num_cpus):
    """Run training and inference for a single fold."""
    # Configure TensorFlow
    tf.config.threading.set_inter_op_parallelism_threads(num_cpus)
    tf.config.threading.set_intra_op_parallelism_threads(num_cpus)

    run_id = run_id_base + fold
    run_name = f"run_{run_id}_mhc{config['MHC_CLASS']}_dim{config['EMBED_DIM']}_h{config['HEADS']}_fold{fold}"
    out_dir = os.path.join(base_output_folder, run_name)
    os.makedirs(out_dir, exist_ok=True)

    original_stdout = sys.stdout
    log_file_path_run = os.path.join(out_dir, "run_stdout.log")

    try:
        with open(log_file_path_run, 'w') as log_file:
            sys.stdout = Tee(original_stdout, log_file)

            print(f"\n{'=' * 80}")
            print(f"Starting Run {run_id} (Fold {fold}): {run_name}")
            print(f"Description: {config['description']}")
            print(f"{'=' * 80}\n")

            # Set paths
            fold_dir_base = "../data/cross_validation_dataset/mhc1/cv_folds"
            paths = {
                "train": os.path.join(fold_dir_base, f"fold_{fold:02d}_train.parquet"),
                "val": os.path.join(fold_dir_base, f"fold_{fold:02d}_val.parquet"),
                "test": os.path.join(os.path.dirname(fold_dir_base), "test_set_rarest_alleles.parquet"),
                "neg_pool": os.path.join(os.path.dirname(fold_dir_base), "negatives_pool.parquet"),
                "embed_npz": f"/media/amirreza/Crucial-500/ESM/esm3-open/PMGen_whole_seq_/mhc{config['MHC_CLASS']}_encodings.npz",
                "embed_key": f"/media/amirreza/Crucial-500/ESM/esm3-open/PMGen_whole_seq_/mhc{config['MHC_CLASS']}_encodings.csv",
                "seq_csv": f"../data/alleles/aligned_PMGen_class_{config['MHC_CLASS']}.csv",
                "out_dir": out_dir,
            }

            # Log experiment
            log_file_path = os.path.join(base_output_folder, "experiment_log.csv")
            with open(log_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([run_id, run_name, fold] + list(config.values()))

            # Train model
            print("Starting Training...")
            max_pep_len, max_mhc_len, seq_map, embed_map = train(
                train_path=paths["train"],
                validation_path=paths["val"],
                test_path=paths["test"],
                negs_pool=paths["neg_pool"],
                embed_npz=paths["embed_npz"],
                seq_csv=paths["seq_csv"],
                embd_key_path=paths["embed_key"],
                out_dir=paths["out_dir"],
                mhc_class=config["MHC_CLASS"],
                epochs=config["EPOCHS"],
                batch_size=config["BATCH_SIZE"],
                lr=config["LEARNING_RATE"],
                embed_dim=config["EMBED_DIM"],
                heads=config["HEADS"],
                noise_std=config["NOISE_STD"]
            )

            # Run inference
            print("\nStarting Inference...")
            best_model_path = os.path.join(paths["out_dir"], "best_model.weights.h5")

            if not os.path.exists(best_model_path):
                print(f"Could not find {best_model_path}. Skipping inference.")
                return

            # Inference on each dataset
            for dset_name in ["train", "val", "test"]:
                infer_out_dir = os.path.join(paths["out_dir"], f"inference_{dset_name}")
                infer(
                    model_weights_path=best_model_path,
                    data_path=paths[dset_name],
                    out_dir=infer_out_dir,
                    name=dset_name,
                    max_pep_len=max_pep_len,
                    max_mhc_len=max_mhc_len,
                    seq_map=seq_map,
                    embed_map=embed_map,
                    mhc_class=config["MHC_CLASS"],
                    embed_dim=config["EMBED_DIM"],
                    heads=config["HEADS"],
                    noise_std=config["NOISE_STD"],
                    batch_size=config["BATCH_SIZE"]
                )

            # Joint inference on train + test
            print("\nStarting Joint Inference on Train + Test Sets...")
            df_train_joint = pd.read_parquet(paths["train"])
            df_train_joint['source'] = 'train'
            df_test_joint = pd.read_parquet(paths["test"])
            df_test_joint['source'] = 'test'
            df_joint = pd.concat([df_train_joint, df_test_joint], ignore_index=True)

            joint_data_path = os.path.join(paths["out_dir"], "joint_train_test_data.parquet")
            df_joint.to_parquet(joint_data_path)

            joint_infer_out_dir = os.path.join(paths["out_dir"], "inference_train_test_joint")
            infer(
                model_weights_path=best_model_path,
                data_path=joint_data_path,
                out_dir=joint_infer_out_dir,
                name="train_test_joint",
                max_pep_len=max_pep_len,
                max_mhc_len=max_mhc_len,
                seq_map=seq_map,
                embed_map=embed_map,
                mhc_class=config["MHC_CLASS"],
                embed_dim=config["EMBED_DIM"],
                heads=config["HEADS"],
                noise_std=config["NOISE_STD"],
                batch_size=config["BATCH_SIZE"],
                source_col='source'
            )

            # Benchmark inference
            bench_root = "../data/cross_validation_dataset/mhc1/benchmarks"
            print(f"\nStarting Benchmark Inference...")
            if os.path.exists(bench_root):
                for root, _, files in os.walk(bench_root):
                    for fname in files:
                        if not fname.lower().endswith(".parquet"):
                            continue
                        bench_path = os.path.join(root, fname)
                        rel = os.path.relpath(bench_path, bench_root)
                        safe_name = re.sub(r'[^0-9A-Za-z._-]+', '_', rel)
                        infer_out_dir = os.path.join(paths["out_dir"], f"inference_benchmark_{safe_name}")
                        print(f"Running inference on benchmark file: {bench_path}")
                        try:
                            infer(
                                model_weights_path=best_model_path, data_path=bench_path, out_dir=infer_out_dir,
                                name=f"benchmark_{safe_name}", max_pep_len=max_pep_len, max_mhc_len=max_mhc_len,
                                seq_map=seq_map, embed_map=embed_map, mhc_class=config["MHC_CLASS"],
                                embed_dim=config["EMBED_DIM"], heads=config["HEADS"],
                                noise_std=config["NOISE_STD"], batch_size=config["BATCH_SIZE"]
                            )
                        except Exception as e:
                            print(f"Failed inference for {bench_path}: {e}")
            else:
                print(f"Benchmarks folder not found at {bench_root}, skipping benchmark inference.")
    finally:
        sys.stdout = original_stdout
        print(f"Finished run {run_id} (Fold {fold}). Log saved to {log_file_path_run}")

def main(fold_to_run, num_cpus=1):
    """Main function to run the training and inference pipeline for multiple configurations."""
    config = {
        "MHC_CLASS": 1, "EPOCHS": 15, "BATCH_SIZE": 100, "LEARNING_RATE": 1e-4,
        "EMBED_DIM": 32, "HEADS": 2, "NOISE_STD": 0.1,
        "description": "Focal Loss + Automatic loss weighting. no additional embedding masking + negative resampling"
    }

    base_output_folder = "../results/PMBind_runs/"
    log_file_path = os.path.join(base_output_folder, "experiment_log.csv")
    os.makedirs(base_output_folder, exist_ok=True)

    log_header = ['run_id', 'output_folder', 'fold'] + list(config.keys())
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_header)

    existing = [d for d in os.listdir(base_output_folder)
                if os.path.isdir(os.path.join(base_output_folder, d)) and d.startswith("run_")]
    pattern = re.compile(r"^run_(\d+)(?:_|$)")
    existing_ids = [int(m.group(1)) for d in existing if (m := pattern.match(d))]
    run_id_base = (max(existing_ids) + 1) if existing_ids else 0

    if not fold_to_run:
        raise ValueError(
            "No fold specified to run. Please provide fold numbers using the --fold argument.")

    # Use 'spawn' start method for better cross-platform compatibility, especially with TensorFlow.
    run_fold(fold_to_run, config, run_id_base, base_output_folder, num_cpus)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training and inference for specified folds.")
    parser.add_argument("--fold", type=int, required=True, help="Fold number to run (e.g., 0-4).")
    parser.add_argument("-c", type=int, default=1, help="Number of CPU cores to use.")
    args = parser.parse_args()
    main(args.fold, args.c)