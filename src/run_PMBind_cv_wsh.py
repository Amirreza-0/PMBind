#!/usr/bin/env python
"""
GradientTape training and inference script for pmbind_multitask model.

This script provides a full pipeline:
1.  Load and preprocess data from Parquet files.
2.  Train a multi-task model using a tf.GradientTape loop for both
    binding classification and sequence reconstruction.
3.  Save the best model based on validation AUC.
4.  Run inference on the train, validation, and test sets using the best model.
5.  Generate and save comprehensive visualizations for both training and inference phases.

---
Changes from original:
-   **Performance Improvement (Focal Loss):** Added Focal Loss for the classification
    task to better handle class imbalance by focusing on hard-to-classify examples.
-   **Performance Improvement (Auto Loss Weighting):** Replaced fixed loss weights
    with an automatic, uncertainty-based loss weighting scheme. This learns to
    balance the classification and reconstruction tasks during training.
-   **Enhanced Logging:** Implemented a 'Tee' mechanism to direct stdout to both
    the console and a log file, ensuring logs are visible during execution and saved.
-   **Improved Progress Bar:** The training progress bar (`tqdm`) now displays the
    validation AUC from the previous epoch for more immediate feedback.
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

# Added for visualizations
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score)
from sklearn.model_selection import train_test_split

# Assuming these are in a 'utils' directory relative to the script
from utils import (seq_to_onehot, get_embed_key, NORM_TOKEN, MASK_TOKEN, PAD_TOKEN, PAD_VALUE, MASK_VALUE,
                   clean_key, masked_categorical_crossentropy, seq_to_blossom62, split_y_true_y_pred, OHE_to_seq_single,
                   peptide_properties_biopython, cn_terminal_amino_acids, reduced_anchor_pair)

from models import pmbind_multitask as pmbind

from visualizations import _analyze_latents

# ──────────────────────────────────────────────────────────────────────
# Globals & Data Preparation
# ----------------------------------------------------------------------
EMB_DB: np.lib.npyio.NpzFile | None = None
MHC_CLASS = 1
ESM_DIM = 1536


def load_embedding_db(npz_path: str):
    return np.load(npz_path, mmap_mode="r")


def random_mask(length: int, mask_fraction: float = 0.15) -> np.ndarray:
    out = np.full((length,), NORM_TOKEN, dtype=np.float32)
    n_mask = int(mask_fraction * length)
    if n_mask > 0:
        idx = np.random.choice(length, n_mask, replace=False)
        out[idx] = MASK_TOKEN
    return out


def rows_to_tensors(rows: pd.DataFrame, max_pep_len: int, max_mhc_len: int, seq_map: dict[str, str],
                        embed_map: dict[str, str], is_training: bool = True) -> dict[str, tf.Tensor]:
        n = len(rows)
        batch_data = {
            "pep_blossom62": np.zeros((n, max_pep_len, 23), np.float32),
            "pep_mask": np.full((n, max_pep_len), PAD_TOKEN, dtype=np.float32),
            "mhc_emb": np.zeros((n, max_mhc_len, ESM_DIM), np.float32),
            "mhc_mask": np.full((n, max_mhc_len), PAD_TOKEN, dtype=np.float32),
            "pep_ohe_target": np.zeros((n, max_pep_len, 21), np.float32),
            "mhc_ohe_target": np.zeros((n, max_mhc_len, 21), np.float32),
            "labels": np.zeros((n, 1), np.int32),
        }

        for i, (_, r) in enumerate(rows.iterrows()):
            if "mhc_embedding_key" not in r or pd.isna(r["mhc_embedding_key"]):
                r["mhc_embedding_key"] = r["allele"].replace(" ", "").replace("*", "").replace(":", "")

            pep_seq = r["long_mer"].upper()
            batch_data["pep_blossom62"][i] = seq_to_blossom62(pep_seq, max_seq_len=max_pep_len)
            batch_data["pep_ohe_target"][i] = seq_to_onehot(pep_seq, max_seq_len=max_pep_len)
            pep_len = len(pep_seq)
            batch_data["pep_mask"][i, :pep_len] = NORM_TOKEN

            if is_training:
                valid_positions = np.where(batch_data["pep_mask"][i] == NORM_TOKEN)[0]
                if len(valid_positions) > 0:
                    mask_fraction = 0.30
                    n_mask = max(1, int(mask_fraction * len(valid_positions)))
                    mask_indices = np.random.choice(valid_positions, size=n_mask, replace=False)
                    batch_data["pep_mask"][i, mask_indices] = MASK_TOKEN
                    batch_data["pep_blossom62"][i, mask_indices, :] = MASK_VALUE

            if MHC_CLASS == 2:
                key_parts = r["mhc_embedding_key"].split("_")
                embd_key1 = get_embed_key(clean_key(key_parts[0]), embed_map)
                embd_key2 = get_embed_key(clean_key(key_parts[1]), embed_map)
                emb1, emb2 = EMB_DB[embd_key1], EMB_DB[embd_key2]
                emb = np.concatenate([emb1, emb2], axis=0)
            else:
                embd_key = get_embed_key(clean_key(r["mhc_embedding_key"]), embed_map)
                emb = EMB_DB[embd_key]

            L = emb.shape[0]
            batch_data["mhc_emb"][i, :L] = emb
            batch_data["mhc_emb"][i, L:, :] = PAD_VALUE
            is_padding = np.all(batch_data["mhc_emb"][i] == PAD_VALUE, axis=-1)
            batch_data["mhc_mask"][i, ~is_padding] = NORM_TOKEN

            if is_training:
                valid_mhc_positions = np.where(batch_data["mhc_mask"][i] == NORM_TOKEN)[0]
                if len(valid_mhc_positions) > 0:
                    mask_fraction = 0.30
                    n_mask = max(1, int(mask_fraction * len(valid_mhc_positions)))
                    mask_indices = np.random.choice(valid_mhc_positions, size=n_mask, replace=False)
                    batch_data["mhc_mask"][i, mask_indices] = MASK_TOKEN
                    batch_data["mhc_emb"][i, mask_indices, :] = MASK_VALUE
                    # additional mask - masking 0.15 on the embedding dimension
                    # emb_dim = batch_data["mhc_emb"].shape[-1]
                    # for idx in valid_mhc_positions:
                    #     emb_mask = random_mask(emb_dim, mask_fraction=0.15)
                    #     emb_mask = np.where(emb_mask == MASK_TOKEN, MASK_VALUE, 1.0)
                    #     batch_data["mhc_emb"][i, idx, :] *= emb_mask

            if MHC_CLASS == 2:
                key_parts = r["mhc_embedding_key"].split("_")
                key_norm1 = get_embed_key(clean_key(key_parts[0]), seq_map)
                key_norm2 = get_embed_key(clean_key(key_parts[1]), seq_map)
                mhc_seq = seq_map[key_norm1] + seq_map[key_norm2]
            else:
                key_norm = get_embed_key(clean_key(r["mhc_embedding_key"]), seq_map)
                mhc_seq = seq_map[key_norm]
            batch_data["mhc_ohe_target"][i] = seq_to_onehot(mhc_seq, max_seq_len=max_mhc_len)
            batch_data["labels"][i, 0] = int(r["assigned_label"])

        return {k: tf.convert_to_tensor(v) for k, v in batch_data.items()}


# ──────────────────────────────────────────────────────────────────────
# Training
# ----------------------------------------------------------------------
def train(train_path, validation_path, test_path, embed_npz, seq_csv, embd_key_path, out_dir,
          mhc_class, epochs, batch_size, lr, embed_dim, heads, noise_std):
    """Training function with automatic loss balancing."""
    global EMB_DB, MHC_CLASS, ESM_DIM
    EMB_DB = load_embedding_db(embed_npz)
    MHC_CLASS = mhc_class
    ESM_DIM = int(next(iter(EMB_DB.values())).shape[1])

    seq_map = pd.read_csv(seq_csv, index_col="allele")["mhc_sequence"].to_dict()
    embed_map = pd.read_csv(embd_key_path, index_col="key")["mhc_sequence"].to_dict()
    seq_map = {clean_key(k): v for k, v in seq_map.items()}

    df_train = pq.ParquetFile(train_path).read().to_pandas()
    df_val = pq.ParquetFile(validation_path).read().to_pandas()
    df_test = pq.ParquetFile(test_path).read().to_pandas()
    print(f"Loaded {len(df_train)} training, {len(df_val)} validation, {len(df_test)} test samples.")

    max_pep_len = int(pd.concat([df_train["long_mer"], df_val["long_mer"], df_test["long_mer"]]).str.len().max())
    max_mhc_len = 500 if mhc_class == 2 else int(next(iter(EMB_DB.values())).shape[0])
    print(f"Max peptide length: {max_pep_len}, Max MHC length: {max_mhc_len}")

    model = pmbind(max_pep_len=max_pep_len, max_mhc_len=max_mhc_len, emb_dim=embed_dim,
                   heads=heads, noise_std=noise_std, transformer_layers=2,
                   latent_dim=embed_dim * 2, ESM_dim=ESM_DIM)

    # --- PERFORMANCE IMPROVEMENT: Automatic Loss Weighting ---
    log_var_cls = tf.Variable(0.0, trainable=True, name='log_var_cls')
    log_var_pep = tf.Variable(0.0, trainable=True, name='log_var_pep')
    log_var_mhc = tf.Variable(0.0, trainable=True, name='log_var_mhc')
    learnable_loss_vars = [log_var_cls, log_var_pep, log_var_mhc]
    print("Initialized learnable log variances for automatic loss balancing.")

    num_train_steps = (len(df_train) // batch_size) * epochs
    lr_schedule = keras.optimizers.schedules.CosineDecay(initial_learning_rate=lr, decay_steps=num_train_steps,
                                                         alpha=1e-6)
    optimizer = keras.optimizers.Lion(learning_rate=lr_schedule)

    # --- PERFORMANCE IMPROVEMENT: Use Focal Loss for classification ---
    focal_loss_fn = tf.keras.losses.BinaryFocalCrossentropy(from_logits=False, reduction="sum_over_batch_size",
                                                            alpha=0.25, gamma=2.0)

    metrics = {
        "train_loss": tf.keras.metrics.Mean(name="train_loss"),
        "val_loss": tf.keras.metrics.Mean(name="val_loss"),
        "test_loss": tf.keras.metrics.Mean(name="test_loss"),
        "train_auc": tf.keras.metrics.AUC(name="train_auc"),
        "val_auc": tf.keras.metrics.AUC(name="val_auc"),
        "test_auc": tf.keras.metrics.AUC(name="test_auc")
    }
    history = {key: [] for key in ["train_loss", "val_loss", "test_loss", "train_auc", "val_auc", "test_auc",
                                   "train_cls_loss", "val_cls_loss", "train_recon_loss", "val_recon_loss"]}
    best_val_auc = 0.0
    os.makedirs(out_dir, exist_ok=True)

    # Build model
    dummy_data = rows_to_tensors(df_train.head(1), max_pep_len, max_mhc_len, seq_map, embed_map)
    model(dummy_data)
    model.summary(print_fn=lambda x: print(x, file=sys.stdout))

    train_indices = np.arange(len(df_train))

    # --- FIX: Use stratified sampling for the fixed validation batch ---
    n_val_samples = min(len(df_val), 256)
    if n_val_samples < len(df_val):
        labels = sorted(df_val['assigned_label'].unique())
        num_classes = len(labels)
        base = n_val_samples // num_classes
        rem = n_val_samples % num_classes
        sampled_parts = []
        for i, lbl in enumerate(labels):
            take = base + (1 if i < rem else 0)
            grp = df_val[df_val['assigned_label'] == lbl]
            if len(grp) == 0:
                continue
            if len(grp) >= take:
                sampled_parts.append(grp.sample(n=take, random_state=999))
            else:
                sampled_parts.append(grp.sample(n=take, replace=True, random_state=999))
        val_df_sampled = pd.concat(sampled_parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        val_df_sampled = df_val.sample(frac=1, random_state=42)

    print(f"Created a fixed validation batch of size {len(val_df_sampled)} balanced.")
    print(f"Validation batch class distribution:\n{val_df_sampled['assigned_label'].value_counts(normalize=True)}\n")
    fixed_val_batch = rows_to_tensors(val_df_sampled, max_pep_len, max_mhc_len, seq_map, embed_map, is_training=False)

    # --- ADD: Use stratified sampling for the fixed test batch ---
    n_test_samples = min(len(df_test), 256)
    if n_test_samples < len(df_test):
        labels_test = sorted(df_test['assigned_label'].unique())
        num_classes_test = len(labels_test)
        base_test = n_test_samples // num_classes_test
        rem_test = n_test_samples % num_classes_test
        sampled_parts_test = []
        for i, lbl in enumerate(labels_test):
            take = base_test + (1 if i < rem_test else 0)
            grp = df_test[df_test['assigned_label'] == lbl]
            if len(grp) == 0: continue
            if len(grp) >= take:
                sampled_parts_test.append(grp.sample(n=take, random_state=998))
            else:
                sampled_parts_test.append(grp.sample(n=take, replace=True, random_state=998))
        test_df_sampled = pd.concat(sampled_parts_test, ignore_index=True).sample(frac=1, random_state=41).reset_index(drop=True)
    else:
        test_df_sampled = df_test.sample(frac=1, random_state=41)

    print(f"Created a fixed test batch of size {len(test_df_sampled)} balanced.")
    print(f"Test batch class distribution:\n{test_df_sampled['assigned_label'].value_counts(normalize=True)}\n")
    fixed_test_batch = rows_to_tensors(test_df_sampled, max_pep_len, max_mhc_len, seq_map, embed_map, is_training=False)

    last_val_auc = 0.0
    last_test_auc = 0.0

    for epoch in range(epochs):
        np.random.shuffle(train_indices)
        pbar = tqdm(range(0, len(train_indices), batch_size),
                    desc=f"Epoch {epoch + 1}/{epochs}",
                    file=sys.stdout)

        for start in pbar:
            batch_idx = train_indices[start:start + batch_size]
            batch_data = rows_to_tensors(df_train.iloc[batch_idx], max_pep_len, max_mhc_len, seq_map, embed_map)

            with tf.GradientTape() as tape:
                outputs = model(batch_data, training=True)

                raw_cls_loss = focal_loss_fn(batch_data["labels"], outputs["cls_ypred"])
                precision_cls = tf.exp(-log_var_cls)
                cls_loss = precision_cls * raw_cls_loss + log_var_cls

                raw_recon_loss_pep = masked_categorical_crossentropy(outputs["pep_ytrue_ypred"], batch_data["pep_mask"])
                precision_pep = tf.exp(-log_var_pep)
                recon_loss_pep = precision_pep * raw_recon_loss_pep + log_var_pep

                raw_recon_loss_mhc = masked_categorical_crossentropy(outputs["mhc_ytrue_ypred"], batch_data["mhc_mask"])
                precision_mhc = tf.exp(-log_var_mhc)
                recon_loss_mhc = precision_mhc * raw_recon_loss_mhc + log_var_mhc

                loss = cls_loss + recon_loss_pep + recon_loss_mhc

            trainable_vars = model.trainable_variables + learnable_loss_vars
            grads = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))

            metrics["train_loss"](loss)
            metrics["train_auc"](batch_data["labels"], outputs["cls_ypred"])

            pbar.set_postfix(
                {"loss": metrics['train_loss'].result().numpy(),
                 "auc": metrics['train_auc'].result().numpy(),
                 "val_auc": f"{last_val_auc:.4f}",
                 "test_auc": f"{last_test_auc:.4f}",
                 "cls_loss(focal)": raw_cls_loss.numpy(),
                 "pep_recon": raw_recon_loss_pep.numpy(),
                 "mhc_recon": raw_recon_loss_mhc.numpy()})

        # Validation step
        val_outputs = model(fixed_val_batch, training=False)
        cls_loss_val = focal_loss_fn(fixed_val_batch["labels"], val_outputs["cls_ypred"])
        recon_loss_pep_val = masked_categorical_crossentropy(val_outputs["pep_ytrue_ypred"],
                                                             fixed_val_batch["pep_mask"])
        recon_loss_mhc_val = masked_categorical_crossentropy(val_outputs["mhc_ytrue_ypred"],
                                                             fixed_val_batch["mhc_mask"])
        loss_val = cls_loss_val + recon_loss_pep_val + recon_loss_mhc_val
        metrics["val_auc"].reset_state()
        metrics["val_loss"](loss_val)
        metrics["val_auc"](fixed_val_batch["labels"], val_outputs["cls_ypred"])
        val_auc_result = metrics['val_auc'].result().numpy()
        val_loss_result = metrics['val_loss'].result().numpy()

        # Test step
        test_outputs = model(fixed_test_batch, training=False)
        cls_loss_test = focal_loss_fn(fixed_test_batch["labels"], test_outputs["cls_ypred"])
        recon_loss_pep_test = masked_categorical_crossentropy(test_outputs["pep_ytrue_ypred"], fixed_test_batch["pep_mask"])
        recon_loss_mhc_test = masked_categorical_crossentropy(test_outputs["mhc_ytrue_ypred"], fixed_test_batch["mhc_mask"])
        loss_test = cls_loss_test + recon_loss_pep_test + recon_loss_mhc_test
        metrics["test_auc"].reset_state()
        metrics["test_loss"](loss_test)
        metrics["test_auc"](fixed_test_batch["labels"], test_outputs["cls_ypred"])
        test_auc_result = metrics['test_auc'].result().numpy()
        test_loss_result = metrics['test_loss'].result().numpy()

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  - Val AUC: {val_auc_result:.4f}, Val Loss: {val_loss_result:.4f}")
        print(f"  - Test AUC: {test_auc_result:.4f}, Test Loss: {test_loss_result:.4f}")
        print(
            f"  - Raw Losses -> Val_Cls(focal): {cls_loss_val.numpy():.4f}, Val_Pep_Recon: {recon_loss_pep_val.numpy():.4f}, Val_MHC_Recon: {recon_loss_mhc_val.numpy():.4f}")
        print(
            f"  - Learned Log Variances -> Cls: {log_var_cls.numpy():.4f}, Pep: {log_var_pep.numpy():.4f}, MHC: {log_var_mhc.numpy():.4f}")

        for key, met in metrics.items():
            history[key].append(float(met.result().numpy()))
            met.reset_state()

        last_val_auc = history['val_auc'][-1]
        last_test_auc = history['test_auc'][-1]

        if history['val_auc'][-1] > best_val_auc:
            best_val_auc = history['val_auc'][-1]
            model.save_weights(os.path.join(out_dir, "best_model.weights.h5"))
            print(f"  -> Best model saved with Val AUC: {best_val_auc:.4f}")

        print(f"  -> Next epoch will display Val AUC: {last_val_auc:.4f}, Test AUC: {last_test_auc:.4f} in progress bar")


    model.save_weights(os.path.join(out_dir, "final_model.weights.h5"))
    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    print("Training complete.")

    visualize_training_history(history, out_dir)
    return max_pep_len, max_mhc_len, seq_map, embed_map


def infer(model_weights_path, data_path, out_dir, name,
          max_pep_len, max_mhc_len, seq_map, embed_map,
          mhc_class, embed_dim, heads, noise_std, batch_size, source_col=None):
    """Runs inference on a given dataset using a pre-trained model."""
    global MHC_CLASS, ESM_DIM
    MHC_CLASS = mhc_class
    ESM_DIM = int(next(iter(EMB_DB.values())).shape[1])

    df_infer = pq.ParquetFile(data_path).read().to_pandas()
    print(f"Loaded {len(df_infer)} samples for inference from {data_path}.")
    os.makedirs(out_dir, exist_ok=True)

    model = pmbind(max_pep_len=max_pep_len, max_mhc_len=max_mhc_len, emb_dim=embed_dim,
                   heads=heads, noise_std=noise_std, transformer_layers=2,
                   latent_dim=embed_dim * 2, ESM_dim=ESM_DIM)

    dummy_input = rows_to_tensors(df_infer.head(1), max_pep_len, max_mhc_len, seq_map, embed_map, is_training=False)
    model(dummy_input, training=False)
    model.load_weights(model_weights_path)
    print(f"Model weights loaded from {model_weights_path} for inference on {name} set.")

    latents_seq_path = os.path.join(out_dir, f"latents_seq_{name}.mmap")
    latents_pooled_path = os.path.join(out_dir, f"latents_pooled_{name}.mmap")

    if os.path.exists(latents_pooled_path) and os.path.exists(latents_seq_path):
        print(f"Latents data already exists. Loading from {latents_pooled_path}")
        latents_seq = np.memmap(latents_seq_path, dtype='float32', mode='r',
                                shape=(len(df_infer), max_pep_len + max_mhc_len, embed_dim))
        latents_pooled = np.memmap(latents_pooled_path, dtype='float32', mode='r',
                                   shape=(len(df_infer), max_pep_len + max_mhc_len + embed_dim))
    else:
        latents_seq = np.memmap(latents_seq_path, dtype='float32', mode='w+',
                                shape=(len(df_infer), max_pep_len + max_mhc_len, embed_dim))
        latents_pooled = np.memmap(latents_pooled_path, dtype='float32', mode='w+',
                                   shape=(len(df_infer), max_pep_len + max_mhc_len + embed_dim))

        all_predictions, all_labels = [], []
        print(f"Processing {len(df_infer)} samples for inference in batches...")
        pbar = tqdm(range(0, len(df_infer), batch_size), desc=f"Inference on {name} set", file=sys.stdout)
        for start in pbar:
            batch_idx = np.arange(start, min(start + batch_size, len(df_infer)))
            batch_df = df_infer.iloc[batch_idx]
            batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len, seq_map, embed_map, is_training=False)
            outputs = model(batch_data, training=False)

            all_predictions.append(outputs["cls_ypred"].numpy())
            all_labels.append(batch_data["labels"].numpy())
            latents_seq[batch_idx] = outputs["latent_seq"].numpy()
            latents_pooled[batch_idx] = outputs["latent_vector"].numpy()

        latents_seq.flush()
        latents_pooled.flush()

        all_predictions = np.concatenate(all_predictions, axis=0).squeeze()
        all_labels = np.concatenate(all_labels, axis=0).squeeze()

        df_infer["prediction_score"] = all_predictions
        df_infer["prediction_label"] = (all_predictions >= 0.5).astype(int)

        output_path = os.path.join(out_dir, f"inference_results_{name}.csv")
        df_infer.to_csv(output_path, index=False)
        print(f"✓ Sequential latents saved to {latents_seq_path}")
        print(f"✓ Pooled latents saved to {latents_pooled_path}")
        print(f"✓ Inference results saved to {output_path}")

        if "assigned_label" in df_infer.columns:
            visualize_inference_results(df_infer, all_labels, all_predictions, out_dir, name)

    vis_out_dir = os.path.join(out_dir, "visualizations")
    print("\n--- Running latent space visualizations ---")
    highlight_mask = None
    if source_col and source_col in df_infer.columns:
        highlight_mask = (df_infer[source_col] == 'test').values
        print(f"Highlighting {highlight_mask.sum()} test samples in visualizations.")
    run_visualizations(
        df=df_infer, latents_seq=latents_seq, latents_pooled=latents_pooled,
        enc_dec=model, max_pep_len=max_pep_len, max_mhc_len=max_mhc_len,
        seq_map=seq_map, embed_map=embed_map, out_dir=vis_out_dir,
        dataset_name=name, figsize=(30, 15), point_size=3,
        highlight_mask=highlight_mask
    )
    print(f"✓ Latent space visualizations saved to {vis_out_dir}")


# ──────────────────────────────────────────────────────────────────────
# Visualizations
# ----------------------------------------------------------------------
def visualize_training_history(history, out_dir):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
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
    """Generates and saves comprehensive evaluation plots for inference results."""
    print(f"Generating visualizations for {name} set...")

    cm = confusion_matrix(true_labels, df_with_results["prediction_label"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"Confusion Matrix on {name} Set")
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{name}.png"))
    plt.close()

    fpr, tpr, _ = roc_curve(true_labels, prediction_scores)
    auc = roc_auc_score(true_labels, prediction_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlabel('False Positive Rate'), plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve on {name} Set'), plt.legend(loc='lower right'), plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(out_dir, f"roc_curve_{name}.png"))
    plt.close()

    precision, recall, _ = precision_recall_curve(true_labels, prediction_scores)
    ap = average_precision_score(true_labels, prediction_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel('Recall'), plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve on {name} Set'), plt.legend(loc='upper right'), plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(out_dir, f"precision_recall_curve_{name}.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_with_results, x='prediction_score', hue='assigned_label', kde=True, bins=50)
    plt.title(f'Prediction Score Distribution on {name} Set')
    plt.savefig(os.path.join(out_dir, f"score_distribution_{name}.png"))
    plt.close()
    print(f"Visualizations for {name} set saved in {out_dir}")


def run_visualizations(df, latents_seq, latents_pooled, enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, out_dir,
                       dataset_name: str, highlight_mask: np.ndarray | None = None,
                       figsize=(40, 15), point_size=2):
    """Generates and saves a series of visualizations for model analysis."""
    print("\nGenerating visualizations...")
    os.makedirs(out_dir, exist_ok=True)
    alleles = df['mhc_embedding_key'].apply(clean_key).astype('category')
    unique_alleles = alleles.cat.categories
    num_to_highlight = min(5, len(unique_alleles))
    np.random.seed(999)
    random_alleles_to_highlight = np.random.choice(unique_alleles, num_to_highlight, replace=False).tolist()
    print(
        f"Found {len(unique_alleles)} unique alleles. Highlighting {num_to_highlight} random alleles: {random_alleles_to_highlight}")

    if len(unique_alleles) <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_alleles)))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_alleles)))
    allele_color_map = {allele: color for allele, color in zip(unique_alleles, colors)}

    df_seq = _analyze_latents(
        latents=latents_seq, df=df, alleles=alleles, allele_color_map=allele_color_map,
        random_alleles_to_highlight=random_alleles_to_highlight, latent_type="seq",
        out_dir=out_dir, dataset_name=dataset_name, highlight_mask=highlight_mask,
        figsize=figsize, point_size=point_size
    )
    df_pooled = _analyze_latents(
        latents=latents_pooled, df=df, alleles=alleles, allele_color_map=allele_color_map,
        random_alleles_to_highlight=random_alleles_to_highlight, latent_type="pooled",
        out_dir=out_dir, dataset_name=dataset_name, highlight_mask=highlight_mask,
        figsize=figsize, point_size=point_size
    )
    output_parquet_path = os.path.join(out_dir, f"{dataset_name}_with_clusters.parquet")
    df_seq.to_parquet(output_parquet_path, index=False)
    df_pooled.to_parquet(output_parquet_path, index=False)
    print(f"\n✓ Saved dataset with all cluster IDs to {output_parquet_path}")

    print("\n--- Generating supplementary plots (inputs, masks, predictions) ---")
    sample_idx = 0
    sample_row = df_seq.iloc[[sample_idx]]
    sample_data = rows_to_tensors(sample_row, max_pep_len, max_mhc_len, seq_map, embed_map)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle(f'Input Data for Sample {sample_idx}', fontsize=16)
    sns.heatmap(sample_data['pep_blossom62'][0].numpy().T, ax=axes[0, 0], cmap='gray_r')
    axes[0, 0].set_title('Peptide Input (One-Hot)')
    sns.heatmap(sample_data['pep_mask'][0].numpy()[np.newaxis, :], ax=axes[0, 1], cmap='viridis', cbar=False)
    axes[0, 1].set_title('Peptide Mask')
    sns.heatmap(sample_data['mhc_emb'][0].numpy().T, ax=axes[1, 0], cmap='viridis')
    axes[1, 0].set_title('MHC Input (Embedding)')
    sns.heatmap(sample_data['mhc_mask'][0].numpy()[np.newaxis, :], ax=axes[1, 1], cmap='viridis', cbar=False)
    axes[1, 1].set_title('MHC Mask')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, "input_and_mask_samples.png"))
    plt.close()
    print("✓ Input and mask plots saved.")

    print("\n--- Comparing 5 Predictions with Inputs ---")
    pred_samples_df = df_seq.head(5)
    pred_data = rows_to_tensors(pred_samples_df, max_pep_len, max_mhc_len, seq_map, embed_map)
    model_inputs = {
        "pep_blossom62": pred_data["pep_blossom62"], "pep_mask": pred_data["pep_mask"],
        "mhc_emb": pred_data["mhc_emb"], "mhc_mask": pred_data["mhc_mask"],
        "mhc_ohe_target": pred_data["mhc_ohe_target"], "pep_ohe_target": pred_data["pep_ohe_target"]
    }
    true_preds = enc_dec(model_inputs, training=False)
    pep_true, pep_pred_ohe = split_y_true_y_pred(true_preds["pep_ytrue_ypred"].numpy())
    mhc_true, mhc_pred_ohe = split_y_true_y_pred(true_preds["mhc_ytrue_ypred"].numpy())
    pep_masks_np = pred_data["pep_mask"].numpy()
    mhc_masks_np = pred_data["mhc_mask"].numpy()

    pred_list = []
    for i in range(5):
        allele = clean_key(pred_samples_df.iloc[i]['mhc_embedding_key'])
        original_peptide_full = OHE_to_seq_single(pep_true[i], gap=True).replace("X", "-")
        predicted_peptide_full = OHE_to_seq_single(pep_pred_ohe[i], gap=True).replace("X", "-")
        pep_valid_mask = (pep_masks_np[i] != PAD_TOKEN) & (np.array(list(original_peptide_full)) != '-')
        original_peptide = "".join(np.array(list(original_peptide_full))[pep_valid_mask])
        predicted_peptide = "".join(np.array(list(predicted_peptide_full))[pep_valid_mask])

        original_mhc_full = OHE_to_seq_single(mhc_true[i], gap=True).replace("X", "-")
        predicted_mhc_full = OHE_to_seq_single(mhc_pred_ohe[i], gap=True).replace("X", "-")
        mhc_valid_mask = (mhc_masks_np[i] != PAD_TOKEN) & (np.array(list(original_mhc_full)) != '-')
        original_mhc = "".join(np.array(list(original_mhc_full))[mhc_valid_mask])
        predicted_mhc = "".join(np.array(list(predicted_mhc_full))[mhc_valid_mask])

        pred_list.append({
            "sample_index": int(pred_samples_df.index[i]), "allele": allele,
            "original_peptide": original_peptide, "predicted_peptide": predicted_peptide,
            "original_mhc": original_mhc, "predicted_mhc": predicted_mhc
        })

    predictions_df = pd.DataFrame(pred_list)
    predictions_output_path = os.path.join(out_dir, f"sequence_predictions_{dataset_name}.csv")
    predictions_df.to_csv(predictions_output_path, index=False)
    print(f"✓ Sequence predictions saved to {predictions_output_path}")


# ──────────────────────────────────────────────────────────────────────
# Main Execution
# ----------------------------------------------------------------------
class Tee:
    """A helper class to redirect stdout to both console and a file."""

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


def main():
    """Main function to run the training and inference pipeline for multiple configurations."""
    config = {
        "MHC_CLASS": 1, "EPOCHS": 3, "BATCH_SIZE": 100, "LEARNING_RATE": 1e-4,
        "EMBED_DIM": 32, "HEADS": 2, "NOISE_STD": 0.1,
        "description": "Focal Loss (gamma=2, alpha=0.25) + Automatic loss weighting. no additional embedding masking"
    }

    base_output_folder = "/media/amirreza/Crucial-500/PMBind_runs/"
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

    cv_base_path = f"../data/cv_mhc{config['MHC_CLASS']}"
    num_folds = 11

    for fold in range(num_folds):
        fold = fold + 1
        run_id = run_id_base + fold
        run_name = f"run_{run_id}_mhc{config['MHC_CLASS']}_dim{config['EMBED_DIM']}_h{config['HEADS']}_fold{fold}"
        out_dir = os.path.join(base_output_folder, run_name)
        os.makedirs(out_dir, exist_ok=True)

        original_stdout = sys.stdout
        log_file_path_run = os.path.join(out_dir, "run_stdout.log")

        try:
            with open(log_file_path_run, 'w') as log_file:
                sys.stdout = Tee(original_stdout, log_file)

                print(f"\n\n{'=' * 80}")
                print(f"--- Starting Run {run_id} (Fold {fold}): {run_name} ---")
                print(f"--- Description: {config['description']} ---")
                print(f"{'=' * 80}\n")

                fold_dir_base = "/home/amirreza/Desktop/PMBind/data/cross_validation_dataset/mhc1/cv_folds"
                paths = {
                    "train": os.path.join(fold_dir_base, f"fold_{fold:02d}_train.parquet"),
                    "val": os.path.join(fold_dir_base, f"fold_{fold:02d}_val.parquet"),
                    "test": os.path.join(os.path.dirname(fold_dir_base), "benchmark_allele_holdout.parquet"),
                    "embed_npz": f"/media/amirreza/Crucial-500/ESM/esm3-open/PMGen_whole_seq_/mhc{config['MHC_CLASS']}_encodings.npz",
                    "embed_key": f"/media/amirreza/Crucial-500/ESM/esm3-open/PMGen_whole_seq_/mhc{config['MHC_CLASS']}_encodings.csv",
                    "seq_csv": f"../data/alleles/aligned_PMGen_class_{config['MHC_CLASS']}.csv",
                    "out_dir": out_dir,
                }

                with open(log_file_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([run_id, run_name, fold] + list(config.values()))

                print("--- Starting Training ---")
                max_pep_len, max_mhc_len, seq_map, embed_map = train(
                    train_path=paths["train"], validation_path=paths["val"], test_path=paths["test"],
                    embed_npz=paths["embed_npz"], seq_csv=paths["seq_csv"], embd_key_path=paths["embed_key"],
                    out_dir=paths["out_dir"], mhc_class=config["MHC_CLASS"], epochs=config["EPOCHS"],
                    batch_size=config["BATCH_SIZE"], lr=config["LEARNING_RATE"], embed_dim=config["EMBED_DIM"],
                    heads=config["HEADS"], noise_std=config["NOISE_STD"]
                )

                print("\n--- Starting Inference ---")
                best_model_path = os.path.join(paths["out_dir"], "best_model.weights.h5")

                if not os.path.exists(best_model_path):
                    print(f"Could not find {best_model_path}. Skipping inference for this run.")
                    continue

                for dset_name in ["train", "val", "test"]:
                    infer_out_dir = os.path.join(paths["out_dir"], f"inference_{dset_name}")
                    infer(
                        model_weights_path=best_model_path, data_path=paths[dset_name], out_dir=infer_out_dir,
                        name=dset_name,
                        max_pep_len=max_pep_len, max_mhc_len=max_mhc_len, seq_map=seq_map, embed_map=embed_map,
                        mhc_class=config["MHC_CLASS"], embed_dim=config["EMBED_DIM"], heads=config["HEADS"],
                        noise_std=config["NOISE_STD"], batch_size=config["BATCH_SIZE"]
                    )

                print("\n--- Starting Joint Inference on Train + Test Sets ---")
                df_train_joint = pd.read_parquet(paths["train"])
                df_train_joint['source'] = 'train'
                df_test_joint = pd.read_parquet(paths["test"])
                df_test_joint['source'] = 'test'
                df_joint = pd.concat([df_train_joint, df_test_joint], ignore_index=True)

                joint_data_path = os.path.join(paths["out_dir"], "joint_train_test_data.parquet")
                df_joint.to_parquet(joint_data_path)
                print(f"Saved joint dataset to {joint_data_path}")

                joint_infer_out_dir = os.path.join(paths["out_dir"], "inference_train_test_joint")
                infer(
                    model_weights_path=best_model_path, data_path=joint_data_path, out_dir=joint_infer_out_dir,
                    name="train_test_joint", max_pep_len=max_pep_len, max_mhc_len=max_mhc_len,
                    seq_map=seq_map, embed_map=embed_map, mhc_class=config["MHC_CLASS"],
                    embed_dim=config["EMBED_DIM"], heads=config["HEADS"],
                    noise_std=config["NOISE_STD"], batch_size=config["BATCH_SIZE"],
                    source_col='source'
                )

                bench_root = "/home/amirreza/Desktop/PMBind/data/cross_validation_dataset/mhc1/benchmarks"
                print(f"\n--- Starting Benchmark Inference under {bench_root} ---")
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


if __name__ == "__main__":
    main()