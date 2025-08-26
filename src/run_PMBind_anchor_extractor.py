#!/usr/bin/env python
"""
GradientTape training loop for `pmclust_subtract`.
This script shows the bare-bones path through your model:

1.  Load all rows from a Parquet file into **pandas**.
2.  Shuffle & slice with NumPy to create mini-batches.
3.  Convert every mini-batch to tensors on-the-fly and feed it through a
    `tf.GradientTape` loop.
"""

from __future__ import annotations
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


# Assuming these are in a 'utils' directory relative to the script
from utils import (seq_to_onehot, get_embed_key, NORM_TOKEN, MASK_TOKEN, PAD_TOKEN, PAD_VALUE, MASK_VALUE,
                   clean_key)

from models import pmbind_anchor_extractor

# ──────────────────────────────────────────────────────────────────────
# 4. DATA PREPARATION & TRAINING LOOP
# ----------------------------------------------------------------------
EMB_DB: np.lib.npyio.NpzFile | None = None


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
                    embed_map: dict[str, str]) -> dict[str, tf.Tensor]:
    n = len(rows)
    # This dictionary contains ALL data needed for a batch, including targets.
    batch_data = {
        "pep_onehot": np.zeros((n, max_pep_len, 21), np.float32),
        "pep_mask": np.full((n, max_pep_len), PAD_TOKEN, dtype=np.float32),  # Initialize with PAD_TOKEN
        "mhc_emb": np.zeros((n, max_mhc_len, 1152), np.float32),
        "mhc_mask": np.full((n, max_mhc_len), PAD_TOKEN, dtype=np.float32),  # Initialize with PAD_TOKEN
        "labels": np.zeros((n,), np.float32),  # classification target
    }

    for i, (_, r) in enumerate(rows.iterrows()):
        ### PEP
        # Process peptide sequence
        pep_seq = r["long_mer"].upper()
        pep_OHE = seq_to_onehot(pep_seq, max_seq_len=max_pep_len)
        batch_data["pep_onehot"][i] = pep_OHE

        # Create peptide mask: 1.0 for valid positions, PAD_TOKEN for padding
        pep_len = len(pep_seq)
        batch_data["pep_mask"][i, :pep_len] = NORM_TOKEN  # Valid positions get NORM_TOKEN (1.0)
        # Positions beyond sequence length remain PAD_TOKEN (-2.0)

        # Randomly mask 30% of valid peptide positions with MASK_TOKEN
        valid_positions = np.where(batch_data["pep_mask"][i] == NORM_TOKEN)[0]
        if len(valid_positions) > 0:
            mask_fraction = 0.15
            n_mask = max(1, int(mask_fraction * len(valid_positions)))  # At least 1 position
            mask_indices = np.random.choice(valid_positions, size=n_mask, replace=False)
            batch_data["pep_mask"][i, mask_indices] = MASK_TOKEN  # Masked positions get MASK_TOKEN (-1.0)
            # Zero out the corresponding one-hot encoding for masked positions
            batch_data["pep_onehot"][i, mask_indices, :] = MASK_VALUE

        ### MHC
        # print(f"Peptide mask for sample {i}: {batch_data['pep_mask'][i]}")  # Debugging line to check peptide mask
        # Process MHC embeddings and sequence
        if MHC_CLASS == 2:
            key_parts = r["mhc_embedding_key"].split("_")
            embd_key1 = get_embed_key(clean_key(key_parts[0]), embed_map)
            embd_key2 = get_embed_key(clean_key(key_parts[1]), embed_map)
            emb1 = EMB_DB[embd_key1]
            emb2 = EMB_DB[embd_key2]
            emb = np.concatenate([emb1, emb2], axis=0)
        else:
            embd_key = get_embed_key(clean_key(r["mhc_embedding_key"]), embed_map)
            emb = EMB_DB[embd_key]
        L = emb.shape[0]
        batch_data["mhc_emb"][i, :L] = emb
        # Set padding positions in embeddings to PAD_VALUE
        batch_data["mhc_emb"][i, L:, :] = PAD_VALUE
        # print(batch_data["mhc_emb"][i, L:, :])  # Debugging line to check padding values

        # Create MHC mask based on the embedding values.
        # A position is considered padding if its embedding vector is all PAD_VALUE.
        # This handles both padding within the sequence and padding at the end.
        is_padding = np.all(batch_data["mhc_emb"][i] == PAD_VALUE, axis=-1)
        batch_data["mhc_mask"][i, ~is_padding] = NORM_TOKEN
        # Positions where is_padding is True will retain their initial PAD_TOKEN value.

        # Randomly mask 20% of valid MHC positions with MASK_TOKEN
        valid_mhc_positions = np.where(batch_data["mhc_mask"][i] == NORM_TOKEN)[0]
        if len(valid_mhc_positions) > 0:
            mask_fraction = 0.15
            n_mask = max(1, int(mask_fraction * len(valid_mhc_positions)))  # At least 1 position
            mask_indices = np.random.choice(valid_mhc_positions, size=n_mask, replace=False)
            batch_data["mhc_mask"][i, mask_indices] = MASK_TOKEN  # Masked positions get MASK_TOKEN (-1.0)
            # Zero out the corresponding embeddings for masked positions
            batch_data["mhc_emb"][i, mask_indices, :] = MASK_VALUE

        ### LABELS
        batch_data["labels"][i] = r["assigned_label"]

    return {k: tf.convert_to_tensor(v) for k, v in batch_data.items()}


def train_minimal(train_path: str, validation_path: str, embed_npz: str, seq_csv: str,
                  embd_key_path: str, out_dir: str,
                  mhc_class: int = 1, num_anchors: int = 2,
                  epochs: int = 3, batch_size: int = 32, lr: float = 1e-4,
                  embed_dim: int = 32, heads: int = 8, noise_std: float = 0.1):
    """
    Minimal training function for combined reconstruction and classification.
    """
    global EMB_DB
    EMB_DB = load_embedding_db(embed_npz)

    # Load data and mappings
    seq_map = pd.read_csv(seq_csv, index_col="allele")["mhc_sequence"].to_dict()
    embed_map = pd.read_csv(embd_key_path, index_col="key")["mhc_sequence"].to_dict()
    seq_map = {clean_key(k): v for k, v in seq_map.items()}

    df_train = pq.ParquetFile(train_path).read().to_pandas()
    df_val = pq.ParquetFile(validation_path).read().to_pandas()
    print(f"Loaded {len(df_train)} training, {len(df_val)} validation samples.")

    # Calculate max lengths
    max_pep_len = int(pd.concat([df_train["long_mer"], df_val["long_mer"]]).str.len().max())
    max_mhc_len = 500 if mhc_class == 2 else int(next(iter(EMB_DB.values())).shape[0])
    print(f"Max peptide length: {max_pep_len}, Max MHC length: {max_mhc_len}")

    # Initialize model
    model = pmbind_anchor_extractor(
            max_pep_len,
            max_mhc_len,
            embed_dim,
            heads,
            noise_std,
            num_anchors,
            MASK_TOKEN,
            PAD_TOKEN
        )

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    bce_loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    train_auc = tf.keras.metrics.AUC(name="train_auc")
    val_auc = tf.keras.metrics.AUC(name="val_auc")
    best_val_auc = 0.0
    os.makedirs(out_dir, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
    # build the model
    model.build(input_shape={
        "pep_onehot": (None, max_pep_len, 21),
        "pep_mask": (None, max_pep_len),
        "mhc_emb": (None, max_mhc_len, 1152),
        "mhc_mask": (None, max_mhc_len),
    })

    model.summary()

    # Get indices for training and validation
    train_indices = np.arange(len(df_train))
    val_indices = np.arange(len(df_val))

    # Create a fixed validation batch for periodic evaluation
    fixed_val_batch_df = df_val.sample(n=batch_size, random_state=42)
    fixed_val_batch = rows_to_tensors(fixed_val_batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)

    for epoch in range(epochs):
        np.random.shuffle(train_indices)
        # Training loop
        for start in tqdm(range(0, len(train_indices), batch_size), desc=f"Epoch {epoch+1}/{epochs} - Training"):
            end = start + batch_size
            batch_idx = train_indices[start:end]
            batch_df = df_train.iloc[batch_idx]
            batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)

            with tf.GradientTape() as tape:
                outputs = model(batch_data, training=True)
                y_true = batch_data["labels"]
                y_pred = outputs["cls_ypred"]
                loss = bce_loss_fn(y_true, y_pred)
            grads = tape.gradient(loss, model.trainable_variables)
            print(f"anchor_positions: {outputs['anchor_positions']}")
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss(loss)
            train_auc(y_true, y_pred)

        # Validation loop
        val_outputs = model(fixed_val_batch, training=False)
        y_true_val = fixed_val_batch["labels"]
        y_pred_val = val_outputs["cls_ypred"]
        v_loss = bce_loss_fn(y_true_val, y_pred_val)

        val_loss(v_loss)
        val_auc(y_true_val, y_pred_val)

        # Log metrics
        history["train_loss"].append(train_loss.result().numpy())
        history["val_loss"].append(val_loss.result().numpy())
        history["train_auc"].append(train_auc.result().numpy())
        history["val_auc"].append(val_auc.result().numpy())

        print(f"Epoch {epoch+1}, Train Loss: {train_loss.result():.4f}, Val Loss: {val_loss.result():.4f}, "
              f"Train AUC: {train_auc.result():.4f}, Val AUC: {val_auc.result():.4f}")

        # Save best model
        if val_auc.result() > best_val_auc:
            best_val_auc = val_auc.result()
            model.save_weights(os.path.join(out_dir, "best_model.weights.h5"))
            print(f"Best model saved with Val AUC: {best_val_auc:.4f}")

        # Reset metrics
        train_loss.reset_state()
        val_loss.reset_state()
        train_auc.reset_state()
        val_auc.reset_state()

    # Save final model and training history
    model.save_weights(os.path.join(out_dir, "final_model.weights.h5"))
    # Convert all values in history to native Python types for JSON serialization
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history_serializable, f, indent=4)

    print("Training complete. Final model and history saved.")

    # Visualizations
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_auc"], label="Train AUC")
    plt.plot(history["val_auc"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Training and Validation AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_history.png"))
    plt.close()

    # Evaluate on full validation set for confusion matrix
    all_val_data = rows_to_tensors(df_val, max_pep_len, max_mhc_len, seq_map, embed_map)
    # Evaluate on full validation set for confusion matrix in batches to avoid OOM
    all_val_outputs_list = []
    for i in range(0, len(df_val), batch_size):
        batch_data = rows_to_tensors(df_val.iloc[i:i + batch_size], max_pep_len, max_mhc_len, seq_map, embed_map)
        batch_outputs = model(batch_data, training=False)
        all_val_outputs_list.append({k: v.numpy() for k, v in batch_outputs.items()})
    all_val_outputs = {k: np.concatenate([d[k] for d in all_val_outputs_list], axis=0) for k in all_val_outputs_list[0]}
    all_y_true = all_val_data["labels"].numpy()
    all_y_pred = (all_val_outputs["cls_ypred"] >= 0.5).astype(int)

    # outputs = {
    #     "cls_ypred": y_pred,
    #     "anchor_positions": inds,
    #     "anchor_weights": weights,
    #     "anchor_embeddings": outs,
    #     "peptide_cross_attn_scores": peptide_cross_attn_scores,
    #     "barcode_out": barcode_out,
    #     "barcode_att": barcode_att
    # },

    # run prediction on validation set and get confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(all_y_true, all_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.get_cmap("Blues"), values_format='d')
    plt.title("Confusion Matrix on Validation Set")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    # visualize # Barcode Attention Heatmap, take 10 random samples from validation set
    sample_indices = np.random.choice(len(df_val), size=10, replace=False)
    sample_df = df_val.iloc[sample_indices]
    sample_data = rows_to_tensors(sample_df, max_pep_len, max_mhc_len, seq_map, embed_map)
    sample_outputs = model(sample_data, training=False)
    barcode_att = sample_outputs["barcode_att"].numpy()  # (batch_size, num_anchors, mhc_len)

    # generate sns heatmap, and put all samples in one figure
    # Reshape: (samples, anchors * mhc_len)
    batch_size, one, pep_len = barcode_att.shape
    heatmap_data = barcode_att.reshape(batch_size, pep_len)

    plt.figure(figsize=(30,15))
    sns.heatmap(heatmap_data, cmap="viridis", cbar=True)
    plt.xlabel("Anchor x MHC Position")
    plt.ylabel("Sample")
    plt.title("Barcode Attention Heatmap (Samples in Rows)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "barcode_attention_heatmap.png"))
    plt.close()



if __name__ == "__main__":
    # --- Configuration ---
    MHC_CLASS = 1  # Set MHC class (1 or 2)
    config = {
        "MHC_CLASS": 1,
        "EPOCHS": 2,
        "BATCH_SIZE": 128,
        "LEARNING_RATE": 1e-3,
        "EMBED_DIM": 4,
        "HEADS": 2,
        "NOISE_STD": 0.1,
    }

    # --- Paths ---
    paths = {
        "train": f"../tests/binding_affinity_dataset_with_swapped_negatives{config['MHC_CLASS']}_train.parquet",
        "val": f"../tests/binding_affinity_dataset_with_swapped_negatives{config['MHC_CLASS']}_val.parquet",
        "embed_npz": f"/media/amirreza/lasse/ESM/esmc_600m/PMGen_whole_seq/mhc{config['MHC_CLASS']}_encodings.npz",
        "embed_key": f"/media/amirreza/lasse/ESM/esmc_600m/PMGen_whole_seq/mhc{config['MHC_CLASS']}_encodings.csv",
        "seq_csv": f"../data/alleles/aligned_PMGen_class_{config['MHC_CLASS']}.csv",
        "out_dir": f"../outputs/minimal_run_mhc{config['MHC_CLASS']}",
        "pretrained_pmclust": f"/media/amirreza/lasse/PMClust_runs/run_PMClust_ns_0.1_hds_4_zdim_21_L1_all/1/best_model.weights.h5"
    }

    train_minimal(
        train_path=paths["train"],
        validation_path=paths["val"],
        embed_npz=paths["embed_npz"],
        seq_csv=paths["seq_csv"],
        embd_key_path=paths["embed_key"],
        out_dir=paths["out_dir"],
        mhc_class=config["MHC_CLASS"],
        num_anchors=2,
        epochs=config["EPOCHS"],
        batch_size=config["BATCH_SIZE"],
        lr=config["LEARNING_RATE"],
        embed_dim=config["EMBED_DIM"],
        heads=config["HEADS"],
        noise_std=config["NOISE_STD"]
    )