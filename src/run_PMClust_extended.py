#!/usr/bin/env python
"""
Minimal GradientTape training loop for `pmclust_subtract` **without using `tf.data.Dataset`**.
------------------------------------------------------------------------------------------------
This script shows the bare-bones path through your model:

1.  Load all rows from a Parquet file into **pandas**.
2.  Shuffle & slice with NumPy to create mini-batches.
3.  Convert every mini-batch to tensors on-the-fly and feed it through a
    `tf.GradientTape` loop.

Everything runs in eager mode – no `Dataset.from_generator`, no
prefetching pipelines – so the control-flow is crystal-clear when you
step through with a debugger.

NOTE: This version has been made self-contained and runnable. It includes
the user's model structure with fixes, placeholder custom layers, and new
visualizations.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pyarrow.parquet as pq
import os

# Added for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import umap  # Make sure you have umap-learn installed: pip install umap-learn

# Assuming these are in a 'utils' directory relative to the script
from utils import seq_to_onehot, get_embed_key, NORM_TOKEN, MASK_TOKEN, PAD_TOKEN, PAD_VALUE, \
    masked_categorical_crossentropy, OHE_to_seq_single, split_y_true_y_pred, OHE_to_seq, clean_key

# Assuming this is in a 'models' directory
from models import pmclust_subtract

# ──────────────────────────────────────────────────────────────────────
# 4. DATA PREPARATION & TRAINING LOOP (Corrected)
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
        "pep_mask": np.zeros((n, max_pep_len), np.float32),
        "mhc_emb": np.zeros((n, max_mhc_len, 1152), np.float32),
        "mhc_mask": np.zeros((n, max_mhc_len), np.float32),
        "mhc_onehot": np.zeros((n, max_mhc_len, 21), np.float32),  # Target
    }

    for i, (_, r) in enumerate(rows.iterrows()):
        pep_seq = r["long_mer"].upper()
        pep_OHE = seq_to_onehot(pep_seq, max_seq_len=max_pep_len)
        batch_data["pep_onehot"][i] = pep_OHE
        pep_mask = np.ones((max_pep_len,), dtype=np.float32)
        pep_mask[np.all(pep_OHE == PAD_VALUE, axis=-1)] = PAD_TOKEN  # Set padding to PAD_TOKEN
        batch_data["pep_mask"][i] = pep_mask

        embd_key = get_embed_key(clean_key(r["mhc_embedding_key"]), embed_map)
        emb = EMB_DB[embd_key]
        L = emb.shape[0]
        batch_data["mhc_emb"][i, :L] = emb
        mhc_mask = np.ones((max_mhc_len,), dtype=np.float32)
        mhc_mask[np.all(emb == PAD_VALUE, axis=-1)] = PAD_TOKEN  # Set padding to PAD_TOKEN
        batch_data["mhc_mask"][i] = mhc_mask
        # locate the row in seq_map that has the same key_norm
        key_norm = get_embed_key(clean_key(r["mhc_embedding_key"]), seq_map)
        mhc_seq = seq_map[key_norm]
        batch_data["mhc_onehot"][i] = seq_to_onehot(mhc_seq, max_seq_len=max_mhc_len)

    return {k: tf.convert_to_tensor(v) for k, v in batch_data.items()}


def run_visualizations(df_all, latents, enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, out_dir):
    """
    Generates and saves a series of visualizations for model analysis.
    """
    print("\nGenerating visualizations...")

    # 1. Extract labels from alleles
    alleles = df_all['mhc_embedding_key'].apply(clean_key).astype('category')
    allele_labels = alleles.cat.codes
    unique_alleles = alleles.cat.categories
    print(f"Found {len(unique_alleles)} unique alleles.")

    # --- Visualization 1: UMAP of latents, colored by allele ---
    print("Running UMAP on latents...")
    # Flatten the latent space for UMAP: (n_samples, max_mhc_len, embed_dim) -> (n_samples, max_mhc_len * embed_dim)
    latents_flat = latents.reshape(latents.shape[0], -1)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(latents_flat)

    # Select top 5 most frequent alleles for clarity in the plot
    top_5_alleles = df_all['mhc_embedding_key'].apply(clean_key).value_counts().nlargest(5).index
    df_plot = pd.DataFrame({'UMAP1': embedding[:, 0], 'UMAP2': embedding[:, 1], 'allele': alleles})
    df_plot_subset = df_plot[df_plot['allele'].isin(top_5_alleles)]

    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df_plot_subset, x='UMAP1', y='UMAP2', hue='allele', s=50, alpha=0.7, palette='viridis')
    plt.title('UMAP of Latent Space (Top 5 Alleles)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Allele', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "umap_latents_by_allele.png"))
    plt.close()
    print("✓ UMAP plot saved.")

    # --- Visualization 2: Show latent of one sample ---
    plt.figure(figsize=(12, 6))
    sns.heatmap(latents[0], cmap='viridis')
    plt.title(f'Latent Representation of Sample 0 (Allele: {alleles[0]})')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Sequence Position')
    plt.savefig(os.path.join(out_dir, "single_sample_latent.png"))
    plt.close()
    print("✓ Single latent plot saved.")

    # --- Visualization 3: Compare mean latents for top 5 alleles ---
    mean_latents = []
    for allele_name in top_5_alleles:
        indices = df_all[df_all['mhc_embedding_key'].apply(clean_key) == allele_name].index
        mean_latent = latents[indices].mean(axis=0)
        mean_latents.append(mean_latent)

    mean_latents_stack = np.stack(mean_latents)

    fig, axes = plt.subplots(len(top_5_alleles), 1, figsize=(10, 2 * len(top_5_alleles)), sharex=True)
    fig.suptitle('Mean Latent Representation per Allele', fontsize=16)
    for i, allele_name in enumerate(top_5_alleles):
        sns.heatmap(mean_latents[i], ax=axes[i], cmap='viridis')
        axes[i].set_title(allele_name)
        axes[i].set_ylabel('Sequence Pos')
    axes[-1].set_xlabel('Embedding Dim')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, "mean_latents_comparison.png"))
    plt.close()
    print("✓ Mean latents comparison plot saved.")

    # --- Visualizations 4 & 5: Show one sample of peptide/MHC input and masks ---
    sample_idx = 0
    sample_row = df_all.iloc[[sample_idx]]
    sample_data = rows_to_tensors(sample_row, max_pep_len, max_mhc_len, seq_map, embed_map)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle(f'Input Data for Sample {sample_idx}', fontsize=16)

    sns.heatmap(sample_data['pep_onehot'][0].numpy().T, ax=axes[0, 0], cmap='gray_r')
    axes[0, 0].set_title('Peptide Input (One-Hot)')
    axes[0, 0].set_ylabel('Amino Acid')
    axes[0, 0].set_xlabel('Sequence Position')

    sns.heatmap(sample_data['pep_mask'][0].numpy()[np.newaxis, :], ax=axes[0, 1], cmap='viridis', cbar=False)
    axes[0, 1].set_title('Peptide Mask')
    axes[0, 1].set_yticks([])

    sns.heatmap(sample_data['mhc_emb'][0].numpy().T, ax=axes[1, 0], cmap='viridis')
    axes[1, 0].set_title('MHC Input (Embedding)')
    axes[1, 0].set_ylabel('Embedding Dim')
    axes[1, 0].set_xlabel('Sequence Position')

    sns.heatmap(sample_data['mhc_mask'][0].numpy()[np.newaxis, :], ax=axes[1, 1], cmap='viridis', cbar=False)
    axes[1, 1].set_title('MHC Mask')
    axes[1, 1].set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, "input_and_mask_samples.png"))
    plt.close()
    print("✓ Input and mask plots saved.")

    # --- Visualization 6: Print 5 predictions and compare them with inputs ---
    print("\n--- Comparing 5 Predictions with Inputs ---")
    pred_samples_df = df_all.head(5)
    pred_data = rows_to_tensors(pred_samples_df, max_pep_len, max_mhc_len, seq_map, embed_map)

    model_inputs = {
        "pep_onehot": pred_data["pep_onehot"],
        "pep_mask": pred_data["pep_mask"],
        "mhc_emb": pred_data["mhc_emb"],
        "mhc_mask": pred_data["mhc_mask"],
        "mhc_onehot": pred_data["mhc_onehot"],
    }

    true_preds = enc_dec(model_inputs, training=False)

    pep_true, pep_pred_ohe = split_y_true_y_pred(true_preds["pep_ytrue_ypred"].numpy())
    mhc_true, mhc_pred_ohe = split_y_true_y_pred(true_preds["mhc_ytrue_ypred"].numpy())

    for i in range(5):
        original_peptide = OHE_to_seq_single(pep_true[i])
        predicted_peptide = OHE_to_seq_single(pep_pred_ohe[i])

        original_mhc = OHE_to_seq_single(mhc_true[i])
        predicted_mhc = OHE_to_seq_single(mhc_pred_ohe[i])

        print(f"\n--- Sample {i} ---")
        print(f"  Allele: {alleles[i]}")
        print(f"  Original Peptide : {original_peptide}")
        print(f"  Predicted Peptide: {predicted_peptide}")
        print(f"  Original MHC     : {original_mhc}")
        print(f"  Predicted MHC    : {predicted_mhc}")
    print("\n✓ Visualizations complete.")


def train(parquet_path: str, embed_npz: str, seq_csv: str, embd_key_path: str,
          out_dir: str, epochs: int = 3, batch_size: int = 32, lr: float = 1e-4, embed_dim: int = 32):
    global EMB_DB
    EMB_DB = load_embedding_db(embed_npz)

    seq_map = pd.read_csv(seq_csv, index_col="allele")["mhc_sequence"].to_dict()
    embed_map = pd.read_csv(embd_key_path, index_col="key")["mhc_sequence"].to_dict()

    # harmonize seq_map keys using clean_keys function, update the keys in seq_map
    seq_map = {clean_key(k): v for k, v in seq_map.items()}

    df_all = pq.ParquetFile(parquet_path).read().to_pandas()

    max_pep_len = df_all["long_mer"].str.len().max()
    max_mhc_len = next(iter(EMB_DB.values())).shape[0]

    enc_dec = pmclust_subtract(max_pep_len,
                               max_mhc_len,
                               emb_dim=embed_dim,
                               heads=8,
                               mask_token=MASK_TOKEN,
                               pad_token=PAD_TOKEN)

    opt = keras.optimizers.Adam(lr)

    loss_fn = masked_categorical_crossentropy
    indices = np.arange(len(df_all))

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        print(f"\nEpoch {epoch}/{epochs}")

        for step in range(0, len(indices), batch_size):
            batch_idx = indices[step:step + batch_size]
            batch_df = df_all.iloc[batch_idx]
            batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)

            model_inputs = {
                "pep_onehot": batch_data["pep_onehot"],
                "pep_mask": batch_data["pep_mask"],
                "mhc_emb": batch_data["mhc_emb"],
                "mhc_mask": batch_data["mhc_mask"],
                "mhc_onehot": batch_data["mhc_onehot"],
            }

            with tf.GradientTape() as tape:
                true_and_preds = enc_dec(model_inputs, training=True)

                pep_loss_per_sample = loss_fn(
                    true_and_preds["pep_ytrue_ypred"],
                    batch_data["pep_mask"]
                )
                mhc_loss_per_sample = loss_fn(
                    true_and_preds["mhc_ytrue_ypred"],
                    batch_data["mhc_mask"]
                )

                pep_loss = tf.reduce_mean(pep_loss_per_sample)
                mhc_loss = tf.reduce_mean(mhc_loss_per_sample)

                loss = pep_loss + mhc_loss

            grads = tape.gradient(loss, enc_dec.trainable_variables)
            opt.apply_gradients(zip(grads, enc_dec.trainable_variables))

            if step % (batch_size * 2) == 0:
                print(
                    f"step {step // batch_size:4d}  loss={loss.numpy():.4f} (pep={pep_loss.numpy():.4f}, mhc={mhc_loss.numpy():.4f})")

    weights_path = os.path.join(out_dir, "enc_dec.weights.h5")
    enc_dec.save_weights(weights_path)
    print(f"✓ Training finished. Weights saved to {weights_path}")

    # --- INFERENCE & LATENT EXTRACTION ---
    # Re-shuffle indices to match the original dataframe order for visualization
    indices = np.arange(len(df_all))
    latents = np.zeros((len(df_all), max_mhc_len, embed_dim), np.float32)
    print("\nRunning inference on the training data to get latents...")
    for step in range(0, len(indices), batch_size):
        batch_idx = indices[step:step + batch_size]
        batch_df = df_all.iloc[batch_idx]
        batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)

        model_inputs = {
            "pep_onehot": batch_data["pep_onehot"],
            "pep_mask": batch_data["pep_mask"],
            "mhc_emb": batch_data["mhc_emb"],
            "mhc_mask": batch_data["mhc_mask"],
            "mhc_onehot": batch_data["mhc_onehot"],
        }

        true_preds = enc_dec(model_inputs, training=False)
        latents[batch_idx] = true_preds["cross_latent"].numpy()

    latents_path = os.path.join(out_dir, "mhc_latents.npy")
    np.save(latents_path, latents)
    print(f"✓ Latents saved to {latents_path}")

    # --- CALL THE NEW VISUALIZATION FUNCTION ---
    run_visualizations(df_all, latents, enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, out_dir)


if __name__ == "__main__":
    # Suppress verbose TensorFlow logging, but keep errors.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    parquet_path = "../data/binding_affinity_data/positives_class1_subset.parquet"
    embed_npz_path = "/media/amirreza/lasse/trash/mhc1_encodings.npz"
    embd_key_path = "/media/amirreza/lasse/trash/mhc1_encodings.csv"
    seq_csv_path = "../data/alleles/aligned_PMGen_class_1.csv"
    out_dir = "run_PMClust"
    os.makedirs(out_dir, exist_ok=True)

    train(
        parquet_path=parquet_path,
        embed_npz=embed_npz_path,
        seq_csv=seq_csv_path,
        embd_key_path=embd_key_path,
        out_dir=out_dir,
        epochs=2,
        batch_size=32,  # Reduced for faster dummy run
    )
