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


# Added for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import umap  # Make sure you have umap-learn installed: pip install umap-learn

# Added for automated DBSCAN eps estimation
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


# Assuming these are in a 'utils' directory relative to the script
from utils import seq_to_onehot, get_embed_key, NORM_TOKEN, MASK_TOKEN, PAD_TOKEN, PAD_VALUE, MASK_VALUE, \
    masked_categorical_crossentropy, OHE_to_seq_single, split_y_true_y_pred, OHE_to_seq, clean_key

# Assuming this is in a 'models' directory
from models import pmclust_subtract, pmclust_cross_attn
from sklearn.cluster import DBSCAN

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
        "mhc_onehot": np.zeros((n, max_mhc_len, 21), np.float32),  # Target
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
        
        # Randomly mask 15% of valid peptide positions with MASK_TOKEN
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
        
        # Randomly mask 15% of valid MHC positions with MASK_TOKEN
        valid_mhc_positions = np.where(batch_data["mhc_mask"][i] == NORM_TOKEN)[0]
        if len(valid_mhc_positions) > 0:
            mask_fraction = 0.15
            n_mask = max(1, int(mask_fraction * len(valid_mhc_positions)))  # At least 1 position
            mask_indices = np.random.choice(valid_mhc_positions, size=n_mask, replace=False)
            batch_data["mhc_mask"][i, mask_indices] = MASK_TOKEN  # Masked positions get MASK_TOKEN (-1.0)
            # Zero out the corresponding embeddings for masked positions
            batch_data["mhc_emb"][i, mask_indices, :] = MASK_VALUE

        # print(f"MHC mask for sample {i}: {batch_data['mhc_mask'][i]}")  # Debugging line to check MHC mask
        
        # Get MHC sequence and convert to one-hot
        key_norm = get_embed_key(clean_key(r["mhc_embedding_key"]), seq_map)
        mhc_seq = seq_map[key_norm]
        batch_data["mhc_onehot"][i] = seq_to_onehot(mhc_seq, max_seq_len=max_mhc_len)

    return {k: tf.convert_to_tensor(v) for k, v in batch_data.items()}


def run_visualizations(df, latents_seq, latents_pooled, enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, out_dir):
    """
    Generates and saves a series of visualizations for model analysis.
    """
    print("\nGenerating visualizations...")

    # Print masking statistics for debugging
    print("\n--- Masking Statistics ---")
    sample_batch_df = df.head(5)
    sample_data = rows_to_tensors(sample_batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)
    
    pep_masks = sample_data["pep_mask"].numpy()
    mhc_masks = sample_data["mhc_mask"].numpy()
    
    print(f"Peptide mask values - PAD_TOKEN: {PAD_TOKEN}, MASK_TOKEN: {MASK_TOKEN}, NORM_TOKEN: {NORM_TOKEN}")
    print(f"Sample peptide shape: {sample_data['pep_onehot'].shape}")
    print(f"Sample peptide mask shape: {pep_masks.shape}")
    print(f"Unique values in peptide masks: {np.unique(pep_masks)}")
    
    print(f"Sample MHC embedding shape: {sample_data['mhc_emb'].shape}")
    print(f"Sample MHC mask shape: {mhc_masks.shape}")
    print(f"Unique values in MHC masks: {np.unique(mhc_masks)}")

    # 1. Extract labels from alleles
    alleles = df['mhc_embedding_key'].apply(clean_key).astype('category')
    allele_labels = alleles.cat.codes
    unique_alleles = alleles.cat.categories
    allele_counts = df['mhc_embedding_key'].apply(clean_key).value_counts()
    print(f"Found {len(unique_alleles)} unique alleles.")

    # Flatten the sequential latents for UMAP and DBSCAN
    n_samples = latents_seq.shape[0]
    latents_seq_flat = latents_seq.reshape(n_samples, -1)

    # Create a publication-friendly color palette for all alleles
    np.random.seed(42)  # for reproducibility
    
    # Use colorblind-friendly palettes appropriate for scientific publications
    if len(unique_alleles) <= 8:
        # For small number of alleles use Set2 (colorblind-friendly)
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_alleles)))
    elif len(unique_alleles) <= 20:
        # For medium number of alleles
        colors = plt.cm.tab20c(np.linspace(0, 1, len(unique_alleles)))
    else:
        # For large number of alleles use viridis (perceptually uniform, colorblind-friendly)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_alleles)))
        
    allele_color_map = {allele: colors[i] for i, allele in enumerate(unique_alleles)}
    
    # Choose 3 random unique alleles to highlight
    random_alleles_to_mark = np.random.choice(unique_alleles, min(3, len(unique_alleles)), replace=False)
    print(f"Highlighting alleles: {list(random_alleles_to_mark)}")

    # --- Visualization 1: UMAP of FLATTENED SEQUENTIAL latents ---
    print("Running UMAP on FLATTENED SEQUENTIAL latents...")
    reducer_seq = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_seq = reducer_seq.fit_transform(latents_seq_flat)

    plt.figure(figsize=(14, 10))
    
    # Plot all points with assigned colors
    for allele in unique_alleles:
        mask = alleles == allele
        plt.scatter(embedding_seq[mask, 0], embedding_seq[mask, 1], 
                   c=[allele_color_map[allele]], label=allele, 
                   s=5, alpha=0.7)
    
    # Highlight the 3 random alleles with distinct markers
    for i, allele in enumerate(random_alleles_to_mark):
        mask = alleles == allele
        markers = ['*', 'D', 'X']  # More publication-friendly markers
        plt.scatter(embedding_seq[mask, 0], embedding_seq[mask, 1], 
                   c='gray', marker=markers[i], s=25, 
                   edgecolor='black', linewidth=0.3, 
                   label=f'{allele} (highlighted)')
    
    plt.title(f'UMAP of Flattened Sequential Latents ({len(df)} Samples)\nColored by {len(unique_alleles)} Unique Alleles')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    
    # Create a better legend for scientific publication
    if len(unique_alleles) > 20:
        # For many alleles, create a compact legend with multiple columns
        handles, labels = plt.gca().get_legend_handles_labels()
        
        # Separate highlighted alleles from regular alleles
        highlight_handles = [h for h, l in zip(handles, labels) if 'highlighted' in l]
        highlight_labels = [l for l in labels if 'highlighted' in l]
        
        regular_handles = [h for h, l in zip(handles, labels) if 'highlighted' not in l]
        regular_labels = [l for l in labels if 'highlighted' not in l]
        
        # First legend for highlighted alleles
        first_legend = plt.legend(highlight_handles, highlight_labels, 
                                 title='Highlighted Alleles',
                                 bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.gca().add_artist(first_legend)
        
        # Second legend for regular alleles in multiple columns
        plt.legend(regular_handles, regular_labels, 
                  title='All Alleles',
                  bbox_to_anchor=(1.05, 0.5), loc='center left',
                  ncol=max(1, len(regular_labels) // 30),  # Adjust columns based on count
                  fontsize=8)
    else:
        # For fewer alleles, a single legend is sufficient
        plt.legend(title='Allele', bbox_to_anchor=(1.05, 1), 
                  loc='upper left', fontsize=8, frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "umap_sequential_latents.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ UMAP plot of sequential latents saved.")

    # --- Visualization 2: UMAP of POOLED latents ---
    print("Running UMAP on POOLED latents...")
    reducer_pooled = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_pooled = reducer_pooled.fit_transform(latents_pooled)

    plt.figure(figsize=(14, 10))
    
    # Plot all points with assigned colors
    for allele in unique_alleles:
        mask = alleles == allele
        plt.scatter(embedding_pooled[mask, 0], embedding_pooled[mask, 1], 
                   c=[allele_color_map[allele]], label=allele, 
                   s=5, alpha=0.7)
    
    # Highlight the 3 random alleles with distinct markers
    for i, allele in enumerate(random_alleles_to_mark):
        mask = alleles == allele
        markers = ['*', 'D', 'X']  # More publication-friendly markers
        plt.scatter(embedding_pooled[mask, 0], embedding_pooled[mask, 1], 
                   c='gray', marker=markers[i], s=25, 
                   edgecolor='black', linewidth=0.3, 
                   label=f'{allele} (highlighted)')
    
    plt.title(f'UMAP of Pooled Latents ({len(df)} Samples)\nColored by {len(unique_alleles)} Unique Alleles')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    
    # Create a better legend for scientific publication
    if len(unique_alleles) > 20:
        # For many alleles, create a compact legend with multiple columns
        handles, labels = plt.gca().get_legend_handles_labels()
        
        # Separate highlighted alleles from regular alleles
        highlight_handles = [h for h, l in zip(handles, labels) if 'highlighted' in l]
        highlight_labels = [l for l in labels if 'highlighted' in l]
        
        regular_handles = [h for h, l in zip(handles, labels) if 'highlighted' not in l]
        regular_labels = [l for l in labels if 'highlighted' not in l]
        
        # First legend for highlighted alleles
        first_legend = plt.legend(highlight_handles, highlight_labels, 
                                 title='Highlighted Alleles',
                                 bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.gca().add_artist(first_legend)
        
        # Second legend for regular alleles in multiple columns
        plt.legend(regular_handles, regular_labels, 
                  title='All Alleles',
                  bbox_to_anchor=(1.05, 0.5), loc='center left',
                  ncol=max(1, len(regular_labels) // 30),  # Adjust columns based on count
                  fontsize=8)
    else:
        # For fewer alleles, a single legend is sufficient
        plt.legend(title='Allele', bbox_to_anchor=(1.05, 1), 
                  loc='upper left', fontsize=8, frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "umap_pooled_latents.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ UMAP plot of pooled latents saved.")

    # --- Visualization 3: DBSCAN clustering on FLATTENED SEQUENTIAL latents ---
    print("Running DBSCAN on FLATTENED SEQUENTIAL latents with automated eps estimation...")
    
    # --- Step 1: Automate eps estimation using k-distance graph ---
    min_samples = 10 
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(latents_seq_flat)
    distances, indices = neighbors_fit.kneighbors(latents_seq_flat)
    
    k_distances = np.sort(distances[:, min_samples-1], axis=0)
    kneedle = KneeLocator(range(len(k_distances)), k_distances, curve="convex", direction="increasing")
    estimated_eps = kneedle.elbow_y
    
    if estimated_eps is None:
        print("Warning: Could not automatically determine eps. Falling back to a default value.")
        estimated_eps = np.median(k_distances) 
    
    print(f"Estimated optimal eps for DBSCAN: {estimated_eps:.4f}")
    
    plt.figure(figsize=(10, 6))
    kneedle.plot_knee()
    plt.xlabel("Points (sorted by distance)")
    plt.ylabel(f"Distance to {min_samples}-th nearest neighbor")
    plt.title("K-Distance Graph for DBSCAN Epsilon Estimation")
    plt.grid(True, linestyle='--', alpha=0.7)  # Add grid for better readability
    plt.savefig(os.path.join(out_dir, "dbscan_k_distance_graph.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Step 2: Run DBSCAN with the estimated eps ---
    dbscan = DBSCAN(eps=estimated_eps, min_samples=min_samples, n_jobs=-1)
    clusters = dbscan.fit_predict(latents_seq_flat)
    
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = np.sum(clusters == -1)
    print(f"✓ DBSCAN found {n_clusters} clusters and {n_noise} noise points.")

    # --- Step 3: Visualize the clustering results ---
    plt.figure(figsize=(14, 10))
    
    # Create publication-friendly color map for clusters
    unique_clusters = sorted(set(clusters))
    
    if len(unique_clusters) <= 8:
        # For small number of clusters, use a qualitative colormap
        cluster_colors = plt.cm.Dark2(np.linspace(0, 1, len(unique_clusters)))
    else:
        # For more clusters, use a sequential/diverging colormap
        cluster_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
        
    # Make noise points gray
    if -1 in unique_clusters:
        noise_idx = unique_clusters.index(-1)
        cluster_colors[noise_idx] = [0.7, 0.7, 0.7, 0.5]  # Light gray with lower alpha
        
    cluster_color_map = {cluster: cluster_colors[i] for i, cluster in enumerate(unique_clusters)}
    
    # Plot all points colored by cluster
    for cluster in unique_clusters:
        mask = clusters == cluster
        label = f'Cluster {cluster}' if cluster != -1 else 'Noise'
        plt.scatter(embedding_seq[mask, 0], embedding_seq[mask, 1], 
                   c=[cluster_color_map[cluster]], label=label, 
                   s=5, alpha=0.7)
    
    # Highlight the 3 random alleles with distinct markers
    for i, allele in enumerate(random_alleles_to_mark):
        mask = alleles == allele
        markers = ['*', 'D', 'X'] 
        plt.scatter(embedding_seq[mask, 0], embedding_seq[mask, 1], 
                   c='gray', marker=markers[i], s=25, 
                   edgecolor='black', linewidth=0.3, 
                   label=f'{allele} (highlighted)')
    
    plt.title(f'DBSCAN Clustering of Sequential Latents (Visualized with UMAP)\nFound {n_clusters} clusters, {n_noise} noise points')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    
    # Create separate legends for clusters and highlighted alleles
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Separate highlighted alleles from clusters
    highlight_handles = [h for h, l in zip(handles, labels) if 'highlighted' in l]
    highlight_labels = [l for l in labels if 'highlighted' in l]
    
    cluster_handles = [h for h, l in zip(handles, labels) if 'highlighted' not in l]
    cluster_labels = [l for l in labels if 'highlighted' not in l]
    
    # First legend for highlighted alleles
    first_legend = plt.legend(highlight_handles, highlight_labels, 
                             title='Highlighted Alleles',
                             bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().add_artist(first_legend)
    
    # Second legend for clusters
    plt.legend(cluster_handles, cluster_labels, 
              title='Clusters',
              bbox_to_anchor=(1.05, 0.6), loc='center left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "umap_dbscan_clusters_sequential.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ DBSCAN clustering plot on sequential latents saved.")


    # --- Visualization 3: Show latent of one sample ---
    plt.figure(figsize=(12, 6))
    sns.heatmap(latents_seq[0], cmap='viridis')
    plt.title(f'Latent Representation of Sample 0 (Allele: {alleles[0]})')
    plt.xlabel('Embedding Dimension'); plt.ylabel('Sequence Position')
    plt.savefig(os.path.join(out_dir, "single_sample_latent.png"))
    plt.close()
    print("✓ Single latent plot saved.")

    # --- Visualization 4: Compare mean latents for top 5 alleles ---
    top_5_alleles = allele_counts.nlargest(5).index.tolist()
    mean_latents = []
    for allele_name in top_5_alleles:
        indices = df[df['mhc_embedding_key'].apply(clean_key) == allele_name].index
        mean_latent = latents_seq[indices].mean(axis=0)
        mean_latents.append(mean_latent)

    fig, axes = plt.subplots(len(top_5_alleles), 1, figsize=(10, 2 * len(top_5_alleles)), sharex=True)
    fig.suptitle('Mean Latent Representation per Allele', fontsize=16)
    for i, allele_name in enumerate(top_5_alleles):
        sns.heatmap(mean_latents[i], ax=axes[i], cmap='viridis')
        axes[i].set_title(f"{allele_name} (n={allele_counts[allele_name]})")
        axes[i].set_ylabel('Sequence Pos')
    axes[-1].set_xlabel('Embedding Dim')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, "mean_latents_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Mean latents comparison plot saved.")

    # --- Visualizations 5 & 6: Show one sample of peptide/MHC input and masks ---
    sample_idx = 0
    sample_row = df.iloc[[sample_idx]]
    sample_data = rows_to_tensors(sample_row, max_pep_len, max_mhc_len, seq_map, embed_map)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle(f'Input Data for Sample {sample_idx}', fontsize=16)

    sns.heatmap(sample_data['pep_onehot'][0].numpy().T, ax=axes[0, 0], cmap='gray_r')
    axes[0, 0].set_title('Peptide Input (One-Hot)')
    axes[0, 0].set_ylabel('Amino Acid'); axes[0, 0].set_xlabel('Sequence Position')

    sns.heatmap(sample_data['pep_mask'][0].numpy()[np.newaxis, :], ax=axes[0, 1], cmap='viridis', cbar=False)
    axes[0, 1].set_title('Peptide Mask'); axes[0, 1].set_yticks([])

    sns.heatmap(sample_data['mhc_emb'][0].numpy().T, ax=axes[1, 0], cmap='viridis')
    axes[1, 0].set_title('MHC Input (Embedding)')
    axes[1, 0].set_ylabel('Embedding Dim'); axes[1, 0].set_xlabel('Sequence Position')

    sns.heatmap(sample_data['mhc_mask'][0].numpy()[np.newaxis, :], ax=axes[1, 1], cmap='viridis', cbar=False)
    axes[1, 1].set_title('MHC Mask'); axes[1, 1].set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, "input_and_mask_samples.png"))
    plt.close()
    print("✓ Input and mask plots saved.")

    # --- Visualization 7: Print 5 predictions and compare them with inputs ---
    print("\n--- Comparing 5 Predictions with Inputs ---")
    pred_samples_df = df.head(5)
    pred_data = rows_to_tensors(pred_samples_df, max_pep_len, max_mhc_len, seq_map, embed_map)

    model_inputs = {
        "pep_onehot": pred_data["pep_onehot"], "pep_mask": pred_data["pep_mask"],
        "mhc_emb": pred_data["mhc_emb"], "mhc_mask": pred_data["mhc_mask"],
        "mhc_onehot": pred_data["mhc_onehot"],
    }
    true_preds = enc_dec(model_inputs, training=False)

    pep_true, pep_pred_ohe = split_y_true_y_pred(true_preds["pep_ytrue_ypred"].numpy())
    mhc_true, mhc_pred_ohe = split_y_true_y_pred(true_preds["mhc_ytrue_ypred"].numpy())

    pep_masks_np = pred_data["pep_mask"].numpy()
    mhc_masks_np = pred_data["mhc_mask"].numpy()

    for i in range(5):
        print(f"\n--- Sample {i} ---")
        print(f"  Allele: {clean_key(pred_samples_df.iloc[i]['mhc_embedding_key'])}")

        # --- Peptide Processing ---
        original_peptide_full = OHE_to_seq_single(pep_true[i], gap=True).replace("X", "-")
        predicted_peptide_full = OHE_to_seq_single(pep_pred_ohe[i], gap=True).replace("X", "-")
        pep_valid_mask = (pep_masks_np[i] != PAD_TOKEN) & (np.array(list(original_peptide_full)) != '-')
        original_peptide = "".join(np.array(list(original_peptide_full))[pep_valid_mask])
        predicted_peptide = "".join(np.array(list(predicted_peptide_full))[pep_valid_mask])
        print(f"  Original Peptide : {original_peptide}")
        print(f"  Predicted Peptide: {predicted_peptide}")

        # --- MHC Processing ---
        original_mhc_full = OHE_to_seq_single(mhc_true[i], gap=True).replace("X", "-")
        predicted_mhc_full = OHE_to_seq_single(mhc_pred_ohe[i], gap=True).replace("X", "-")
        mhc_valid_mask = (mhc_masks_np[i] != PAD_TOKEN) & (np.array(list(original_mhc_full)) != '-')
        original_mhc = "".join(np.array(list(original_mhc_full))[mhc_valid_mask])
        predicted_mhc = "".join(np.array(list(predicted_mhc_full))[mhc_valid_mask])
        print(f"  Original MHC     : {original_mhc}")
        print(f"  Predicted MHC    : {predicted_mhc}")

    print("\n✓ Visualizations complete.")


def train(train_path: str, validation_path: str, test_path: str,  embed_npz: str, seq_csv: str, embd_key_path: str,
          out_dir: str, epochs: int = 3, batch_size: int = 32,
            lr: float = 1e-4, embed_dim: int = 32, heads: int = 8, noise_std: float = 0.1):
    """
    Trains the model on the training set, validates on the validation set,
    and finally runs inference and generates visualizations for train, validation, and test sets.
    """
    global EMB_DB
    EMB_DB = load_embedding_db(embed_npz)

    seq_map = pd.read_csv(seq_csv, index_col="allele")["mhc_sequence"].to_dict()
    embed_map = pd.read_csv(embd_key_path, index_col="key")["mhc_sequence"].to_dict()
    seq_map = {clean_key(k): v for k, v in seq_map.items()}

    # Load all datasets
    print("Loading datasets...")
    df_train = pq.ParquetFile(train_path).read().to_pandas()
    df_val = pq.ParquetFile(validation_path).read().to_pandas()
    df_test = pq.ParquetFile(test_path).read().to_pandas()
    print(f"Loaded {len(df_train)} training, {len(df_val)} validation, and {len(df_test)} test samples.")

    # Calculate max lengths across all datasets to ensure consistency
    max_pep_len = int(pd.concat([df_train["long_mer"], df_val["long_mer"], df_test["long_mer"]]).str.len().max())
    max_mhc_len = int(next(iter(EMB_DB.values())).shape[0])
    print(f"Max peptide length: {max_pep_len}, Max MHC length: {max_mhc_len}")

    # Initialize model and optimizer
    enc_dec = pmclust_subtract(
        max_pep_len,
        max_mhc_len,
        emb_dim=embed_dim,
        heads=heads,
        mask_token=MASK_TOKEN,
        pad_token=PAD_TOKEN,
        noise_std=noise_std)
    
    opt = keras.optimizers.Adam(lr)
    loss_fn = masked_categorical_crossentropy

    # Get indices for training and validation
    train_indices = np.arange(len(df_train))
    val_indices = np.arange(len(df_val))

    # Create a fixed validation batch for periodic evaluation
    fixed_val_batch_df = df_val.sample(n=batch_size, random_state=42)
    fixed_val_batch = rows_to_tensors(fixed_val_batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)

    # --- Training Loop ---
    history = {
        'train_loss': [], 'train_pep_loss': [], 'train_mhc_loss': [],
        'val_loss': [], 'val_pep_loss': [], 'val_mhc_loss': []
    }
    best_val_loss = float('inf')
    best_weights_path = os.path.join(out_dir, "best_enc_dec.weights.h5")

    for epoch in range(1, epochs + 1):
        np.random.shuffle(train_indices)
        print(f"\nEpoch {epoch}/{epochs}")
        
        # --- Training Step ---
        epoch_loss_sum, epoch_pep_loss_sum, epoch_mhc_loss_sum, num_steps = 0, 0, 0, 0
        for step in range(0, len(train_indices), batch_size):
            batch_idx = train_indices[step:step + batch_size]
            batch_df = df_train.iloc[batch_idx]
            batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)

            with tf.GradientTape() as tape:
                true_and_preds = enc_dec(batch_data, training=True)
                pep_loss = tf.reduce_mean(loss_fn(true_and_preds["pep_ytrue_ypred"], batch_data["pep_mask"]))
                mhc_loss = tf.reduce_mean(loss_fn(true_and_preds["mhc_ytrue_ypred"], batch_data["mhc_mask"]))
                loss = pep_loss + mhc_loss

            grads = tape.gradient(loss, enc_dec.trainable_variables)
            opt.apply_gradients(zip(grads, enc_dec.trainable_variables))

            epoch_loss_sum += loss.numpy()
            epoch_pep_loss_sum += pep_loss.numpy()
            epoch_mhc_loss_sum += mhc_loss.numpy()
            num_steps += 1
            
            # Periodically calculate and print validation loss on the fixed batch
            if step % (batch_size * 10) == 0:
                val_preds = enc_dec(fixed_val_batch, training=False)
                val_pep_loss = tf.reduce_mean(loss_fn(val_preds["pep_ytrue_ypred"], fixed_val_batch["pep_mask"]))
                val_mhc_loss = tf.reduce_mean(loss_fn(val_preds["mhc_ytrue_ypred"], fixed_val_batch["mhc_mask"]))
                val_loss = val_pep_loss + val_mhc_loss
                print(f"  step {step // batch_size:4d}  train_loss={loss.numpy():.4f}  val_loss={val_loss.numpy():.4f}")

        history['train_loss'].append(epoch_loss_sum / num_steps)
        history['train_pep_loss'].append(epoch_pep_loss_sum / num_steps)
        history['train_mhc_loss'].append(epoch_mhc_loss_sum / num_steps)
        print(f"Epoch {epoch} average train loss: {history['train_loss'][-1]:.4f}, pep={history['train_pep_loss'][-1]:.4f}, mhc={history['train_mhc_loss'][-1]:.4f}")

        # --- Full Validation Step (at end of epoch) ---
        val_loss_sum, val_pep_loss_sum, val_mhc_loss_sum, num_val_steps = 0, 0, 0, 0
        for val_step in range(0, len(val_indices), batch_size):
            batch_idx = val_indices[val_step:val_step + batch_size]
            batch_df = df_val.iloc[batch_idx]
            batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)
            
            true_and_preds = enc_dec(batch_data, training=False)
            pep_loss = tf.reduce_mean(loss_fn(true_and_preds["pep_ytrue_ypred"], batch_data["pep_mask"]))
            mhc_loss = tf.reduce_mean(loss_fn(true_and_preds["mhc_ytrue_ypred"], batch_data["mhc_mask"]))
            loss = pep_loss + mhc_loss
            
            val_loss_sum += loss.numpy()
            val_pep_loss_sum += pep_loss.numpy()
            val_mhc_loss_sum += mhc_loss.numpy()
            num_val_steps += 1
            
        avg_val_loss = val_loss_sum / num_val_steps
        history['val_loss'].append(avg_val_loss)
        history['val_pep_loss'].append(val_pep_loss_sum / num_val_steps)
        history['val_mhc_loss'].append(val_mhc_loss_sum / num_val_steps)
        print(f"Epoch {epoch} full validation loss: {avg_val_loss:.4f} (pep={history['val_pep_loss'][-1]:.4f}, mhc={history['val_mhc_loss'][-1]:.4f})")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            enc_dec.save_weights(best_weights_path)
            print(f"✓ New best model saved to {best_weights_path} with validation loss {best_val_loss:.4f}")

    # --- Post-Training Analysis ---
    print("\n--- Training Complete ---")
    # Plot and save loss curves
    plt.figure(figsize=(12, 5))
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epoch_range, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Total Training and Validation Loss (End of Epoch)')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Loss curves saved.")

    # Save final model weights
    final_weights_path = os.path.join(out_dir, "final_enc_dec.weights.h5")
    enc_dec.save_weights(final_weights_path)
    print(f"✓ Final weights saved to {final_weights_path}")

    # Load the best weights for final analysis
    print(f"\nLoading best weights from {best_weights_path} for final analysis.")
    enc_dec.load_weights(best_weights_path)

    # Helper function for inference and visualization
    def run_inference_and_visualizations(df, dataset_name):
        print(f"\n--- Running Inference & Visualization on {dataset_name.upper()} SET ---")
        dataset_out_dir = os.path.join(out_dir, f"visuals_{dataset_name}")
        os.makedirs(dataset_out_dir, exist_ok=True)

        indices = np.arange(len(df))
        # These will store the outputs from the model
        latents_seq = np.zeros((len(df), max_mhc_len, embed_dim), np.float32)
        latents_pooled = np.zeros((len(df), embed_dim), np.float32)
        
        for step in range(0, len(indices), batch_size):
            batch_idx = indices[step:step + batch_size]
            batch_df = df.iloc[batch_idx]
            batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)
            true_preds = enc_dec(batch_data, training=False)
            latents_seq[batch_idx] = true_preds["cross_latent"].numpy()
            latents_pooled[batch_idx] = true_preds["latent_vector"].numpy()

        # Save the sequential latents (for heatmaps, etc.)
        latents_seq_path = os.path.join(dataset_out_dir, f"mhc_latents_sequential_{dataset_name}.npy")
        np.save(latents_seq_path, latents_seq)
        print(f"✓ Sequential latents for {dataset_name} set saved to {latents_seq_path}")

        # Pass both latent types to the visualization function
        run_visualizations(df, latents_seq, latents_pooled, enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, dataset_out_dir)

    # Run analysis on all three datasets
    run_inference_and_visualizations(df_train, "train")
    run_inference_and_visualizations(df_val, "validation")
    run_inference_and_visualizations(df_test, "test")

    # Save model configuration
    config = {'max_pep_len': max_pep_len, 'max_mhc_len': max_mhc_len, 'embed_dim': embed_dim}
    with open(os.path.join(out_dir, "model_config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    print(f"\n✓ Model configuration saved.")


def infer(parquet_path: str, embed_npz: str, seq_csv: str, embd_key_path: str,
          out_dir: str, batch_size: int = 32):
    """
    Run inference on a dataset using a pre-trained model.
    Loads model configuration and best weights from the output directory.
    """
    global EMB_DB
    EMB_DB = load_embedding_db(embed_npz)

    # Load model configuration from the training output directory
    config_path = os.path.join(out_dir, "model_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found at {config_path}. Please train the model first.")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract all necessary parameters from the config file
    try:
        max_pep_len = config['max_pep_len']
        max_mhc_len = config['max_mhc_len']
        embed_dim = config['embed_dim']
        heads = config['heads']
        noise_std = config.get('noise_std', 0.0) # Default to 0.0 if not in older configs
    except KeyError as e:
        raise ValueError(f"Missing required parameter in model_config.json: {e}")

    # Load data and maps
    seq_map = pd.read_csv(seq_csv, index_col="allele")["mhc_sequence"].to_dict()
    embed_map = pd.read_csv(embd_key_path, index_col="key")["mhc_sequence"].to_dict()
    seq_map = {clean_key(k): v for k, v in seq_map.items()}
    
    df_infer = pq.ParquetFile(parquet_path).read().to_pandas()
    print(f"Loaded {len(df_infer)} samples for inference from {parquet_path}")

    # Initialize model with parameters from the config file
    enc_dec = pmclust_subtract(
        max_pep_len,
        max_mhc_len,
        emb_dim=embed_dim,
        heads=heads,
        mask_token=MASK_TOKEN,
        pad_token=PAD_TOKEN,
        noise_std=noise_std)

    # Build model by running a dummy batch to initialize weights
    dummy_batch_df = df_infer.head(1)
    dummy_data = rows_to_tensors(dummy_batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)
    _ = enc_dec(dummy_data, training=False)

    # Load best weights saved during training
    weights_path = os.path.join(out_dir, "best_enc_dec.weights.h5")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Best model weights not found at {weights_path}. Please train the model first.")
    enc_dec.load_weights(weights_path)
    print(f"✓ Model weights loaded from {weights_path}")

    # --- INFERENCE & LATENT EXTRACTION ---
    print("\nRunning inference to extract latents...")
    # Create a dedicated directory for inference results
    infer_out_dir = os.path.join(out_dir, "inference_visuals")
    os.makedirs(infer_out_dir, exist_ok=True)

    indices = np.arange(len(df_infer))
    latents_seq = np.zeros((len(df_infer), max_mhc_len, embed_dim), np.float32)
    latents_pooled = np.zeros((len(df_infer), embed_dim), np.float32)
    
    for step in range(0, len(indices), batch_size):
        batch_idx = indices[step:step + batch_size]
        batch_df = df_infer.iloc[batch_idx]
        batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)
        # Run model in inference mode
        true_preds = enc_dec(batch_data, training=False)
        latents_seq[batch_idx] = true_preds["cross_latent"].numpy()
        latents_pooled[batch_idx] = true_preds["latent_vector"].numpy()

    latents_seq_path = os.path.join(infer_out_dir, "mhc_latents_sequential_infer.npy")
    np.save(latents_seq_path, latents_seq)
    print(f"✓ Sequential latents from inference saved to {latents_seq_path}")

    # --- CALL THE VISUALIZATION FUNCTION ---
    run_visualizations(df_infer, latents_seq, latents_pooled, enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, infer_out_dir)
    print("\n✓ Inference and visualization complete.")



if __name__ == "__main__":
    # Suppress verbose TensorFlow logging, but keep errors.
    noise_std = 0.1
    heads = 4
    embed_dim = 96
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    train_path = "../data/binding_affinity_data/positives_class1_train.parquet"
    validate_path = "../data/binding_affinity_data/positives_class1_val.parquet"
    test_path = "../data/binding_affinity_data/positives_class1_test.parquet"
    embed_npz_path = "../data/ESM/esmc_600m/PMGen_whole_seq/mhc1_encodings.npz"
    embd_key_path = "../data/ESM/esmc_600m/PMGen_whole_seq/mhc1_encodings.csv"
    seq_csv_path = "../data/alleles/aligned_PMGen_class_1.csv"
    base_out_dir = f"../results/run_PMClust_ns_{noise_std}_hds_{heads}_zdim_{embed_dim}/"
    counter = 1
    while True:
        out_dir = f"{base_out_dir}{counter}"
        if not os.path.exists(out_dir):
            break
        counter += 1
    os.makedirs(out_dir, exist_ok=True)

    train(
        train_path=train_path,
        validation_path=validate_path,
        test_path=test_path,
        embed_npz=embed_npz_path,
        seq_csv=seq_csv_path,
        embd_key_path=embd_key_path,
        out_dir=out_dir,
        epochs=2,
        batch_size=128,
        lr=1e-3,
        embed_dim=96,
        heads=4,
        noise_std=noise_std
    )

    # infer(
    #     parquet_path=validate_path,
    #     embed_npz=embed_npz_path,
    #     seq_csv=seq_csv_path,
    #     embd_key_path=embd_key_path,
    #     out_dir=out_dir,
    #     batch_size=256,  # Reduced for faster dummy run
    # )
