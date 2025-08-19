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
from scipy.spatial.distance import cdist

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
        if MHC_CLASS == 2:
            key_parts = r["mhc_embedding_key"].split("_")
            key_norm1 = get_embed_key(clean_key(key_parts[0]), seq_map)
            key_norm2 = get_embed_key(clean_key(key_parts[1]), seq_map)
            mhc_seq = seq_map[key_norm1] + seq_map[key_norm2]
            batch_data["mhc_onehot"][i] = seq_to_onehot(mhc_seq, max_seq_len=max_mhc_len)
        else:
            key_norm = get_embed_key(clean_key(r["mhc_embedding_key"]), seq_map)
            mhc_seq = seq_map[key_norm]
            batch_data["mhc_onehot"][i] = seq_to_onehot(mhc_seq, max_seq_len=max_mhc_len)

    return {k: tf.convert_to_tensor(v) for k, v in batch_data.items()}


def run_visualizations(df, latents_pooled, enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, out_dir, dataset_name: str):
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
    #n_samples = latents_seq.shape[0]
    #latents_seq_flat = latents_seq.reshape(n_samples, -1)
    latents_seq_flat = latents_pooled

    # Create a publication-friendly color palette for all alleles
    np.random.seed(10)  # for reproducibility
    
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
    
    # Choose top 5 most frequent alleles to highlight
    top_5_alleles_to_mark = allele_counts.head(5).index.tolist()
    print(f"Highlighting top 5 alleles: {top_5_alleles_to_mark}")

    # # --- Visualization 1: UMAP of FLATTENED SEQUENTIAL latents ---
    # print("Running UMAP on FLATTENED SEQUENTIAL latents...")
    # reducer_seq = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    # embedding_seq = reducer_seq.fit_transform(latents_seq_flat)
    #
    # plt.figure(figsize=(14, 10))
    #
    # # Plot all points with assigned colors
    # for allele in unique_alleles:
    #     mask = alleles == allele
    #     plt.scatter(embedding_seq[mask, 0], embedding_seq[mask, 1],
    #                c=[allele_color_map[allele]], label=allele,
    #                s=5, alpha=0.3)
    #
    # # Highlight the 5 top alleles with distinct markers
    # for i, allele in enumerate(top_5_alleles_to_mark):
    #     mask = alleles == allele
    #     markers = ['*', 'D', 'X', 's', 'p']  # More publication-friendly markers
    #     plt.scatter(embedding_seq[mask, 0], embedding_seq[mask, 1],
    #                c='gray', marker=markers[i], s=3,
    #                edgecolor='black', linewidth=0.2,
    #                label=f'{allele} (highlighted)')
    #
    # plt.title(f'UMAP of Flattened Sequential Latents ({len(df)} Samples)\nColored by {len(unique_alleles)} Unique Alleles')
    # plt.xlabel('UMAP Dimension 1')
    # plt.ylabel('UMAP Dimension 2')
    #
    # # Create a better legend for scientific publication
    # if len(unique_alleles) > 20:
    #     # For many alleles, create a compact legend with multiple columns
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #
    #     # Separate highlighted alleles from regular alleles
    #     highlight_handles = [h for h, l in zip(handles, labels) if 'highlighted' in l]
    #     highlight_labels = [l for l in labels if 'highlighted' in l]
    #
    #     regular_handles = [h for h, l in zip(handles, labels) if 'highlighted' not in l]
    #     regular_labels = [l for l in labels if 'highlighted' not in l]
    #
    #     # First legend for highlighted alleles
    #     first_legend = plt.legend(highlight_handles, highlight_labels,
    #                              title='Highlighted Alleles',
    #                              bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.gca().add_artist(first_legend)
    #
    #     # Second legend for regular alleles in multiple columns
    #     plt.legend(regular_handles, regular_labels,
    #               title='All Alleles',
    #               bbox_to_anchor=(1.05, 0.5), loc='center left',
    #               ncol=max(1, len(regular_labels) // 30),  # Adjust columns based on count
    #               fontsize=8)
    # else:
    #     # For fewer alleles, a single legend is sufficient
    #     plt.legend(title='Allele', bbox_to_anchor=(1.05, 1),
    #               loc='upper left', fontsize=8, frameon=True)
    #
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, "umap_sequential_latents.png"), dpi=300, bbox_inches='tight')
    # plt.close()
    # print("✓ UMAP plot of sequential latents saved.")

    # --- Visualization 2: UMAP of POOLED latents ---
    print("Running UMAP on POOLED latents...")
    reducer_pooled = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=42)
    embedding_pooled = reducer_pooled.fit_transform(latents_pooled)

    plt.figure(figsize=(14, 10))
    
    # Plot all points with assigned colors
    for allele in unique_alleles:
        mask = alleles == allele
        plt.scatter(embedding_pooled[mask, 0], embedding_pooled[mask, 1], 
                   c=[allele_color_map[allele]], label=allele, 
                   s=3, alpha=0.3)
    
    # Highlight the 5 top alleles with distinct markers
    for i, allele in enumerate(top_5_alleles_to_mark):
        mask = alleles == allele
        markers = ['*', 'D', 'X', 's', 'p']  # More publication-friendly markers
        plt.scatter(embedding_pooled[mask, 0], embedding_pooled[mask, 1], 
                   c='gray', marker=markers[i], s=5,
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

    # --- Visualization 3: DBSCAN clustering on POOLED latents ---
    print("Running DBSCAN on UMAP-reduced latents for visual consistency...")

    # --- Step 1: Automate eps estimation using k-distance graph on UMAP data ---
    min_samples = 15  # A fixed, reasonable value for min_samples
    print(f"Using min_samples: {min_samples}")

    # Calculate distances on the 2D UMAP embedding
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(embedding_pooled)
    distances, indices = neighbors_fit.kneighbors(embedding_pooled)

    k_distances = np.sort(distances[:, min_samples - 1], axis=0)

    # Use KneeLocator to find the "elbow" of the k-distance plot
    kneedle = KneeLocator(range(len(k_distances)), k_distances, curve="convex", direction="increasing")
    estimated_eps = kneedle.elbow_y

    if estimated_eps is None:
        print("Warning: KneeLocator failed to find an elbow. Falling back to 95th percentile of distances.")
        estimated_eps = np.percentile(k_distances, 95)

    print(f"Estimated DBSCAN eps: {estimated_eps:.4f}")

    # Plot the k-distance graph for manual inspection
    plt.figure(figsize=(10, 6))
    kneedle.plot_knee()
    plt.xlabel("Points (sorted by distance)")
    plt.ylabel(f"Distance to {min_samples}-th nearest neighbor")
    plt.title("K-Distance Graph for DBSCAN Epsilon Estimation (on UMAP data)")
    plt.axhline(y=estimated_eps, color='r', linestyle='--', label=f'Selected eps = {estimated_eps:.4f}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(out_dir, "dbscan_k_distance_graph.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Step 2: Run final DBSCAN with the estimated eps ---
    dbscan = DBSCAN(eps=estimated_eps, min_samples=min_samples, n_jobs=-1)
    clusters = dbscan.fit_predict(embedding_pooled)

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = np.sum(clusters == -1)
    print(f"✓ DBSCAN found {n_clusters} clusters and {n_noise} noise points.")

    # --- Step 3: Add cluster labels to DataFrame and save ---
    df['cluster_id'] = clusters
    output_parquet_path = os.path.join(out_dir, f"{dataset_name}_with_clusters.parquet")
    df.to_parquet(output_parquet_path)
    print(f"✓ Saved dataset with cluster IDs to {output_parquet_path}")

    # --- Step 4: Visualize the clustering results ---
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
        plt.scatter(embedding_pooled[mask, 0], embedding_pooled[mask, 1],
                   c=[cluster_color_map[cluster]], label=label,
                   s=3, alpha=0.3)

    # Highlight the 5 top alleles with distinct markers
    for i, allele in enumerate(top_5_alleles_to_mark):
        mask = alleles == allele
        markers = ['*', 'D', 'X', 's', 'p'] 
        plt.scatter(embedding_pooled[mask, 0], embedding_pooled[mask, 1],
                   c='gray', marker=markers[i], s=5,
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


    # # --- Visualization 3: Show latent of one sample ---
    # plt.figure(figsize=(12, 6))
    # sns.heatmap(latents_seq[0], cmap='viridis')
    # plt.title(f'Latent Representation of Sample 0 (Allele: {alleles[0]})')
    # plt.xlabel('Embedding Dimension'); plt.ylabel('Sequence Position')
    # plt.savefig(os.path.join(out_dir, "single_sample_latent.png"))
    # plt.close()
    # print("✓ Single latent plot saved.")
    #
    # # --- Visualization 4: Compare mean latents for top 5 alleles ---
    # top_5_alleles = allele_counts.nlargest(5).index.tolist()
    # mean_latents = []
    # for allele_name in top_5_alleles:
    #     indices = df[df['mhc_embedding_key'].apply(clean_key) == allele_name].index
    #     mean_latent = latents_seq[indices].mean(axis=0)
    #     mean_latents.append(mean_latent)
    #
    # fig, axes = plt.subplots(len(top_5_alleles), 1, figsize=(10, 2 * len(top_5_alleles)), sharex=True)
    # fig.suptitle('Mean Latent Representation per Allele', fontsize=16)
    # for i, allele_name in enumerate(top_5_alleles):
    #     sns.heatmap(mean_latents[i], ax=axes[i], cmap='viridis')
    #     axes[i].set_title(f"{allele_name} (n={allele_counts[allele_name]})")
    #     axes[i].set_ylabel('Sequence Pos')
    # axes[-1].set_xlabel('Embedding Dim')
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig(os.path.join(out_dir, "mean_latents_comparison.png"), dpi=150, bbox_inches='tight')
    # plt.close()
    # print("✓ Mean latents comparison plot saved.")

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

# Helper function for inference and visualization
def run_inference_and_visualizations(df, dataset_name, enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, out_dir, batch_size=32, embed_dim=32):
    """
    Runs inference on a dataset, saves latents to memory-mapped files, and generates visualizations.
    """
    print(f"\n--- Running Inference & Visualization on {dataset_name.upper()} SET ---")
    dataset_out_dir = os.path.join(out_dir, f"visuals_{dataset_name}")
    os.makedirs(dataset_out_dir, exist_ok=True)

    indices = np.arange(len(df))

    # Define paths for memory-mapped arrays
    #latents_seq_path = os.path.join(dataset_out_dir, f"mhc_latents_sequential_{dataset_name}.mmap")
    latents_pooled_path = os.path.join(dataset_out_dir, f"mhc_latents_pooled_{dataset_name}.mmap")

    # Create memory-mapped arrays on disk
    #latents_seq = np.memmap(latents_seq_path, dtype='float32', mode='w+', shape=(len(df), max_mhc_len, embed_dim))
    latents_pooled = np.memmap(latents_pooled_path, dtype='float32', mode='w+', shape=(len(df), embed_dim))

    print(f"Processing {len(df)} samples in batches...")
    for step in range(0, len(indices), batch_size):
        batch_idx = indices[step:step + batch_size]
        batch_df = df.iloc[batch_idx]
        batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)

        # Run model inference
        true_preds = enc_dec(batch_data, training=False)

        # Write batch results directly to memory-mapped arrays
        #latents_seq[batch_idx] = true_preds["cross_latent"].numpy()
        latents_pooled[batch_idx] = true_preds["latent_vector"].numpy()

        # Flush changes to disk periodically to ensure data is saved
        if step % (batch_size * 10) == 0:
            #latents_seq.flush()
            latents_pooled.flush()

    # Final flush to save any remaining data
    #latents_seq.flush()
    latents_pooled.flush()

    #print(f"✓ Sequential latents for {dataset_name} set saved to {latents_seq_path}")
    print(f"✓ Pooled latents for {dataset_name} set saved to {latents_pooled_path}")

    # Now, call the visualization function with the paths to the memory-mapped files
    run_visualizations(df, latents_pooled, enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, dataset_out_dir, dataset_name)


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
    if MHC_CLASS == 2:
        max_mhc_len = 400 # manually set
    else:
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

    # Save model configuration
    config = {
        'max_pep_len': max_pep_len, 
        'max_mhc_len': max_mhc_len, 
        'embed_dim': embed_dim,
        'heads': heads,
        'noise_std': noise_std
    }
    with open(os.path.join(out_dir, "model_config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    print(f"\n✓ Model configuration saved.")

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

    # Run analysis on all three datasets using memory-mapped arrays
    run_inference_and_visualizations(df_train, "train", enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, out_dir, batch_size, embed_dim)
    run_inference_and_visualizations(df_val, "validation", enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, out_dir, batch_size, embed_dim)
    run_inference_and_visualizations(df_test, "test", enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, out_dir, batch_size, embed_dim)


def run_inference_and_visualizations(df, dataset_name, enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, out_dir, batch_size=32, embed_dim=32):
    """
    Runs inference on a dataset, saves latents to memory-mapped files, and generates visualizations.
    """
    print(f"\n--- Running Inference & Visualization on {dataset_name.upper()} SET ---")
    dataset_out_dir = os.path.join(out_dir, f"visuals_{dataset_name}")
    os.makedirs(dataset_out_dir, exist_ok=True)

    indices = np.arange(len(df))

    # Define paths for memory-mapped arrays
    #latents_seq_path = os.path.join(dataset_out_dir, f"mhc_latents_sequential_{dataset_name}.mmap")
    latents_pooled_path = os.path.join(dataset_out_dir, f"mhc_latents_pooled_{dataset_name}.mmap")

    # Create memory-mapped arrays on disk
    #latents_seq = np.memmap(latents_seq_path, dtype='float32', mode='w+', shape=(len(df), max_mhc_len, embed_dim))
    latents_pooled = np.memmap(latents_pooled_path, dtype='float32', mode='w+', shape=(len(df), embed_dim))

    print(f"Processing {len(df)} samples in batches...")
    for step in range(0, len(indices), batch_size):
        batch_idx = indices[step:step + batch_size]
        batch_df = df.iloc[batch_idx]
        batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)

        # Run model inference
        true_preds = enc_dec(batch_data, training=False)

        # Write batch results directly to memory-mapped arrays
        #latents_seq[batch_idx] = true_preds["cross_latent"].numpy()
        latents_pooled[batch_idx] = true_preds["latent_vector"].numpy()

        # Flush changes to disk periodically to ensure data is saved
        if step % (batch_size * 10) == 0:
            #latents_seq.flush()
            latents_pooled.flush()

    # Final flush to save any remaining data
    #latents_seq.flush()
    latents_pooled.flush()

    #print(f"✓ Sequential latents for {dataset_name} set saved to {latents_seq_path}")
    print(f"✓ Pooled latents for {dataset_name} set saved to {latents_pooled_path}")

    # Now, call the visualization function with the paths to the memory-mapped files
    run_visualizations(df, latents_pooled, enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, dataset_out_dir, dataset_name)


def infer(parquet_path: str, embed_npz: str, seq_csv: str, embd_key_path: str,
          out_dir: str, batch_size: int = 256, df_name="inference"):
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
        print("Extracting model configuration...")
        max_pep_len = config['max_pep_len']
        max_mhc_len = config['max_mhc_len']
        embed_dim = config['embed_dim']
        heads = config['heads']
        noise_std = config.get('noise_std', 0.0)  # Default to 0.0 if not in older configs
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
        raise FileNotFoundError(f"Best model weights not found at {weights_path} Please train the model first.")
    enc_dec.load_weights(weights_path)
    print(f"✓ Model weights loaded from {weights_path}")

    # --- INFERENCE & LATENT EXTRACTION ---
    print("\nRunning inference to extract latents...")
    # Create a dedicated directory for inference results
    infer_out_dir = os.path.join(out_dir, df_name)
    os.makedirs(infer_out_dir, exist_ok=True)

    indices = np.arange(len(df_infer))

    # Define paths for memory-mapped arrays
    #latents_seq_path = os.path.join(infer_out_dir, "mhc_latents_sequential_infer.mmap")
    latents_pooled_path = os.path.join(infer_out_dir, "mhc_latents_pooled_infer.mmap")

    # Create memory-mapped arrays
    #latents_seq = np.memmap(latents_seq_path, dtype='float32', mode='w+', shape=(len(df_infer), max_mhc_len, embed_dim))
    latents_pooled = np.memmap(latents_pooled_path, dtype='float32', mode='w+', shape=(len(df_infer), embed_dim))

    print(f"Processing {len(df_infer)} samples for inference in batches...")
    for step in tqdm(range(0, len(indices), batch_size), desc="Inference Progress"):
        batch_idx = indices[step:step + batch_size]
        batch_df = df_infer.iloc[batch_idx]
        batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)

        # TODO TEMP - DEBUG - Remove LATER
        # Save all batches into a single HDF5 file by appending slices
        import h5py  # local import to avoid global dependency issues
        batch_data_path = os.path.join(infer_out_dir, "batch_data.h5")
        start, end = step, step + len(batch_idx)
        with h5py.File(batch_data_path, "a") as h5f:
            if "pep_onehot" not in h5f:
                N = len(df_infer)
                h5f.create_dataset("pep_onehot", (N, max_pep_len, 21), dtype="float32")
                h5f.create_dataset("pep_mask", (N, max_pep_len), dtype="float32")
                h5f.create_dataset("mhc_emb", (N, max_mhc_len, 1152), dtype="float32")
                h5f.create_dataset("mhc_mask", (N, max_mhc_len), dtype="float32")
                h5f.create_dataset("mhc_onehot", (N, max_mhc_len, 21), dtype="float32")
            h5f["pep_onehot"][start:end] = batch_data["pep_onehot"].numpy()
            h5f["pep_mask"][start:end] = batch_data["pep_mask"].numpy()
            h5f["mhc_emb"][start:end] = batch_data["mhc_emb"].numpy()
            h5f["mhc_mask"][start:end] = batch_data["mhc_mask"].numpy()
            h5f["mhc_onehot"][start:end] = batch_data["mhc_onehot"].numpy()

        # Run model in inference mode
        true_preds = enc_dec(batch_data, training=False)



        # Write results to memory-mapped arrays
        #latents_seq[batch_idx] = true_preds["cross_latent"].numpy()
        latents_pooled[batch_idx] = true_preds["latent_vector"].numpy()

    # Flush to ensure all data is written to disk
    #latents_seq.flush()
    latents_pooled.flush()

    #print(f"✓ Sequential latents from inference saved to {latents_seq_path}")
    print(f"✓ Pooled latents from inference saved to {latents_pooled_path}")

    # --- CALL THE VISUALIZATION FUNCTION ---
    # Pass the paths to the memory-mapped files
    run_visualizations(df_infer, latents_pooled, enc_dec, max_pep_len, max_mhc_len, seq_map,
                       embed_map, infer_out_dir, df_name)
    print("\n✓ Inference and visualization complete.")


def generate_negatives(model_out_dir: str, positives_train: str, positives_validation: str, embed_npz: str, seq_csv: str, embd_key_path: str, batch_size: int = 256):
    """
    Generates a dataset of negative samples using a trained model.

    This function works by:
    1. Running inference on a complete dataset of peptide-MHC pairs to get latent representations.
    2. Clustering these samples in the latent space using DBSCAN (performed inside `infer`).
    3. Calculating the centroid (mean latent vector) for each cluster.
    4. For each cluster, identifying the most distant clusters based on centroid distance.
    5. Creating new "negative" pairs by swapping the MHC alleles of samples with alleles from these distant clusters.
    6. Saving the newly generated negative dataset.

    Args:
        model_out_dir (str): Path to the output directory of a trained model, containing weights and config.
        whole_dataset_pq (str): Path to the Parquet file containing the entire dataset to be clustered.
        embed_npz (str): Path to the NPZ file with MHC embeddings.
        seq_csv (str): Path to the CSV file with MHC sequences.
        embd_key_path (str): Path to the CSV file mapping embedding keys.
        batch_size (int): Batch size for the inference process.
    """
    print("\n--- Starting Negative Sample Generation ---")

    # Load the positive samples
    positives_train_df = pd.read_parquet(positives_train)
    positives_validation_df = pd.read_parquet(positives_validation)

    positives_comb = pd.concat([positives_train_df, positives_validation_df], axis=0, ignore_index=True)

    # save the combined positives dataset
    positives_comb_path = os.path.join(model_out_dir, "positives_combined.parquet")
    positives_comb.to_parquet(positives_comb_path, index=False)
    print(f"✓ Combined positives dataset saved to {positives_comb_path}")

    # 1. Run inference and clustering on the entire dataset
    # The `infer` function will create a subdirectory and save a parquet file with cluster IDs.
    df_name = "positives_combined"
    infer_dir = os.path.join(model_out_dir, df_name)
    clustered_df_path = os.path.join(infer_dir, f"{df_name}_with_clusters.parquet")

    if not os.path.exists(clustered_df_path):
        print(f"Clustered data not found at {clustered_df_path}. Running inference first...")
        infer(
            parquet_path=positives_comb_path,
            embed_npz=embed_npz,
            seq_csv=seq_csv,
            embd_key_path=embd_key_path,
            out_dir=model_out_dir,
            batch_size=batch_size,
            df_name=df_name
        )
    else:
        print(f"Found existing clustered data at {clustered_df_path}.")

    # 2. Load the clustered data and the corresponding latent vectors
    latents_path = os.path.join(infer_dir, "mhc_latents_pooled_infer.mmap")

    if not os.path.exists(clustered_df_path) or not os.path.exists(latents_path):
        raise FileNotFoundError(f"Required files for negative generation not found in {infer_dir}")

    df_clustered = pd.read_parquet(clustered_df_path)

    # Load the pooled latents from the memory-mapped file
    # We need to know the shape, which we can get from the model config
    config_path = os.path.join(model_out_dir, "model_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found at {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    embed_dim = config['embed_dim']

    # Correct the shape to be 2D: (n_samples, embed_dim)
    latents_pooled = np.memmap(latents_path, dtype='float32', mode='r', shape=(len(df_clustered), embed_dim))
    # save mmap configs
    mmap_config_path = os.path.join(infer_dir, "mmap_config.json")
    mmap_config = {
        'latents_shape': latents_pooled.shape,
        'embed_dim': embed_dim,
        'max_pep_len': config['max_pep_len'],
        'max_mhc_len': config['max_mhc_len']
    }
    with open(mmap_config_path, 'w') as f:
        json.dump(mmap_config, f, indent=4)
    print(f"✓ Saved memory-mapped latents configuration to {mmap_config_path}")

    print(f"✓ Loaded {len(df_clustered)} samples with cluster IDs and memory-mapped latents.")

    # Exclude noise points from negative generation logic
    non_noise_mask = df_clustered['cluster_id'] != -1
    df_non_noise = df_clustered[non_noise_mask].copy()
    latents_non_noise = latents_pooled[non_noise_mask]
    print(f"  Proceeding with {len(df_non_noise)} non-noise samples.")

    if len(df_non_noise) == 0:
        print("Warning: No clusters found (all points are noise). Cannot generate negatives.")
        return

    # 3. Calculate the centroid for each cluster
    print("Calculating cluster centroids...")
    cluster_ids = sorted(df_non_noise['cluster_id'].unique())
    cluster_centroids = {}

    for cid in cluster_ids:
        cluster_mask = df_non_noise['cluster_id'] == cid
        cluster_centroids[cid] = latents_non_noise[cluster_mask].mean(axis=0)

    print(f"✓ Calculated centroids for {len(cluster_centroids)} clusters.")

    if len(cluster_centroids) < 2:
        print("Warning: Fewer than 2 clusters found. Cannot determine distant clusters for negative sampling.")
        return

    # 4. For each cluster, find the most distant cluster
    sorted_cids = sorted(cluster_centroids.keys())
    centroid_matrix = np.array([cluster_centroids[cid] for cid in sorted_cids])
    dist_matrix = cdist(centroid_matrix, centroid_matrix, metric='euclidean')

    # Visualize and save the distance matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(dist_matrix, xticklabels=sorted_cids, yticklabels=sorted_cids, cmap="viridis", annot=True, fmt=".2f")
    plt.title("Centroid Distance Matrix Between Clusters")
    plt.xlabel("Cluster ID")
    plt.ylabel("Cluster ID")
    plt.tight_layout()
    dist_matrix_path = os.path.join(model_out_dir, "cluster_distance_matrix.png")
    plt.savefig(dist_matrix_path, dpi=300)
    plt.close()
    print(f"✓ Saved cluster distance matrix plot to {dist_matrix_path}")

    # Set diagonal to a large value to ignore self-distance
    np.fill_diagonal(dist_matrix, -np.inf)  # Use -inf for argmax

    # Find the index of the most distant cluster for each cluster
    farthest_cluster_indices = np.argmax(dist_matrix, axis=1)

    farthest_cluster_map = {
        own_cid: sorted_cids[farthest_idx]
        for own_cid, farthest_idx in zip(sorted_cids, farthest_cluster_indices)
    }
    print("✓ Determined the most distant cluster for each cluster.")

    # 5. Generate negative samples
    new_negatives = []
    # Get a representative set of alleles from each cluster
    alleles_per_cluster = df_non_noise.groupby('cluster_id')['mhc_embedding_key'].apply(
        lambda x: list(set(x))
    )

    for _, row in df_non_noise.iterrows():
        current_cluster = row['cluster_id']
        target_cluster = farthest_cluster_map[current_cluster]

        # Get potential alleles from the target cluster
        target_alleles = alleles_per_cluster.get(target_cluster)
        if not target_alleles:
            continue

        # Choose a random allele from the target cluster
        new_allele = np.random.choice(target_alleles)

        # Create the new negative sample row
        new_row = row.copy()
        new_row['mhc_embedding_key'] = new_allele.replace("*", "").replace(":", "")
        new_row['allele'] = new_allele
        new_row['assigned_label'] = 0  # Explicitly set as non-binder
        new_negatives.append(new_row)

    if not new_negatives:
        print("Warning: Could not generate any negative samples.")
        return

    df_negatives = pd.DataFrame(new_negatives)
    print(f"✓ Generated {len(df_negatives)} new negative samples.")

    # 6. Save the new dataset
    negatives_output_path = os.path.join(model_out_dir, f"generated_negatives{MHC_CLASS}.parquet")
    df_negatives.to_parquet(negatives_output_path, index=False)

    # Save allele names to a text file, separated by commas
    alleles_output_path = os.path.join(model_out_dir, f"generated_negatives{MHC_CLASS}_alleles.txt")
    if 'mhc_embedding_key' in df_negatives.columns:
        alleles = df_negatives['mhc_embedding_key'].unique()
        with open(alleles_output_path, 'w') as f:
            f.write(",".join(map(str, alleles)))
        print(f"✓ Saved allele names to {alleles_output_path}")
    else:
        print("Warning: 'mhc_embedding_key' column not found in negatives, skipping allele file generation.")

    # Save peptide sequences to a FASTA file
    fasta_output_path = os.path.join(model_out_dir, f"generated_negatives{MHC_CLASS}_peptides.fasta")
    with open(fasta_output_path, 'w') as f:
        for i, peptide in enumerate(df_negatives['long_mer']):
            f.write(f">{i}\n{peptide}\n")
    print(f"✓ Saved peptide sequences to {fasta_output_path}")
    print(f"✓ Saved generated negatives to {negatives_output_path}")

    # save positives_dataset with negatives
    # Use original positives, not the clustered/filtered one
    final_dataset = pd.concat([positives_comb, df_negatives], ignore_index=True)
    # drop cluster_id col
    final_dataset = final_dataset.drop(columns=['cluster_id'], errors='ignore')
    final_dataset_path = os.path.join(model_out_dir,
                                      f"binding_affinity_dataset_with_swapped_negatives{MHC_CLASS}.parquet")
    final_dataset.to_parquet(final_dataset_path, index=False)
    print(f"✓ Saved combined dataset with positives and negatives to {final_dataset_path}")


if __name__ == "__main__":
    # Suppress verbose TensorFlow logging, but keep errors.
    MHC_CLASS = 1
    noise_std = 0.1
    heads = 4
    embed_dim = 96
    batch_size = 32
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    train_path = f"../data/binding_affinity_data/positives_class{MHC_CLASS}_train.parquet"
    validate_path = f"../data/binding_affinity_data/positives_class{MHC_CLASS}_val.parquet"
    test_path = f"../data/binding_affinity_data/positives_class{MHC_CLASS}_test.parquet"
    embed_npz_path = f"/media/amirreza/lasse/ESM/esmc_600m/PMGen_whole_seq/mhc{MHC_CLASS}_encodings.npz"
    embd_key_path = f"/media/amirreza/lasse/ESM/esmc_600m/PMGen_whole_seq/mhc{MHC_CLASS}_encodings.csv"
    seq_csv_path = f"../data/alleles/aligned_PMGen_class_{MHC_CLASS}.csv"
    base_out_dir = f"../results/run_PMClust_ns_{noise_std}_hds_{heads}_zdim_{embed_dim}_L{MHC_CLASS}/"
    whole_dataset = f"../../data/binding_affinity_data/concatenated_class{MHC_CLASS}_all.parquet"


    # Training
    # counter = 1
    # while True:
    #     out_dir = f"{base_out_dir}{counter}"
    #     if not os.path.exists(out_dir):
    #         break
    #     counter += 1
    # os.makedirs(out_dir, exist_ok=True)

    # train(
    #     train_path=train_path,
    #     validation_path=validate_path,
    #     test_path=test_path,
    #     embed_npz=embed_npz_path,
    #     seq_csv=seq_csv_path,
    #     embd_key_path=embd_key_path,
    #     out_dir=out_dir,
    #     epochs=4,
    #     batch_size=batch_size,
    #     lr=1e-5,
    #     embed_dim=96,
    #     heads=4,
    #     noise_std=noise_std
    # )

    # out_dir = f"{base_out_dir}4"
    # infer(
    #     parquet_path=test_path,
    #     embed_npz=embed_npz_path,
    #     seq_csv=seq_csv_path,
    #     embd_key_path=embd_key_path,
    #     out_dir=out_dir,
    #     batch_size=256,
    #     df_name="inference_validation"
    # )

    out_dir = f"../pretrained/mhc1_pmclust/"
    # generate negatives
    generate_negatives(
        model_out_dir=out_dir,
        positives_train=train_path,
        positives_validation=validate_path,
        embed_npz=embed_npz_path,
        seq_csv=seq_csv_path,
        embd_key_path=embd_key_path,
        batch_size=batch_size
    )
