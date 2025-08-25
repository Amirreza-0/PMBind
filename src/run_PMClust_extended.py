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
import matplotlib.colors as mcolors

# Added for automated DBSCAN eps estimation
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import cophenet


# Assuming these are in a 'utils' directory relative to the script
from utils import (seq_to_onehot, get_embed_key, NORM_TOKEN, MASK_TOKEN, PAD_TOKEN, PAD_VALUE, MASK_VALUE,
                   masked_categorical_crossentropy, OHE_to_seq_single, split_y_true_y_pred, OHE_to_seq, clean_key,
                   mmseqs2_reduced_alphabet, reduced_anchor_pair, peptide_properties_biopython, cn_terminal_amino_acids)

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
        
        # Randomly mask 30% of valid peptide positions with MASK_TOKEN
        valid_positions = np.where(batch_data["pep_mask"][i] == NORM_TOKEN)[0]
        if len(valid_positions) > 0:
            mask_fraction = 0.30
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
            mask_fraction = 0.20
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


# def _plot_umap(
#         embedding: np.ndarray,
#         labels: pd.Series,
#         color_map: dict,
#         title: str,
#         filename: str,
#         legend_name: str = "Alleles",
#         alleles_to_highlight: list = None,
#         highlight_labels_series: pd.Series = None,
#         figsize=(40, 15),
#         point_size=2,
#         legend_: bool = True
# ):
#     """
#     Helper function to create and save a UMAP plot in a scientific paper format.
#
#     Args:
#         embedding: The 2D UMAP embedding data.
#         labels: A pandas Series of labels for coloring the points (e.g., alleles or clusters).
#         color_map: A dictionary mapping labels to colors.
#         title: The main title for the plot.
#         filename: The full path to save the plot image.
#         alleles_to_highlight: A list of specific alleles to highlight with distinct markers.
#         highlight_labels_series: A pandas Series with the true labels used for highlighting.
#                                  If None, `labels` will be used.
#     """
#     plt.style.use('seaborn-v0_8-whitegrid')
#     fig, ax = plt.subplots(figsize=figsize)
#
#     unique_labels = sorted(labels.unique())
#
#     # Plot all data points first
#     for label in unique_labels:
#         mask = (labels == label)
#         # Use a default gray color if a label is not in the color map (e.g., noise)
#         color = color_map.get(label, [0.5, 0.5, 0.5, 0.5])
#         ax.scatter(
#             embedding[mask, 0], embedding[mask, 1],
#             c=[color], label=label, s=point_size, alpha=0.8, rasterized=True
#         )
#
#     # Highlight the specified random alleles with distinct markers
#     if alleles_to_highlight:
#         label_source = highlight_labels_series if highlight_labels_series is not None else labels
#         markers = ['*', 'D', 'X', 's', 'p']
#         for i, allele in enumerate(alleles_to_highlight):
#             mask = (label_source == allele)
#             if np.any(mask):
#                 ax.scatter(
#                     embedding[mask, 0], embedding[mask, 1],
#                     c='none', marker=markers[i % len(markers)], s=point_size * 2.5,
#                     edgecolor='black', linewidth=0.3,
#                     label=f'{allele} (highlighted)'
#                 )
#
#     ax.set_title(title, fontsize=16, fontweight='bold')
#     ax.set_xlabel('UMAP Dimension 1', fontsize=12)
#     ax.set_ylabel('UMAP Dimension 2', fontsize=12)
#
#     # --- Publication Quality Legend Handling ---
#     handles, legend_labels = ax.get_legend_handles_labels()
#
#     highlight_handles = [h for h, l in zip(handles, legend_labels) if 'highlighted' in l]
#     highlight_labels = [l for l in legend_labels if 'highlighted' in l]
#
#     regular_handles = [h for h, l in zip(handles, legend_labels) if 'highlighted' not in l]
#     regular_labels = [l for l in legend_labels if 'highlighted' not in l]
#
#     # Create a legend for the highlighted markers if they exist
#     if highlight_handles:
#         first_legend = ax.legend(
#             highlight_handles, highlight_labels,
#             title='Highlighted Alleles', bbox_to_anchor=(1.05, 1),
#             loc='upper left', frameon=True, fontsize=10, title_fontsize=12, markerscale=2.5
#         )
#         ax.add_artist(first_legend)
#
#     # Create the main legend for all alleles or clusters
#     legend_title = legend_name
#     if len(regular_labels) > 25:
#         if legend_:
#             colour_list = [color_map.get(lbl, [0.5, 0.5, 0.5, 0.5])
#                            for lbl in regular_labels]
#
#             cmap = mcolors.ListedColormap(colour_list)
#             bounds = np.arange(len(regular_labels) + 1) - 0.5
#             norm = mcolors.BoundaryNorm(bounds, cmap.N)
#
#             sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#             sm.set_array([])
#
#             cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
#
#             # -------- keep ticks below the 1 000 limit --------
#             MAX_TICKS = 150
#             if len(regular_labels) <= MAX_TICKS:
#                 tick_positions = np.arange(len(regular_labels))
#                 tick_labels = regular_labels
#             else:
#                 step = int(np.ceil(len(regular_labels) / MAX_TICKS))
#                 tick_positions = np.arange(0, len(regular_labels), step)
#                 tick_labels = [regular_labels[i] for i in tick_positions]
#
#             cbar.set_ticks(tick_positions)
#             cbar.ax.set_yticklabels(tick_labels, fontsize=6)
#
#             cbar.set_label(legend_title, fontsize=10)
#             # ax.legend(
#             #     regular_handles, regular_labels, title=legend_title,
#             #     bbox_to_anchor=(1.05, 0.75 if highlight_handles else 1), loc='upper left',
#             #     ncol=max(1, len(regular_labels) // 30), fontsize=8, frameon=True, title_fontsize=12
#             # )
#     else:
#         if legend_:
#             ax.legend(
#                 regular_handles, regular_labels, title=legend_title,
#                 bbox_to_anchor=(1.05, 0.75 if highlight_handles else 1), loc='upper left',
#                 fontsize=10, frameon=True, title_fontsize=12, markerscale=2.5
#             )
#
#     plt.tight_layout(rect=[0, 0, 0.85, 1])
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     plt.close(fig)
#     print(f"✓ Plot saved to {filename}")


def _plot_umap(
        embedding: np.ndarray,
        labels: pd.Series,
        color_map: dict,
        title: str,
        filename: str,
        legend_name: str = "Alleles",
        alleles_to_highlight: list | None = None,
        highlight_labels_series: pd.Series | None = None,
        figsize: tuple = (40, 15),
        point_size: int = 2,
        legend_: bool = True,
        legend_style: str = "detailed",
        bar_cmap_name: str = "tab20"
):
    """
    Draw a UMAP scatter plot.

    Parameters
    ----------
    embedding : np.ndarray
        2-D embeddings with shape (n_samples, 2).
    labels : pd.Series
        Categorical labels (strings).
    color_map : dict
        Mapping label → colour (RGBA tuples or hex).  Used by the "detailed"
        legend.  For legend_style == "bar" the order of colours in this dict
        is preserved so every label keeps the same hue.
    title : str
        Plot title.
    filename : str
        Output path (png, pdf…).
    legend_name : str
        Title of the legend / colour-bar.
    alleles_to_highlight : list[str] | None
        If given, plot these alleles with a special marker.
    highlight_labels_series : pd.Series | None
        Series to use for matching the highlight list.  If None, `labels`
        is used.
    figsize : tuple
        Figure size.
    point_size : int
        Scatter point size.
    legend_ : bool
        Completely switch legend / colour-bar on or off.
    legend_style : {"detailed", "bar"}
        "detailed" – original behaviour (individual legend entries
        + discrete colour-bar for >25 labels).
        "bar"      – one scatter call + compact colour-bar showing the full
                     label range (recommended for many categories).
    bar_cmap_name : str
        Matplotlib colormap to use when legend_style == "bar".
    """

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    # 0.  Prepare palettes and (optionally) integer codes
    unique_labels = sorted(labels.unique())
    n_labels = len(unique_labels)
    # Store default grey for missing labels
    default_grey = (0.5, 0.5, 0.5, 0.5)
    # 1.  Plot points
    if legend_style.lower() == "bar":
        # -- build integer codes
        lbl_to_code = {lbl: i for i, lbl in enumerate(unique_labels)}
        codes = labels.map(lbl_to_code).values

        # -- build discrete cmap
        palette = [color_map.get(lbl, default_grey) for lbl in unique_labels]
        cmap = mcolors.ListedColormap(palette, name="listed_palette")
        norm = mcolors.BoundaryNorm(np.arange(n_labels + 1) - .5, n_labels)

        # -- single scatter call
        sc = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=codes, cmap=cmap, norm=norm,
            s=point_size, alpha=0.8, rasterized=True
        )

    else:  # "detailed"  (original behaviour)
        sc = None  # in case someone needs it later
        for lbl in unique_labels:
            mask = (labels == lbl)
            colour = color_map.get(lbl, default_grey)
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=[colour], label=lbl, s=point_size, alpha=0.8, rasterized=True
            )

    # 2.  Highlight selected alleles
    if alleles_to_highlight:
        label_source = highlight_labels_series if highlight_labels_series is not None else labels
        markers = ['*', 'D', 'X', 's', 'p']
        for i, allele in enumerate(alleles_to_highlight):
            mask = (label_source == allele)
            if np.any(mask):
                ax.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    c='none', marker=markers[i % len(markers)],
                    s=point_size * 2.5, edgecolor='black', linewidth=0.3,
                    label=f'{allele} (highlighted)'
                )

    # 3.  Axes labels & title
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)

    # 4.  Legend / colour-bar handling
    if legend_:
        if legend_style.lower() == "bar":
            # --- compact colour-bar
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(legend_name, fontsize=10)

            # Show a manageable number of ticks
            MAX_TICKS = 25
            if n_labels <= MAX_TICKS:
                tick_positions = np.arange(n_labels)
                tick_labels = unique_labels
            else:
                tick_positions = [0, n_labels - 1]
                tick_labels = [unique_labels[0], unique_labels[-1]]

            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels, fontsize=6)

            # remove colour-bar borders that create a grey stripe
            if hasattr(cbar, "solids"):
                cbar.solids.set_edgecolor("face")
                cbar.solids.set_linewidth(0)
            cbar.outline.set_visible(False)

        else:  # --- detailed legend
            handles, legend_labels = ax.get_legend_handles_labels()

            highlight_handles = [h for h, l in zip(handles, legend_labels)
                                 if 'highlighted' in l]
            highlight_labels  = [l for l in legend_labels
                                 if 'highlighted' in l]

            regular_handles = [h for h, l in zip(handles, legend_labels)
                               if 'highlighted' not in l]
            regular_labels  = [l for   l in legend_labels
                               if 'highlighted' not in l]

            # separate legend for highlights
            if highlight_handles:
                first_legend = ax.legend(
                    highlight_handles, highlight_labels,
                    title='Highlighted Alleles', bbox_to_anchor=(1.05, 1),
                    loc='upper left', frameon=True, fontsize=10,
                    title_fontsize=12, markerscale=2.5
                )
                ax.add_artist(first_legend)

            # main legend or colour-bar for clusters
            if len(regular_labels) > 25:
                # ---- colour-bar with category ticks
                colour_list = [color_map.get(lbl, default_grey)
                               for lbl in regular_labels]

                cmap = mcolors.ListedColormap(colour_list)
                bounds = np.arange(len(regular_labels) + 1) - 0.5
                norm = mcolors.BoundaryNorm(bounds, cmap.N)

                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])

                cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

                MAX_TICKS = 150
                if len(regular_labels) <= MAX_TICKS:
                    tick_positions = np.arange(len(regular_labels))
                    tick_labels = regular_labels
                else:
                    step = int(np.ceil(len(regular_labels) / MAX_TICKS))
                    tick_positions = np.arange(0, len(regular_labels), step)
                    tick_labels = [regular_labels[i] for i in tick_positions]

                cbar.set_ticks(tick_positions)
                cbar.ax.set_yticklabels(tick_labels, fontsize=6)
                cbar.set_label(legend_name, fontsize=10)

                if hasattr(cbar, "solids"):
                    cbar.solids.set_edgecolor("face")
                    cbar.solids.set_linewidth(0)
                cbar.outline.set_visible(False)

            else:
                # ---- normal legend with handles
                ax.legend(
                    regular_handles, regular_labels, title=legend_name,
                    bbox_to_anchor=(1.05, 0.75 if highlight_handles else 1),
                    loc='upper left', fontsize=10, frameon=True,
                    title_fontsize=12, markerscale=2.5
                )

    # 5.  Save & close
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Plot saved to {filename}")

def _run_dbscan_and_plot(
            embedding: np.ndarray,
            alleles: pd.Series,
            random_alleles_to_highlight: list,
            latent_type: str,
            out_dir: str,
            figsize=(40, 15),
            point_size=2
    ):
        """Helper to run DBSCAN, estimate eps, and plot results."""
        print(f"\nRunning DBSCAN on {latent_type} UMAP embedding...")

        # 1. Estimate eps using the k-distance graph for robust parameter selection
        min_samples = 50
        neighbors = NearestNeighbors(n_neighbors=min_samples).fit(embedding)
        distances, _ = neighbors.kneighbors(embedding)
        k_distances = np.sort(distances[:, min_samples - 1], axis=0)

        kneedle = KneeLocator(range(len(k_distances)), k_distances, curve="convex", direction="increasing")
        estimated_eps = kneedle.elbow_y
        if estimated_eps is None:
            estimated_eps = np.percentile(k_distances, 75)
            print(f"Warning: KneeLocator failed. Falling back to eps={estimated_eps:.4f}")
        else:
            print(f"Estimated DBSCAN eps for {latent_type} latents: {estimated_eps:.4f}")

        # 2. Run DBSCAN with estimated parameters
        dbscan = DBSCAN(eps=estimated_eps, min_samples=min_samples, n_jobs=-1)
        clusters = dbscan.fit_predict(embedding)
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = np.sum(clusters == -1)
        print(f"✓ Found {n_clusters} clusters and {n_noise} noise points.")

        # 3. Visualize the clustering results
        cluster_labels = pd.Series([f'Cluster {c}' if c != -1 else 'Noise' for c in clusters])
        unique_cluster_labels = sorted(cluster_labels.unique(), key=lambda x: (x == 'Noise', x))

        # Use a more distinctive color palette for clusters
        n_labels = len(unique_cluster_labels)
        if n_labels <= 20:
            colors = sns.color_palette("tab20", n_colors=n_labels)
        else:
            colors = sns.color_palette("hls", n_colors=n_labels)
        cluster_color_map = {label: color for label, color in zip(unique_cluster_labels, colors)}
        if 'Noise' in cluster_color_map:
            cluster_color_map['Noise'] = [0.7, 0.7, 0.7, 0.5]  # Muted gray for noise points

        _plot_umap(
            embedding=embedding,
            labels=cluster_labels,
            color_map=cluster_color_map,
            title=f'DBSCAN Clustering of {latent_type.capitalize()} Latents\n({n_clusters} clusters, {n_noise} noise points)',
            filename=os.path.join(out_dir, f"umap_dbscan_{latent_type}.png"),
            legend_name='DBSCAN Clusters',
            alleles_to_highlight=random_alleles_to_highlight,
            highlight_labels_series=alleles,  # Pass original alleles for highlighting,
            figsize=figsize,
            point_size=point_size
        )
        return clusters


# --- Main Visualization Function ---
def run_visualizations(df, latents_seq, latents_pooled, enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, out_dir,
                       dataset_name: str, figsize=(40, 15), point_size=2):
    """
    Generates and saves a series of publication-quality visualizations for model analysis.
    """
    print("\nGenerating visualizations...")
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Data and Color Preparation ---
    alleles = df['mhc_embedding_key'].apply(clean_key).astype('category')
    unique_alleles = alleles.cat.categories

    # Select 5 random unique alleles to highlight for reproducibility
    num_to_highlight = min(5, len(unique_alleles))
    np.random.seed(999)  # for reproducible random selection
    random_alleles_to_highlight = np.random.choice(unique_alleles, num_to_highlight, replace=False).tolist()
    print(
        f"Found {len(unique_alleles)} unique alleles. Highlighting {num_to_highlight} random alleles: {random_alleles_to_highlight}")

    # Create a publication-friendly color palette for all alleles
    if len(unique_alleles) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_alleles)))
    elif len(unique_alleles) <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_alleles)))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_alleles)))
    allele_color_map = {allele: color for allele, color in zip(unique_alleles, colors)}

    # # --- 2. Sequential Latents Analysis ---
    # print("\n--- Processing Sequential Latents ---")
    # latents_seq_flat = latents_seq.reshape(latents_seq.shape[0], -1)
    #
    # reducer_seq = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    # embedding_seq = reducer_seq.fit_transform(latents_seq_flat)
    #
    # # Plot Raw UMAP of Sequential Latents (all alleles, no highlights)
    # # Plot Raw UMAP of Sequential Latents (with highlights)
    # _plot_umap(
    #     embedding=embedding_seq, labels=alleles, color_map=allele_color_map,
    #     title=f'UMAP of Sequential Latents ({len(df)} Samples)\nColored by {len(unique_alleles)} Unique Alleles',
    #     filename=os.path.join(out_dir, "umap_raw_sequential.png"),
    #     alleles_to_highlight=random_alleles_to_highlight,
    #     figsize=figsize,
    #     point_size=point_size
    # )
    #
    # # Run DBSCAN and Plot Results for Sequential Latents (with random allele highlights)
    # clusters_seq = _run_dbscan_and_plot(
    #     embedding=embedding_seq, alleles=alleles,
    #     random_alleles_to_highlight=random_alleles_to_highlight,
    #     latent_type="sequential", out_dir=out_dir,
    #     figsize=figsize,
    #     point_size=point_size
    # )
    # df['cluster_id_seq'] = clusters_seq

    # --- 3. Pooled Latents Analysis ---
    print("\n--- Processing Pooled Latents ---")
    # using no seed to enable parallelism
    np.random.seed(None)

    reducer_pooled = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_pooled = reducer_pooled.fit_transform(latents_pooled)

    # Plot Raw UMAP of Pooled Latents (with highlights)
    _plot_umap(
        embedding=embedding_pooled, labels=alleles, color_map=allele_color_map,
        title=f'UMAP of Pooled Latents ({len(df)} Samples)\nColored by {len(unique_alleles)} Unique Alleles',
        filename=os.path.join(out_dir, "umap_raw_pooled.png"),
        alleles_to_highlight=random_alleles_to_highlight,
        figsize=figsize,
        point_size=point_size,
        legend_=False
    )

    # Run DBSCAN and Plot Results for Pooled Latents (with random allele highlights)
    clusters_pooled = _run_dbscan_and_plot(
        embedding=embedding_pooled, alleles=alleles,
        random_alleles_to_highlight=random_alleles_to_highlight,
        latent_type="pooled", out_dir=out_dir,
        figsize=figsize,
        point_size=point_size
    )
    df['cluster_id_pooled'] = clusters_pooled

    # Save the dataframe with both cluster assignments
    output_parquet_path = os.path.join(out_dir, f"{dataset_name}_with_clusters.parquet")
    df.to_parquet(output_parquet_path)
    print(f"\n✓ Saved dataset with cluster IDs to {output_parquet_path}")

    # --- 4. Other Visualizations (Inputs and Predictions) ---
    # This part of your code was well-structured and is kept as is.
    print("\n--- Generating supplementary plots (inputs, masks, predictions) ---")

    # Example for the input/mask plots:
    sample_idx = 0
    sample_row = df.iloc[[sample_idx]]
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

    # UMAP plot colored by unique peptide lengths --- print("\n--- Generating UMAP colored by peptide length ---") pep_lengths = df['long_mer'].str.len().astype('category') unique_lengths = sorted(pep_lengths.cat.categories)
    # # Create a color palette for peptide lengths
    pep_lengths = df['long_mer'].str.len().astype('category')
    unique_lengths = sorted(pep_lengths.unique())
    if len(unique_lengths) <= 20:
        colors = sns.color_palette("tab20", n_colors=len(unique_lengths))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_lengths)))
    length_color_map = {length: color for length, color in zip(unique_lengths, colors)}

    _plot_umap(
        embedding=embedding_pooled,
        labels=pep_lengths,
        color_map=length_color_map,
        title=f'UMAP of Pooled Latents Colored by Peptide Length\n({len(unique_lengths)} unique lengths)',
        filename=os.path.join(out_dir, "umap_pooled_by_pep_length.png"),
        legend_name='Peptide Length',
        alleles_to_highlight=None,
        figsize=figsize,
        point_size=point_size
    )

    # UMAP that highlight it wtih only 9 colors: MAMU, PATR, SLA, BOLA, DLA, H-2, HLA-A, HLA-B, and HLA-C, find these based on unique alleles
    print("\n--- Generating UMAP highlighting major allele groups ---")
    major_allele_groups = ['MAMU', 'PATR', 'SLA', 'BOLA', 'DLA', 'H-2', 'HLA-A', 'HLA-B', 'HLA-C', 'HLA-E', 'HLA-DRB', 'HLA-DQA', 'HLA-DQB', 'HLA-DPA', 'HLA-DPB', 'EQCA', 'GOGO']
    def get_allele_group(allele: str, major_groups: list) -> str:
        for group in major_groups:
            if allele.startswith(group):
                return group
        return 'Other'
    # Map each allele to its major group
    group_labels = alleles.apply(lambda x: get_allele_group(x, major_allele_groups)).astype('category')
    unique_groups = sorted(group_labels.unique())
    other_mask = group_labels == 'Other'
    other_alleles = alleles[other_mask].unique()
    other_alleles = np.sort(other_alleles)
    print("\n=== Alleles that fell into the 'Other' category ===")
    if len(other_alleles) == 0:
        print("  (none – every allele matched a major group)")
    else:
        for a in other_alleles:
            print(f"  {a}")
    group_colors = sns.color_palette("tab10", n_colors=len(unique_groups))
    group_color_map = {group: color for group, color in zip(unique_groups, group_colors)}
    if 'Other' in group_color_map:
        # Force gray for the 'Other' group
        group_color_map['Other'] = [0.7, 0.7, 0.7, 0.5]  # muted gray
        # Re‑generate colours for the remaining groups to avoid clashes
        non_other_groups = [g for g in unique_groups if g != 'Other']
        non_other_colors = sns.color_palette("tab10", n_colors=len(non_other_groups))
        for group, color in zip(non_other_groups, non_other_colors):
            group_color_map[group] = color
    _plot_umap(
        embedding=embedding_pooled,
        labels=group_labels,
        color_map=group_color_map,
        title=f'UMAP of Pooled Latents Highlighted by Major Allele Groups\n({len(unique_groups)} groups)',
        filename=os.path.join(out_dir, "umap_pooled_by_allele_group.png"),
        legend_name='Allele Group',
        alleles_to_highlight=None,  # No individual highlighting needed here
        figsize=figsize,
        point_size=point_size,
    )
    print("✓ UMAP by major allele groups saved.")

    # a UMAP plot the categorize by anchor combination: 1- take the second and last position of each peptide amino acids itentity as a pair. 2- you will have 400 types of pair combinations. 3- reduce it to 144 pair combinations by reducing your amino acid alphabet to 12 using MMSEQ2 reduced alphabet. then visualize
    # Apply the function to get anchor pair labels for each peptide
    anchor_pair_labels = df['long_mer'].apply(reduced_anchor_pair).astype('category')
    unique_anchor_pairs = sorted(anchor_pair_labels.unique())

    # Create a color palette for the anchor pairs
    colors = sns.color_palette("hls", n_colors=len(unique_anchor_pairs))

    anchor_color_map = {pair: color for pair, color in zip(unique_anchor_pairs, colors)}
    if 'Short' in anchor_color_map:
        anchor_color_map['Short'] = [0.7, 0.7, 0.7, 0.5]  # Muted gray

    # Plot the UMAP
    _plot_umap(
        embedding=embedding_pooled,
        labels=anchor_pair_labels,
        color_map=anchor_color_map,
        title=f'UMAP of Pooled Latents Colored by Reduced Anchor Pairs\n({len(unique_anchor_pairs)} unique pairs)',
        filename=os.path.join(out_dir, "umap_pooled_by_anchor_pair.png"),
        legend_name='Anchor Pair (Reduced)',
        alleles_to_highlight=None,
        figsize=figsize,
        point_size=point_size,
        legend_=True  # Legend is likely too large, disable it
    )
    print("✓ UMAP by reduced anchor pair saved.")

    # UMAP plot colored by physiochemical properties of peptides: including: 1- hydrophobicity of peptide, 2- charge, 3- polarity
    physiochem_df = df['long_mer'].apply(peptide_properties_biopython).apply(pd.Series)

    # --- Plotting for each property ---
    for prop in ['hydrophobicity', 'charge', 'fraction_polar']:
        print("\n--- Generating UMAP colored by peptide " + prop + " ---")
        prop_values = physiochem_df[prop]
        # Create a continuous color map
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=prop_values.min(), vmax=prop_values.max())
        prop_color_map = {val: cmap(norm(val)) for val in prop_values.unique()}
        _plot_umap(
            embedding=embedding_pooled,
            labels=prop_values,
            color_map=prop_color_map,
            title=f'UMAP of Pooled Latents Colored by Peptide {prop.capitalize()}\n(Continuous Scale)',
            filename=os.path.join(out_dir, f"umap_pooled_by_pep_{prop}.png"),
            legend_style='bar',
        )
        print(f"✓ UMAP by peptide {prop} saved.")

    # C/N - terminal plots
    print("\n--- Generating C/N-terminal amino acid frequency plots ---")
    CN_peptide_df = df['long_mer'].apply(cn_terminal_amino_acids).apply(pd.Series)
    # define a color map for segment_signatures. three regions: N-term, core, C-term, and each one has 3 colors (polar, hydro, charged)
    segment_color_map = {
        'N-term_polar': '#1f77b4', 'N-term_hydrophobic': '#ff7f0e', 'N-term_charged': '#2ca02c',
        'Core_polar': '#d62728', 'Core_hydrophobic': '#9467bd', 'Core_charged': '#8c564b',
        'C-term_polar': '#e377c2', 'C-term_hydrophobic': '#7f7f7f', 'C-term_charged': '#bcbd22'
    }
    for pos in ['N-term', 'Core', 'C-term']:
        prop_values = CN_peptide_df[pos]
        _plot_umap(
            embedding=embedding_pooled,
            labels=prop_values,
            color_map=segment_color_map,
            title=f'UMAP of Pooled Latents Colored by Peptide {pos} Amino Acid Type',
            filename=os.path.join(out_dir, f"umap_pooled_by_pep_{pos}_type.png"),
            legend_name=f'Peptide {pos} Type',
            alleles_to_highlight=None,
            figsize=figsize,
            point_size=point_size,
            legend_=True
        )
        print(f"✓ UMAP by peptide {pos} type saved.")




# Helper function for inference and visualizations
# def run_inference_and_visualizations(df, dataset_name, enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, out_dir, batch_size=32, embed_dim=32):
#     """
#     Runs inference on a dataset, saves latents to memory-mapped files, and generates visualizations.
#     """
#     print(f"\n--- Running Inference & Visualization on {dataset_name.upper()} SET ---")
#     dataset_out_dir = os.path.join(out_dir, f"visuals_{dataset_name}")
#     os.makedirs(dataset_out_dir, exist_ok=True)
#
#     indices = np.arange(len(df))
#
#     # Define paths for memory-mapped arrays
#     # latents_seq_path = os.path.join(dataset_out_dir, f"mhc_latents_sequential_{dataset_name}.mmap")
#     latents_pooled_path = os.path.join(dataset_out_dir, f"mhc_latents_pooled_{dataset_name}.mmap")
#
#     # Create memory-mapped arrays on disk
#     # latents_seq = np.memmap(latents_seq_path, dtype='float32', mode='w+', shape=(len(df), max_mhc_len, embed_dim))
#     latents_pooled = np.memmap(latents_pooled_path, dtype='float32', mode='w+', shape=(len(df), max_mhc_len * 2))
#
#     print(f"Processing {len(df)} samples in batches...")
#     for step in range(0, len(indices), batch_size):
#         batch_idx = indices[step:step + batch_size]
#         batch_df = df.iloc[batch_idx]
#         batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)
#
#         # Run model inference
#         true_preds = enc_dec(batch_data, training=False)
#
#         # Write batch results directly to memory-mapped arrays
#         # latents_seq[batch_idx] = true_preds["cross_latent"].numpy()
#         latents_pooled[batch_idx] = true_preds["latent_vector"].numpy()
#
#         # Flush changes to disk periodically to ensure data is saved
#         if step % (batch_size * 10) == 0:
#             # latents_seq.flush()
#             latents_pooled.flush()
#
#     # Final flush to save any remaining data
#     # latents_seq.flush()
#     latents_pooled.flush()
#
#     # print(f"✓ Sequential latents for {dataset_name} set saved to {latents_seq_path}")
#     print(f"✓ Pooled latents for {dataset_name} set saved to {latents_pooled_path}")
#
#     latents_seq = None
#
#     # Now, call the visualization function with the paths to the memory-mapped files
#     run_visualizations(df, latents_seq, latents_pooled, enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, dataset_out_dir, dataset_name)


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

        # Set up learning rate scheduler
        num_train_steps = (len(df_train) // batch_size) * epochs
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=num_train_steps,
            alpha=0.0  # End at 0
        )
        opt = keras.optimizers.Lion(lr_schedule)
        loss_fn = masked_categorical_crossentropy

        # print model summary
        print("\n--- Model Summary ---")
        enc_dec.build(input_shape={
            "pep_onehot": (None, max_pep_len, 21),  # 21 for amino acids + PAD/MASK
            "pep_mask": (None, max_pep_len),
            "mhc_emb": (None, max_mhc_len, 1152),
            "mhc_mask": (None, max_mhc_len),
            "mhc_onehot": (None, max_mhc_len, 21)  # Assuming MHC also uses one-hot encoding
        })
        enc_dec.summary()

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
            'val_loss': [], 'val_pep_loss': [], 'val_mhc_loss': [], 'lr': []
        }
        best_val_loss = float('inf')
        best_weights_path = os.path.join(out_dir, "best_enc_dec.weights.h5")

        for epoch in range(1, epochs + 1):
            np.random.shuffle(train_indices)
            print(f"\nEpoch {epoch}/{epochs}")

            # --- Training Step ---
            epoch_loss_sum, epoch_pep_loss_sum, epoch_mhc_loss_sum, num_steps = 0, 0, 0, 0
            pbar = tqdm(range(0, len(train_indices), batch_size), desc=f"Epoch {epoch} Training")
            for step in pbar:
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
                current_lr = opt.learning_rate.numpy() if not isinstance(opt.learning_rate, float) else opt.learning_rate
                pbar.set_postfix(loss=f"{loss.numpy():.4f}", lr=f"{current_lr:.2e}")

            history['train_loss'].append(epoch_loss_sum / num_steps)
            history['train_pep_loss'].append(epoch_pep_loss_sum / num_steps)
            history['train_mhc_loss'].append(epoch_mhc_loss_sum / num_steps)
            history['lr'].append(current_lr)
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
        fig, ax1 = plt.subplots(figsize=(12, 5))
        epoch_range = range(1, epochs + 1)

        ax1.plot(epoch_range, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epoch_range, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(epoch_range, history['lr'], 'g--', label='Learning Rate')
        ax2.set_ylabel('Learning Rate', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.legend(loc='upper right')

        fig.suptitle('Training and Validation Loss with Learning Rate Schedule')
        fig.tight_layout()
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
        # run_inference_and_visualizations(df_train, "train", enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, out_dir, batch_size, embed_dim)
        # run_inference_and_visualizations(df_val, "validation", enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, out_dir, batch_size, embed_dim)
        # run_inference_and_visualizations(df_test, "test", enc_dec, max_pep_len, max_mhc_len, seq_map, embed_map, out_dir, batch_size, embed_dim)



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
    # latents_seq_path = os.path.join(infer_out_dir, "mhc_latents_sequential_infer.mmap")
    latents_pooled_path = os.path.join(infer_out_dir, "mhc_latents_pooled_infer.mmap")

    if os.path.exists(latents_pooled_path):
        print(f"Latents data already exists at {latents_pooled_path}. Skipping inference.")
        # Load existing memory-mapped arrays
        # latents_seq = np.memmap(latents_seq_path, dtype='float32',
        #                          mode='r', shape=(len(df_infer), max_mhc_len, embed_dim))
        latents_seq = None
        latents_pooled = np.memmap(latents_pooled_path, dtype='float32',
                                   mode='r', shape=(len(df_infer), max_mhc_len + embed_dim))
        # --- CALL THE VISUALIZATION FUNCTION ---
        run_visualizations(df_infer, latents_seq, latents_pooled, enc_dec, max_pep_len, max_mhc_len, seq_map,
                           embed_map, infer_out_dir, df_name, figsize=(30, 15), point_size=3)
        print("\n✓ Visualization complete.")
        return

    # Create memory-mapped arrays
    # latents_seq = np.memmap(latents_seq_path, dtype='float32', mode='w+', shape=(len(df_infer), max_mhc_len, embed_dim))
    latents_pooled = np.memmap(latents_pooled_path, dtype='float32', mode='w+', shape=(len(df_infer), max_mhc_len + embed_dim))

    print(f"Processing {len(df_infer)} samples for inference in batches...")
    for step in tqdm(range(0, len(indices), batch_size), desc="Inference Progress"):
        batch_idx = indices[step:step + batch_size]
        batch_df = df_infer.iloc[batch_idx]
        batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)

        # Run model in inference mode
        true_preds = enc_dec(batch_data, training=False)

        # Write results to memory-mapped arrays
        # latents_seq[batch_idx] = true_preds["cross_latent"].numpy()
        latents_pooled[batch_idx] = true_preds["latent_vector"].numpy()

    # Flush to ensure all data is written to disk
    # latents_seq.flush()
    latents_pooled.flush()

    # print(f"✓ Sequential latents from inference saved to {latents_seq_path}")
    print(f"✓ Pooled latents from inference saved to {latents_pooled_path}")

    latents_seq = None

    # --- CALL THE VISUALIZATION FUNCTION ---
    # Pass the paths to the memory-mapped files
    run_visualizations(df_infer, latents_seq, latents_pooled, enc_dec, max_pep_len, max_mhc_len, seq_map,
                       embed_map, infer_out_dir, df_name, figsize=(30, 15), point_size=3)
    print("\n✓ Inference and visualization complete.")


def generate_negatives(model_out_dir: str, positives_train: str, positives_validation: str, embed_npz: str,
                       seq_csv: str, embd_key_path: str, batch_size: int = 256):
    """
    Generates a dataset of negative samples using a trained model.

    This function works by:
    1. Running inference on a complete dataset of peptide-MHC pairs to get latent representations.
    2. Clustering these samples in the latent space using DBSCAN (performed inside `infer`).
    3. Calculating the centroid (mean latent vector) for each cluster.
    4. Applying hierarchical clustering to the cluster centroids to determine cluster relationships.
    5. For each cluster, identifying the most distant clusters based on hierarchical clustering.
    6. Creating new "negative" pairs by swapping the MHC alleles of samples with alleles from these distant clusters.
    7. Saving the newly generated negative dataset.

    Args:
        model_out_dir (str): Path to the output directory of a trained model, containing weights and config.
        positives_train (str): Path to the Parquet file containing training positive samples.
        positives_validation (str): Path to the Parquet file containing validation positive samples.
        embed_npz (str): Path to the NPZ file with MHC embeddings.
        seq_csv (str): Path to the CSV file with MHC sequences.
        embd_key_path (str): Path to the CSV file mapping embedding keys.
        batch_size (int): Batch size for the inference process.
    """
    print("\n--- Starting Negative Sample Generation (DBSCAN + Hierarchical) ---")
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
        print(f"Latents data not found at {clustered_df_path}. Running inference first...")
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
        print(f"Found existing latents data at {clustered_df_path}.")
    # 2. Load the clustered data and the corresponding latent vectors
    df_clustered = pd.read_parquet(clustered_df_path)
    # Load the pooled latents from the memory-mapped file
    # We need to know the shape, which we can get from the model config
    config_path = os.path.join(model_out_dir, "model_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found at {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    embed_dim = config['embed_dim']
    max_pep_len = config['max_pep_len']
    max_mhc_len = config['max_mhc_len']
    latents_pooled_path = os.path.join(infer_dir, "mhc_latents_pooled_infer.mmap")
    latents_pooled = np.memmap(latents_pooled_path, dtype='float32', mode='r',
                               shape=(len(df_clustered), max_mhc_len + embed_dim))
    # save mmap configs
    mmap_config_path = os.path.join(infer_dir, "mmap_config.json")
    mmap_config = {
        'latents_pooled_shape': latents_pooled.shape,
        'embed_dim': embed_dim,
        'max_pep_len': config['max_pep_len'],
        'max_mhc_len': config['max_mhc_len']
    }
    with open(mmap_config_path, 'w') as f:
        json.dump(mmap_config, f, indent=4)
    print(f"✓ Saved memory-mapped latents configuration to {mmap_config_path}")
    print(f"✓ Loaded {len(df_clustered)} samples with cluster IDs and memory-mapped latents.")
    # Exclude noise points from negative generation logic
    non_noise_mask = df_clustered['cluster_id_pooled'] != -1
    df_non_noise = df_clustered[non_noise_mask].copy()
    latents_non_noise = latents_pooled[non_noise_mask]
    print(f"  Proceeding with {len(df_non_noise)} non-noise samples.")
    if len(df_non_noise) == 0:
        print("Warning: No clusters found (all points are noise). Cannot generate negatives.")
        return
    # 3. Calculate the centroid for each cluster
    print("Calculating cluster centroids...")
    cluster_ids = sorted(df_non_noise['cluster_id_pooled'].unique())
    cluster_centroids = {}
    for cid in cluster_ids:
        cluster_mask = df_non_noise['cluster_id_pooled'] == cid
        cluster_centroids[cid] = latents_non_noise[cluster_mask].mean(axis=0)
    print(f"✓ Calculated centroids for {len(cluster_centroids)} clusters.")
    if len(cluster_centroids) < 2:
        print("Warning: Fewer than 2 clusters found. Cannot determine distant clusters for negative sampling.")
        return
    # 4. Apply hierarchical clustering to the DBSCAN cluster centroids
    print("Applying hierarchical clustering to DBSCAN cluster centroids...")
    sorted_cids = sorted(cluster_centroids.keys())
    centroid_matrix = np.array([cluster_centroids[cid] for cid in sorted_cids])
    # Perform hierarchical clustering on centroids
    linkage_matrix = linkage(centroid_matrix, method='ward')
    # Create dendrogram and save it
    plt.figure(figsize=(15, 10))
    dendrogram(linkage_matrix, labels=sorted_cids)
    plt.title("Hierarchical Clustering Dendrogram of DBSCAN Cluster Centroids")
    plt.xlabel("Cluster ID")
    plt.ylabel("Distance")
    dendrogram_path = os.path.join(model_out_dir, "cluster_dendrogram.png")
    plt.savefig(dendrogram_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved dendrogram to {dendrogram_path}")
    # 5. Extract hierarchical distances from linkage matrix
    # Convert linkage matrix to distance matrix format
    # Get cophenetic distances (hierarchical distances between clusters)
    cophenet_dists = cophenet(linkage_matrix)
    dist_matrix = squareform(cophenet_dists)
    # Visualize and save the hierarchical distance matrix
    plt.figure(figsize=(30, 15))
    sns.heatmap(dist_matrix, xticklabels=sorted_cids, yticklabels=sorted_cids, cmap="viridis", annot=True, fmt=".2f")
    plt.title("Hierarchical Distance Matrix Between DBSCAN Clusters")
    plt.xlabel("Cluster ID")
    plt.ylabel("Cluster ID")
    plt.tight_layout()
    dist_matrix_path = os.path.join(model_out_dir, "cluster_hierarchical_distance_matrix.png")
    plt.savefig(dist_matrix_path, dpi=600)
    plt.close()
    print(f"✓ Saved hierarchical distance matrix plot to {dist_matrix_path}")
    # Set diagonal to a large value to ignore self-distance
    np.fill_diagonal(dist_matrix, -np.inf)  # Use -inf for argmax
    # Find the index of the most distant cluster for each cluster
    farthest_cluster_indices = np.argmax(dist_matrix, axis=1)
    farthest_cluster_map = {
        own_cid: sorted_cids[farthest_idx]
        for own_cid, farthest_idx in zip(sorted_cids, farthest_cluster_indices)
    }
    print("✓ Determined the most distant cluster for each cluster using hierarchical clustering analysis.")
    # 6. Generate negative samples
    new_negatives = []
    # Get a representative set of alleles from each cluster
    alleles_per_cluster = df_non_noise.groupby('cluster_id_pooled')['allele'].apply(
        lambda x: list(set(x))
    )
    for _, row in df_non_noise.iterrows():
        current_cluster = row['cluster_id_pooled']
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
    # 7. Save the new dataset
    negatives_output_path = os.path.join(model_out_dir, f"generated_negatives{MHC_CLASS}.parquet")
    df_negatives.to_parquet(negatives_output_path, index=False)
    # Save allele names to a text file, separated by commas
    alleles_output_path = os.path.join(model_out_dir, f"generated_negatives{MHC_CLASS}_alleles.txt")
    if 'allele' in df_negatives.columns:
        alleles = df_negatives['allele'].unique()
        with open(alleles_output_path, 'w') as f:
            f.write(",".join(map(str, alleles)))
        print(f"✓ Saved allele names to {alleles_output_path}")
    else:
        print("Warning: 'allele' column not found in negatives, skipping allele file generation.")
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
    final_dataset = final_dataset.drop(columns=['cluster_id_pooled', 'cluster_id_seq'], errors='ignore')
    final_dataset_path = os.path.join(model_out_dir,
                                      f"binding_affinity_dataset_with_swapped_negatives{MHC_CLASS}.parquet")
    final_dataset.to_parquet(final_dataset_path, index=False)
    print(f"✓ Saved combined dataset with positives and negatives to {final_dataset_path}")


if __name__ == "__main__":
    # Suppress verbose TensorFlow logging, but keep errors.
    MHC_CLASS = 1
    noise_std = 0.1
    heads = 4
    embed_dim = 21
    batch_size = 100
    epochs = 3
    lr = 1e-5
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    train_path = f"../data/binding_affinity_data/positives_class{MHC_CLASS}_train.parquet"
    validate_path = f"../data/binding_affinity_data/positives_class{MHC_CLASS}_val.parquet"
    test_path = f"../data/binding_affinity_data/positives_class{MHC_CLASS}_test.parquet"
    embed_npz_path = f"/media/amirreza/lasse/ESM/esmc_600m/PMGen_whole_seq/mhc{MHC_CLASS}_encodings.npz"
    embd_key_path = f"/media/amirreza/lasse/ESM/esmc_600m/PMGen_whole_seq/mhc{MHC_CLASS}_encodings.csv"
    seq_csv_path = f"../data/alleles/aligned_PMGen_class_{MHC_CLASS}.csv"
    base_out_dir = f"/media/amirreza/lasse/PMClust_runs/run_PMClust_ns_{noise_std}_hds_{heads}_zdim_{embed_dim}_L{MHC_CLASS}_all/"
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
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     lr=lr,
    #     embed_dim=embed_dim,
    #     heads=heads,
    #     noise_std=noise_std
    # )

    out_dir = f"{base_out_dir}1"
    # infer(
    #     parquet_path=validate_path,
    #     embed_npz=embed_npz_path,
    #     seq_csv=seq_csv_path,
    #     embd_key_path=embd_key_path,
    #     out_dir=out_dir,
    #     batch_size=256,
    #     df_name="inference_validation"
    # )

    # out_dir = f"/home/amirreza/Desktop/PMBind/results/run_PMClust_ns_0.1_hds_2_zdim_21_L1/12"
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
