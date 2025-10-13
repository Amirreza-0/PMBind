# visualize tensorflow model

import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import uuid
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import matplotlib.colors as mcolors
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from utils import (AttentionLayer, PositionalEncoding, AnchorPositionExtractor, SplitLayer, ConcatMask,
                   MaskedEmbedding, reduced_anchor_pair, cn_terminal_amino_acids, peptide_properties_biopython)

from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from collections import Counter


def visualize_tf_model(model_path='h5/bicross_encoder_decoder.h5'):
    """
    Visualizes the TensorFlow model architecture and saves it as an image.
    :return:
        None
    """

    def wrap_layer(layer_class):
        def fn(**config):
            config.pop('trainable', None)
            config.pop('dtype', None)
            # assign a unique name to avoid duplicates
            config['name'] = f"{layer_class.__name__.lower()}_{uuid.uuid4().hex[:8]}"
            return layer_class.from_config(config)

        return fn

    # Load the model with wrapped custom objects
    model = load_model(
        model_path,
        custom_objects={
            'AttentionLayer': wrap_layer(AttentionLayer),
            'PositionalEncoding': wrap_layer(PositionalEncoding),
            'AnchorPositionExtractor': wrap_layer(AnchorPositionExtractor),
            'SplitLayer': wrap_layer(SplitLayer),
            'ConcatMask': wrap_layer(ConcatMask),
            'MaskedEmbedding': wrap_layer(MaskedEmbedding),
        }
    )

    # Display model summary
    model.summary()

    # Create better visualization as SVG with cleaner layout
    tf.keras.utils.plot_model(
        model,
        to_file='h5/model_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',  # Top to bottom layout
        dpi=200,  # Higher resolution
        expand_nested=True,  # Expand nested models to show all layers
        show_layer_activations=True  # Show activation functions
    )


def visualize_attention_weights(attn_weights, peptide_seq, mhc_seq, max_pep_len, max_mhc_len,
                               out_dir, sample_idx=0, head_idx=None, save_all_heads=False):
    """
    Visualize attention weights from the PMBind model.

    Args:
        attn_weights: Tensor of shape (B, heads, P+M, P+M) with attention scores.
        peptide_seq: Peptide sequence string.
        mhc_seq: MHC sequence string (truncated to fit max_mhc_len).
        max_pep_len: Maximum peptide length.
        max_mhc_len: Maximum MHC length.
        out_dir: Output directory to save plots.
        sample_idx: Index of sample to visualize (default: 0).
        head_idx: Index of attention head to visualize (default: None, average all heads).
        save_all_heads: Whether to save individual plots for all attention heads.

    Returns:
        None
    """
    os.makedirs(out_dir, exist_ok=True)

    # Convert to numpy for visualization
    try:
        attn_weights = attn_weights.numpy()
    except:
        pass  # Already numpy

    # Get attention weights for the specified sample
    sample_attn = attn_weights[sample_idx]  # Shape: (heads, P+M, P+M)

    # Prepare sequence labels
    pep_len = min(len(peptide_seq), max_pep_len)
    mhc_len = min(len(mhc_seq), max_mhc_len)

    # Create sequence labels for visualization
    pep_labels = [f"P{i+1}:{peptide_seq[i]}" for i in range(pep_len)]
    mhc_labels = [f"M{i+1}:{mhc_seq[i]}" for i in range(mhc_len)]
    seq_labels = pep_labels + mhc_labels

    # Only consider the actual sequence lengths, not padded positions
    relevant_attn = sample_attn[:, :pep_len+mhc_len, :pep_len+mhc_len]

    if head_idx is not None:
        # Visualize specific head
        _plot_single_attention_head(relevant_attn[head_idx], seq_labels, pep_len, mhc_len,
                                  f'Attention Weights - Head {head_idx} (Sample {sample_idx})',
                                  os.path.join(out_dir, f"attention_head_{head_idx}_sample_{sample_idx}.png"))
    else:
        # Average across all heads
        avg_attn = np.mean(relevant_attn, axis=0)
        _plot_single_attention_head(avg_attn, seq_labels, pep_len, mhc_len,
                                  f'Average Attention Weights (Sample {sample_idx})',
                                  os.path.join(out_dir, f"attention_avg_sample_{sample_idx}.png"))

    if save_all_heads:
        # Save individual plots for each attention head
        for h in range(relevant_attn.shape[0]):
            _plot_single_attention_head(relevant_attn[h], seq_labels, pep_len, mhc_len,
                                      f'Attention Weights - Head {h} (Sample {sample_idx})',
                                      os.path.join(out_dir, f"attention_head_{h}_sample_{sample_idx}.png"))

    # Create peptide-to-MHC and MHC-to-peptide cross-attention visualizations
    _plot_cross_attention_maps(relevant_attn, peptide_seq, mhc_seq, pep_len, mhc_len,
                              out_dir, sample_idx)


def _plot_single_attention_head(attn_matrix, seq_labels, pep_len, mhc_len, title, filename):
    """Helper function to plot a single attention head."""
    plt.figure(figsize=(max(12, len(seq_labels) * 0.5), max(10, len(seq_labels) * 0.4)))

    # Create heatmap
    sns.heatmap(attn_matrix,
                xticklabels=seq_labels,
                yticklabels=seq_labels,
                cmap='viridis',
                cbar_kws={'label': 'Attention Score'},
                square=True,
                linewidths=0.1)

    # Add separator lines between peptide and MHC
    plt.axvline(x=pep_len, color='red', linewidth=2, linestyle='--', alpha=0.7)
    plt.axhline(y=pep_len, color='red', linewidth=2, linestyle='--', alpha=0.7)

    # Add text annotations for quadrants
    plt.text(pep_len/2, -1, 'Peptide', ha='center', va='top', fontweight='bold', fontsize=12)
    plt.text(pep_len + mhc_len/2, -1, 'MHC', ha='center', va='top', fontweight='bold', fontsize=12)
    plt.text(-1, pep_len/2, 'Peptide', ha='right', va='center', fontweight='bold', fontsize=12, rotation=90)
    plt.text(-1, pep_len + mhc_len/2, 'MHC', ha='right', va='center', fontweight='bold', fontsize=12, rotation=90)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Key/Value Positions', fontsize=12)
    plt.ylabel('Query Positions', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Attention plot saved to {filename}")


def _plot_cross_attention_maps(attn_weights, peptide_seq, mhc_seq, pep_len, mhc_len, out_dir, sample_idx):
    """Plot cross-attention maps between peptide and MHC."""
    # Average across heads for cross-attention analysis
    avg_attn = np.mean(attn_weights, axis=0)

    # Extract cross-attention blocks
    pep_to_mhc = avg_attn[:pep_len, pep_len:pep_len+mhc_len]  # Peptide queries attending to MHC
    mhc_to_pep = avg_attn[pep_len:pep_len+mhc_len, :pep_len]  # MHC queries attending to peptide

    # Plot peptide-to-MHC attention
    plt.figure(figsize=(max(10, mhc_len * 0.5), max(6, pep_len * 0.5)))
    sns.heatmap(pep_to_mhc,
                xticklabels=[f"M{i+1}:{mhc_seq[i]}" for i in range(mhc_len)],
                yticklabels=[f"P{i+1}:{peptide_seq[i]}" for i in range(pep_len)],
                cmap='viridis',
                cbar_kws={'label': 'Attention Score'},
                annot=True if pep_len <= 15 and mhc_len <= 20 else False,
                fmt='.2f')
    plt.title(f'Peptide → MHC Cross-Attention (Sample {sample_idx})', fontsize=14, fontweight='bold')
    plt.xlabel('MHC Positions', fontsize=12)
    plt.ylabel('Peptide Positions', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cross_attention_pep_to_mhc_sample_{sample_idx}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Plot MHC-to-peptide attention
    plt.figure(figsize=(max(8, pep_len * 0.5), max(8, mhc_len * 0.4)))
    sns.heatmap(mhc_to_pep,
                xticklabels=[f"P{i+1}:{peptide_seq[i]}" for i in range(pep_len)],
                yticklabels=[f"M{i+1}:{mhc_seq[i]}" for i in range(mhc_len)],
                cmap='viridis',
                cbar_kws={'label': 'Attention Score'},
                annot=True if pep_len <= 15 and mhc_len <= 20 else False,
                fmt='.2f')
    plt.title(f'MHC → Peptide Cross-Attention (Sample {sample_idx})', fontsize=14, fontweight='bold')
    plt.xlabel('Peptide Positions', fontsize=12)
    plt.ylabel('MHC Positions', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cross_attention_mhc_to_pep_sample_{sample_idx}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Cross-attention plots saved for sample {sample_idx}")


def visualize_cross_attention_weights(cross_attn_scores, peptide_seq, mhc_seq):
    """
    Legacy function for backward compatibility.
    Visualize cross-attention weights between peptide and MHC sequences.

    Args:
        cross_attn_scores: Tensor of shape (B, N_peptide, N_mhc) with attention scores.
        peptide_seq: List of peptide sequences.
        mhc_seq: List of MHC sequences.

    Returns:
        None
    """

    # Convert to numpy for visualization
    try:
        cross_attn_scores = cross_attn_scores.numpy()
    except:
        cross_attn_scores = cross_attn_scores

    cross_attn_scores = cross_attn_scores.mean(axis=0)  # Average over heads dimension
    plt.figure(figsize=(12, 8))
    if cross_attn_scores.shape[0] == cross_attn_scores.shape[1]:
        # If square matrix, use heatmap
        sns.heatmap(cross_attn_scores, annot=True, fmt=".2f",
                    yticklabels=peptide_seq, xticklabels=peptide_seq,
                    cmap='viridis', cbar_kws={'label': 'Attention Score'})
    else:
        sns.heatmap(cross_attn_scores, annot=True, fmt=".2f",
                    yticklabels=mhc_seq, xticklabels=peptide_seq,
                    cmap='viridis', cbar_kws={'label': 'Attention Score'})
    plt.title(f'Cross-Attention Weights for Sample')
    plt.xlabel('Sequence')
    plt.ylabel('Sequence')
    plt.show()


# if __name__ == "__main__":
#     visualize_tf_model()
#     print("Model visualization saved as 'h5/model_architecture_decoder.png'")


def plot_1d_heatmap(data, cmap='viridis', title='1D Heatmap', xlabel='', ylabel=''):
    """
    Plots a heatmap of a 1D array by expanding it to 2D (N, 1).

    Parameters:
        data (array-like): 1D input array of shape (N,)
        cmap (str): Matplotlib colormap
        title (str): Plot title
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
    """
    data = np.array(data).reshape(-1, 1)  # Shape (N,) → (N, 1)

    plt.figure(figsize=(10, 7))  # Adjust figure size based on N
    plt.imshow(data, aspect='auto', cmap=cmap)
    plt.colorbar(label='Value')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks([0])  # Only one column
    plt.yticks(np.arange(len(data)))
    plt.tight_layout()
    plt.show()


def _plot_umap(
        embedding: np.ndarray,
        labels: pd.Series,
        color_map: dict,
        title: str,
        filename: str,
        legend_name: str = "Alleles",
        alleles_to_highlight: list | None = None,
        highlight_labels_series: pd.Series | None = None,
        highlight_mask: np.ndarray | None = None,
        figsize: tuple = (40, 15),
        point_size: int = 2,
        legend_: bool = True,
        legend_style: str = "detailed",
        legend_font_size: int = 16,
        cbar_font_size: int = 12,
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
    # Store default grey for missing labels
    default_grey = (0.5, 0.5, 0.5, 0.5)
    # 1.  Plot points
    unique_labels_from_series = labels.unique()
    # Ensure labels are sorted for consistent color mapping, especially for numeric labels
    try:
        unique_labels = sorted(unique_labels_from_series)
    except TypeError:  # handles non-sortable types if they occur
        unique_labels = list(unique_labels_from_series)

    n_labels = len(unique_labels)
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
            s=point_size, alpha=0.7, rasterized=True
        )

    else:  # "detailed"
        sc = None  # in case someone needs it later
        for lbl in unique_labels:
            mask = (labels == lbl)
            colour = color_map.get(lbl, default_grey)
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=[colour], label=lbl, s=point_size, alpha=0.7, rasterized=True
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
    # Highlight samples based on the mask with a different shape
    if highlight_mask is not None and np.any(highlight_mask):
        ax.scatter(embedding[highlight_mask, 0], embedding[highlight_mask, 1],
                   c='none', marker='X', s=point_size * 3,
                   edgecolor='black', linewidth=0.5, label='Test Sample')
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1', fontsize=16)
    ax.set_ylabel('UMAP Dimension 2', fontsize=16)

    # 4.  Legend / colour-bar handling
    if legend_:
        if legend_style.lower() == "bar":
            # --- compact colour-bar
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(legend_name, fontsize=cbar_font_size)

            # Show a manageable number of ticks
            MAX_TICKS = 25
            if n_labels <= MAX_TICKS:
                tick_positions = np.arange(n_labels)
                tick_labels = unique_labels
            else:
                tick_positions = [0, n_labels - 1]
                tick_labels = [unique_labels[0], unique_labels[-1]]

            # Format numeric tick labels to show fewer decimal places
            formatted_tick_labels = []
            for label in tick_labels:
                try:
                    # Try to format as a number with 2 decimal places
                    formatted_label = f"{float(label):.2f}"
                    formatted_tick_labels.append(formatted_label)
                except (ValueError, TypeError):
                    # If it's not a number, use the original label
                    formatted_tick_labels.append(str(label))

            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(formatted_tick_labels, fontsize=cbar_font_size)

            # remove colour-bar borders that create a grey stripe
            if hasattr(cbar, "solids"):
                cbar.solids.set_edgecolor("face")
                cbar.solids.set_linewidth(0)
            cbar.outline.set_visible(False)

        else:  # --- detailed legend
            handles, legend_labels = ax.get_legend_handles_labels()

            highlight_handles = [h for h, l in zip(handles, legend_labels)
                                 if 'highlighted' in l]
            highlight_labels = [l for l in legend_labels
                                if 'highlighted' in l]

            regular_handles = [h for h, l in zip(handles, legend_labels)
                               if 'highlighted' not in l]
            regular_labels = [l for l in legend_labels
                              if 'highlighted' not in l]

            # separate legend for highlights
            if highlight_handles:
                first_legend = ax.legend(
                    highlight_handles, highlight_labels,
                    title='Highlighted Alleles', bbox_to_anchor=(1.05, 1),
                    loc='upper left', frameon=True, fontsize=legend_font_size,
                    title_fontsize=20, markerscale=4.0
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

                # Format numeric tick labels to show fewer decimal places
                formatted_tick_labels = []
                for label in tick_labels:
                    try:
                        # Try to format as a number with 2 decimal places
                        formatted_label = f"{float(label):.2f}"
                        formatted_tick_labels.append(formatted_label)
                    except (ValueError, TypeError):
                        # If it's not a number, use the original label
                        formatted_tick_labels.append(str(label))

                cbar.set_ticks(tick_positions)
                cbar.ax.set_yticklabels(formatted_tick_labels, fontsize=cbar_font_size)
                cbar.set_label(legend_name, fontsize=20)

                if hasattr(cbar, "solids"):
                    cbar.solids.set_edgecolor("face")
                    cbar.solids.set_linewidth(0)
                cbar.outline.set_visible(False)

            else:
                # ---- normal legend with handles
                ax.legend(
                    regular_handles, regular_labels, title=legend_name,
                    bbox_to_anchor=(1.05, 0.75 if highlight_handles else 1),
                    loc='upper left', fontsize=legend_font_size, frameon=True,
                    title_fontsize=20, markerscale=4.0
                )

    # 5.  Save & close
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Plot saved to {filename}")


def _run_dbscan_and_plot(
        embedding: np.ndarray,
        alleles: pd.Series,
        latent_type: str,
        out_dir: str,
        random_alleles_to_highlight: list | None = None,
        figsize=(40, 15),
        point_size=2,
        highlight_mask: np.ndarray | None = None,
        legend_font_size: int = 16,
        cbar_font_size: int = 12
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
        highlight_mask=highlight_mask,
        alleles_to_highlight=random_alleles_to_highlight if random_alleles_to_highlight else None,
        highlight_labels_series=alleles,  # Pass original alleles for highlighting,
        figsize=figsize,
        point_size=point_size,
        legend_=True,
        legend_style='detailed',
        legend_font_size=legend_font_size,
        cbar_font_size=cbar_font_size
    )
    return clusters


def _analyze_latents(latents, df, alleles, allele_color_map, random_alleles_to_highlight,
                     latent_type: str, out_dir: str, dataset_name: str,
                     figsize=(40, 15), point_size=2, highlight_mask: np.ndarray | None = None,
                     legend_font_size: int = 25, cbar_font_size: int = 20):
    """
    Internal helper to run UMAP, DBSCAN, and generate standard plots for a given latent space.
    """
    print(f"\n--- Processing {latent_type.capitalize()} Latents ---")
    np.random.seed(999)  # for reproducibility

    # Ensure latents are 2D
    if latents.ndim > 2:
        latents = latents.reshape((latents.shape[0], -1))

    # --- UMAP Reduction ---
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, random_state=42)
    embedding = reducer.fit_transform(latents)

    # --- Initial UMAP Plot (Colored by Allele) ---
    _plot_umap(
        embedding=embedding, labels=alleles, color_map=allele_color_map,
        title=f'UMAP of {latent_type.capitalize()} Latents ({len(df)} Samples)\nColored by {len(alleles.cat.categories)} Unique Alleles',
        filename=os.path.join(out_dir, f"umap_raw_{latent_type}.png"),
        alleles_to_highlight=random_alleles_to_highlight,
        figsize=figsize,
        point_size=point_size,
        legend_=False,
        highlight_mask=highlight_mask,
        legend_font_size=legend_font_size,
        cbar_font_size=cbar_font_size
    )

    # --- DBSCAN Clustering ---
    label_colors = {0: 'blue', 1: 'red'}
    _plot_umap(
        embedding=embedding, labels=df['assigned_label'], color_map=label_colors,
        title=f'UMAP of {latent_type.capitalize()} Latents Colored by Assigned Label',
        filename=os.path.join(out_dir, f"umap_{latent_type}_by_assigned_label.png"),
        legend_name='Assigned Label (0=Neg, 1=Pos)',
        highlight_mask=highlight_mask,
        figsize=figsize, point_size=point_size,
        legend_=True,
        legend_font_size=legend_font_size,
        cbar_font_size=cbar_font_size
    )

    clusters = _run_dbscan_and_plot(
        embedding=embedding, alleles=alleles,
        latent_type=latent_type, out_dir=out_dir,
        figsize=figsize,
        point_size=point_size,
        highlight_mask=highlight_mask,
        legend_font_size=legend_font_size,
        cbar_font_size=cbar_font_size
    )
    df[f'cluster_id_{latent_type}'] = clusters

    # --- UMAP Plots with Categorical Coloring ---
    # Plot by Peptide Length
    pep_lengths = df['long_mer'].str.len().astype('category')
    unique_lengths = sorted(pep_lengths.unique())
    if len(unique_lengths) <= 20:
        colors = sns.color_palette("tab20", n_colors=len(unique_lengths))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_lengths)))
    length_color_map = {length: color for length, color in zip(unique_lengths, colors)}

    _plot_umap(
        embedding=embedding, labels=pep_lengths, color_map=length_color_map,
        title=f'UMAP of {latent_type.capitalize()} Latents Colored by Peptide Length\n({len(unique_lengths)} unique lengths)',
        filename=os.path.join(out_dir, f"umap_{latent_type}_by_pep_length.png"),
        legend_name='Peptide Length', highlight_mask=highlight_mask, figsize=figsize, point_size=point_size,
        legend_=True, legend_font_size=legend_font_size, cbar_font_size=cbar_font_size
    )

    # Plot by Major Allele Group
    major_allele_groups = ['MAMU', 'PATR', 'SLA', 'BOLA', 'DLA', 'H-2', 'HLA-A', 'HLA-B', 'HLA-C', 'HLA-E', 'HLA-DRB',
                           'HLA-DQA', 'HLA-DQB', 'HLA-DPA', 'HLA-DPB', 'EQCA', 'GOGO']

    def get_allele_group(allele: str, major_groups: list) -> str:
        for group in major_groups:
            if allele.startswith(group):
                return group
        return 'Other'

    group_labels = alleles.apply(lambda x: get_allele_group(x, major_allele_groups)).astype('category')
    unique_groups = sorted(group_labels.unique())
    group_colors = sns.color_palette("tab10", n_colors=len(unique_groups))
    group_color_map = {group: color for group, color in zip(unique_groups, group_colors)}
    if 'Other' in group_color_map:
        group_color_map['Other'] = [0.7, 0.7, 0.7, 0.5]  # Muted gray

    _plot_umap(
        embedding=embedding, labels=group_labels, color_map=group_color_map,
        title=f'UMAP of {latent_type.capitalize()} Latents by Major Allele Groups\n({len(unique_groups)} groups)',
        filename=os.path.join(out_dir, f"umap_{latent_type}_by_allele_group.png"),
        legend_name='Allele Group', figsize=figsize, point_size=point_size,
        legend_=True, legend_font_size=legend_font_size, cbar_font_size=cbar_font_size,
        alleles_to_highlight=random_alleles_to_highlight,
    )

    # Plot by Reduced Anchor Pair (old method for comparison)
    anchor_pair_labels = df['long_mer'].apply(reduced_anchor_pair).astype('category')
    unique_anchor_pairs = sorted(anchor_pair_labels.unique())
    colors = sns.color_palette("hls", n_colors=len(unique_anchor_pairs))
    anchor_color_map = {pair: color for pair, color in zip(unique_anchor_pairs, colors)}
    if 'Short' in anchor_color_map:
        anchor_color_map['Short'] = [0.7, 0.7, 0.7, 0.5]

    _plot_umap(
        embedding=embedding, labels=anchor_pair_labels, color_map=anchor_color_map,
        title=f'UMAP of {latent_type.capitalize()} Latents by Reduced Anchor Pairs\n({len(unique_anchor_pairs)} unique pairs)',
        filename=os.path.join(out_dir, f"umap_{latent_type}_by_anchor_pair_reduced_legacy.png"),
        legend_name='Anchor Pair (Reduced)', figsize=figsize, point_size=point_size, legend_=True,
        legend_font_size=legend_font_size // 2, cbar_font_size=cbar_font_size // 2
    )

    # Plot by Anchor Pair with Amino Acid Colors
    plot_anchor_pairs_with_amino_acid_colors_extended(
        embedding=embedding,
        peptide_sequences=df['long_mer'],
        title=f'UMAP of {latent_type.capitalize()} Latents by Anchor Pairs\nColored by 1st Anchor AA (face) and 2nd Anchor AA (outline)',
        filename=os.path.join(out_dir, f"umap_{latent_type}_by_anchor_pair_aa_colors_extended.png"),
        figsize=figsize,
        point_size=point_size
    )
    plot_anchor_pairs_with_amino_acid_colors(
        embedding=embedding,
        peptide_sequences=df['long_mer'],
        title=f'UMAP of {latent_type.capitalize()} Latents by Anchor Pairs\nColored by 1st Anchor AA (face) and 2nd Anchor AA (outline)',
        filename=os.path.join(out_dir, f"umap_{latent_type}_by_anchor_pair_aa_colors_legacy.png"),
        figsize=figsize,
        point_size=point_size
    )


    # Plot by C/N-terminal Amino Acid Type
    CN_peptide_df = df['long_mer'].apply(cn_terminal_amino_acids).apply(pd.Series)
    segment_color_map = {
        'N-term_polar': '#1f77b4', 'N-term_hydrophobic': '#ff7f0e', 'N-term_charged': '#2ca02c',
        'Core_polar': '#d62728', 'Core_hydrophobic': '#9467bd', 'Core_charged': '#8c564b',
        'C-term_polar': '#e377c2', 'C-term_hydrophobic': '#7f7f7f', 'C-term_charged': '#bcbd22'
    }
    for pos in ['N-term', 'Core', 'C-term']:
        prop_values = CN_peptide_df[pos]
        _plot_umap(
            embedding=embedding, labels=prop_values, color_map=segment_color_map,
            title=f'UMAP of {latent_type.capitalize()} Latents by Peptide {pos} Type',
            filename=os.path.join(out_dir, f"umap_{latent_type}_by_pep_{pos}_type.png"),
            legend_name=f'Peptide {pos} Type', figsize=figsize, point_size=point_size, legend_=True,
            legend_font_size=legend_font_size, cbar_font_size=cbar_font_size
        )

    # --- UMAP Plots with Continuous Coloring (Physiochemical Properties) ---
    physiochem_df = df['long_mer'].apply(peptide_properties_biopython).apply(pd.Series)
    for prop in ['hydrophobicity', 'charge', 'fraction_polar']:
        prop_values = physiochem_df[prop]
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=prop_values.min(), vmax=prop_values.max())
        prop_color_map = {val: cmap(norm(val)) for val in prop_values.unique()}
        _plot_umap(
            embedding=embedding, labels=prop_values, color_map=prop_color_map,
            title=f'UMAP of {latent_type.capitalize()} Latents by Peptide {prop.capitalize()}',
            filename=os.path.join(out_dir, f"umap_{latent_type}_by_pep_{prop}.png"),
            legend_style='bar', figsize=figsize, point_size=point_size,
            legend_=True, legend_font_size=legend_font_size, cbar_font_size=cbar_font_size
        )

    print(f"✓ All visualizations for {latent_type.capitalize()} latents saved.")

    # --- 4. Other Visualizations (Inputs and Predictions) ---

    return df


def visualize_training_history(history, out_path='h5'):
    """
    Plots training and validation metrics over epochs.

    Parameters:
        history: Dict with training history, keys like 'train_loss', 'val_auc', etc.
        out_path: Path to save the plot image.
    """
    import matplotlib.pyplot as plt

    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)

    # Determine epoch range from any available metric
    epoch_key = 'loss' if 'loss' in history else 'auc' if 'auc' in history else list(history.keys())[0]
    epochs = range(1, len(history[epoch_key]) + 1)

    # Plot losses
    if 'loss' in history:
        axes[0, 0].plot(epochs, history['loss'], label='Train Loss', color='blue')
    if 'val_loss' in history:
        axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', color='orange')
    if 'class_loss' in history:
        axes[0, 0].plot(epochs, history['class_loss'], label='CLS Loss', color='red')
    if 'pep_recon_loss' in history:
        axes[0, 0].plot(epochs, history['pep_recon_loss'], label='PEP Loss', color='green')
    if 'mhc_recon_loss' in history:
        axes[0, 0].plot(epochs, history['mhc_recon_loss'], label='MHC Loss', color='purple')
    axes[0, 0].set_title('Losses')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot AUC
    if 'auc' in history:
        axes[0, 1].plot(epochs, history['auc'], label='Train AUC', color='blue')
    if 'val_auc' in history:
        axes[0, 1].plot(epochs, history['val_auc'], label='Val AUC', color='orange')
    if 'mcc' in history:
        axes[0, 1].plot(epochs, history['mcc'], label='Train MCC', color='red')
    if 'val_mcc' in history:
        axes[0, 1].plot(epochs, history['val_mcc'], label='Val MCC', color='green')
    axes[0, 1].set_title('AUC-MCC')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('AUC-MCC')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot Accuracy
    if 'acc' in history:
        axes[1, 0].plot(epochs, history['acc'], label='Train Acc', color='blue')
    if 'val_acc' in history:
        axes[1, 0].plot(epochs, history['val_acc'], label='Val Acc', color='orange')
    axes[1, 0].set_title('Accuracy')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot benchmark metrics if available
    if 'val_class_loss' in history:
        axes[1, 1].plot(epochs, history['val_class_loss'], label='Val CLS Loss', color='red')
    if 'val_pep_loss' in history:
        axes[1, 1].plot(epochs, history['val_pep_loss'], label='Val PEP Loss', color='green')
    if 'val_mhc_loss' in history:
        axes[1, 1].plot(epochs, history['val_mhc_loss'], label='Val MHC Loss', color='blue')
    if 'val_loss' in history:
        axes[1, 1].plot(epochs, history['val_loss'], label='Val Loss', color='orange')
        axes[1, 1].set_title('Validation Loss')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Validation Loss\nAvailable', ha='center', va='center',
                        transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Validation Loss (N/A)')

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history plot saved to {os.path.join(out_path, 'training_history.png')}")


# Define 21 distinct colors for amino acids (20 standard + 1 for unknown/special)
AMINO_ACID_COLORS = {
    'A': '#FF6B6B',  # Alanine - Red
    'C': '#4ECDC4',  # Cysteine - Teal
    'D': '#45B7D1',  # Aspartic acid - Blue
    'E': '#96CEB4',  # Glutamic acid - Green
    'F': '#FFEAA7',  # Phenylalanine - Yellow
    'G': '#DDA0DD',  # Glycine - Plum
    'H': '#98D8C8',  # Histidine - Mint
    'I': '#F7DC6F',  # Isoleucine - Light yellow
    'K': '#BB8FCE',  # Lysine - Light purple
    'L': '#85C1E9',  # Leucine - Light blue
    'M': '#F8C471',  # Methionine - Peach
    'N': '#82E0AA',  # Asparagine - Light green
    'P': '#F1948A',  # Proline - Light red
    'Q': '#85C1E9',  # Glutamine - Sky blue
    'R': '#D7BDE2',  # Arginine - Lavender
    'S': '#A9DFBF',  # Serine - Pale green
    'T': '#FAD7A0',  # Threonine - Beige
    'V': '#AED6F1',  # Valine - Pale blue
    'W': '#D5A6BD',  # Tryptophan - Pink
    'Y': '#F9E79F',  # Tyrosine - Pale yellow
    'X': '#808080',  # Unknown/Other - Gray
}


def get_anchor_pair_amino_acids(peptide_seq: str) -> tuple:
    """
    Gets the second and last amino acid from peptide sequence.

    Args:
        peptide_seq: Peptide sequence string

    Returns:
        tuple: (second_aa, last_aa) or ('X', 'X') for short sequences
    """
    # Clean sequence
    peptide_seq = peptide_seq.replace("-", "").replace("*", "").replace(" ", "").upper()

    if len(peptide_seq) < 2:
        return ('X', 'X')

    p2 = peptide_seq[1].upper()  # second amino acid
    p_omega = peptide_seq[-1].upper()  # last amino acid

    return (p2, p_omega)


def plot_anchor_pairs_with_amino_acid_colors(
        embedding: np.ndarray,
        peptide_sequences: pd.Series,
        title: str,
        filename: str,
        figsize: tuple = (20, 8),
        point_size: int = 30,
        alpha: float = 0.6
):
    """
    Plot UMAP embedding with side-by-side views of first and second anchor amino acids,
    plus a combined view using marker shapes.

    Args:
        embedding: 2D UMAP embedding coordinates
        peptide_sequences: Series of peptide sequences
        title: Plot title
        filename: Output filename
        figsize: Figure size
        point_size: Size of scatter points
        alpha: Point transparency
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Get anchor pairs for all peptides
    anchor_pairs = peptide_sequences.apply(get_anchor_pair_amino_acids)
    first_anchors = anchor_pairs.apply(lambda x: x[0])
    second_anchors = anchor_pairs.apply(lambda x: x[1])

    # Get unique amino acids
    unique_first = sorted(set(first_anchors.values))
    unique_second = sorted(set(second_anchors.values))

    # Define marker shapes for second anchor (for combined plot)
    marker_shapes = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '8', 'P', 'X']
    aa_to_marker = {aa: marker_shapes[i % len(marker_shapes)] for i, aa in enumerate(unique_second)}

    # --- Panel 1: First Anchor ---
    ax1 = axes[0]
    for aa in unique_first:
        mask = (first_anchors == aa)
        if not np.any(mask):
            continue
        color = AMINO_ACID_COLORS.get(aa, AMINO_ACID_COLORS['X'])
        ax1.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[color],
            s=point_size,
            alpha=alpha,
            label=aa,
            rasterized=True,
            edgecolors='black',
            linewidths=0.3
        )

    ax1.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax1.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax1.set_title('First Anchor (P2)', fontsize=14, fontweight='bold')
    ax1.legend(title='Amino Acid', bbox_to_anchor=(1.02, 1), loc='upper left',
               fontsize=9, ncol=1, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Second Anchor ---
    ax2 = axes[1]
    for aa in unique_second:
        mask = (second_anchors == aa)
        if not np.any(mask):
            continue
        color = AMINO_ACID_COLORS.get(aa, AMINO_ACID_COLORS['X'])
        ax2.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[color],
            s=point_size,
            alpha=alpha,
            label=aa,
            rasterized=True,
            edgecolors='black',
            linewidths=0.3
        )

    ax2.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax2.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax2.set_title('Second Anchor (C-term)', fontsize=14, fontweight='bold')
    ax2.legend(title='Amino Acid', bbox_to_anchor=(1.02, 1), loc='upper left',
               fontsize=9, ncol=1, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Combined View (color = 1st anchor, shape = 2nd anchor) ---
    ax3 = axes[2]
    for aa1 in unique_first:
        mask1 = (first_anchors == aa1)
        if not np.any(mask1):
            continue
        color = AMINO_ACID_COLORS.get(aa1, AMINO_ACID_COLORS['X'])

        for aa2 in unique_second:
            mask2 = (second_anchors == aa2)
            combined_mask = mask1 & mask2
            if not np.any(combined_mask):
                continue

            marker = aa_to_marker[aa2]
            ax3.scatter(
                embedding[combined_mask, 0],
                embedding[combined_mask, 1],
                c=[color],
                s=point_size * 1.2,
                alpha=alpha,
                marker=marker,
                rasterized=True,
                edgecolors='black',
                linewidths=0.5
            )

    ax3.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax3.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax3.set_title('Combined (Color=1st, Shape=2nd)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Create combined legend with both colors and shapes
    # Color legend (first anchor)
    color_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=AMINO_ACID_COLORS.get(aa, AMINO_ACID_COLORS['X']),
                   markersize=8, label=aa, markeredgecolor='black', markeredgewidth=0.5)
        for aa in unique_first
    ]
    first_legend = ax3.legend(
        handles=color_handles,
        title='1st Anchor (Color)',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=8,
        ncol=1,
        framealpha=0.9
    )
    ax3.add_artist(first_legend)

    # Shape legend (second anchor)
    shape_handles = [
        plt.Line2D([0], [0], marker=aa_to_marker[aa], color='w', markerfacecolor='gray',
                   markersize=8, label=aa, markeredgecolor='black', markeredgewidth=0.5)
        for aa in unique_second
    ]
    second_legend_y = max(0.1, 0.95 - len(unique_first) * 0.035)
    ax3.legend(
        handles=shape_handles,
        title='2nd Anchor (Shape)',
        bbox_to_anchor=(1.02, second_legend_y),
        loc='upper left',
        fontsize=8,
        ncol=1,
        framealpha=0.9
    )

    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    # Save plot
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Anchor pair plot saved to {filename}")


# Alternative: Heatmap-style visualization
def plot_anchor_pairs_heatmap(
        embedding: np.ndarray,
        peptide_sequences: pd.Series,
        title: str,
        filename: str,
        figsize: tuple = (14, 12),
        bins: int = 50
):
    """
    Alternative visualization: Show anchor pair combinations as a heatmap grid.

    Args:
        embedding: 2D UMAP embedding coordinates
        peptide_sequences: Series of peptide sequences
        title: Plot title
        filename: Output filename
        figsize: Figure size
        bins: Number of bins for 2D histogram
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    # Get anchor pairs
    anchor_pairs = peptide_sequences.apply(get_anchor_pair_amino_acids)
    first_anchors = anchor_pairs.apply(lambda x: x[0])
    second_anchors = anchor_pairs.apply(lambda x: x[1])

    # Get unique combinations
    unique_pairs = sorted(set(zip(first_anchors, second_anchors)))

    # Calculate grid dimensions
    n_pairs = len(unique_pairs)
    ncols = min(4, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, (aa1, aa2) in enumerate(unique_pairs):
        ax = axes[idx]
        mask = (first_anchors == aa1) & (second_anchors == aa2)

        if np.any(mask):
            color = AMINO_ACID_COLORS.get(aa1, AMINO_ACID_COLORS['X'])
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[color],
                s=20,
                alpha=0.6,
                rasterized=True,
                edgecolors='black',
                linewidths=0.3
            )
            ax.set_title(f'{aa1} → {aa2}\n(n={np.sum(mask)})', fontsize=10, fontweight='bold')
        else:
            ax.set_title(f'{aa1} → {aa2}\n(n=0)', fontsize=10, color='gray')

        ax.set_xlabel('UMAP 1', fontsize=8)
        ax.set_ylabel('UMAP 2', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename.replace('.png', '_grid.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Anchor pair grid plot saved to {filename.replace('.png', '_grid.png')}")

def plot_anchor_pairs_with_amino_acid_colors_extended(
        embedding: np.ndarray,
        peptide_sequences: pd.Series,
        title: str,
        filename: str,
        figsize: tuple = (20, 12),
        point_size: int = 30,
        alpha: float = 0.6,
        show_top_n_pairs: int = 9
):
    """
    Improved visualization showing anchor pairs with multiple views:
    1. Side-by-side UMAP plots for each anchor position
    2. Frequency heatmap of anchor pair combinations
    3. Small multiples for most common anchor pairs

    Args:
        embedding: 2D UMAP embedding coordinates
        peptide_sequences: Series of peptide sequences
        title: Plot title
        filename: Output filename
        figsize: Figure size
        point_size: Size of scatter points
        alpha: Point transparency
        show_top_n_pairs: Number of top anchor pairs to highlight in small multiples
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    # Get anchor pairs for all peptides
    anchor_pairs = peptide_sequences.apply(get_anchor_pair_amino_acids)
    first_anchors = anchor_pairs.apply(lambda x: x[0])
    second_anchors = anchor_pairs.apply(lambda x: x[1])

    # Create combined anchor pair labels
    pair_labels = first_anchors + '-' + second_anchors

    # Count anchor pair frequencies
    pair_counts = Counter(pair_labels)
    top_pairs = [pair for pair, _ in pair_counts.most_common(show_top_n_pairs)]

    # Set up the grid layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3,
                  height_ratios=[1.2, 1, 1, 1], width_ratios=[1, 1, 1, 1])

    # === Top Row: Side-by-side anchor position plots ===
    ax_first = fig.add_subplot(gs[0, 0:2])
    ax_second = fig.add_subplot(gs[0, 2:4])

    # Plot first anchor position
    unique_first = sorted(set(first_anchors.values))
    for aa in unique_first:
        mask = (first_anchors == aa)
        color = AMINO_ACID_COLORS.get(aa, AMINO_ACID_COLORS['X'])
        ax_first.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[color],
            s=point_size,
            alpha=alpha,
            label=aa,
            edgecolors='white',
            linewidth=0.5,
            rasterized=True
        )

    ax_first.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
    ax_first.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
    ax_first.set_title('First Anchor Position (P2)', fontsize=14, fontweight='bold', pad=10)
    ax_first.legend(loc='best', ncol=2, fontsize=9, framealpha=0.9)
    ax_first.grid(True, alpha=0.3)

    # Plot second anchor position
    unique_second = sorted(set(second_anchors.values))
    for aa in unique_second:
        mask = (second_anchors == aa)
        color = AMINO_ACID_COLORS.get(aa, AMINO_ACID_COLORS['X'])
        ax_second.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[color],
            s=point_size,
            alpha=alpha,
            label=aa,
            edgecolors='white',
            linewidth=0.5,
            rasterized=True
        )

    ax_second.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
    ax_second.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
    ax_second.set_title('Second Anchor Position (C-terminal)', fontsize=14, fontweight='bold', pad=10)
    ax_second.legend(loc='best', ncol=2, fontsize=9, framealpha=0.9)
    ax_second.grid(True, alpha=0.3)

    # === Middle Left: Anchor pair frequency heatmap ===
    ax_heatmap = fig.add_subplot(gs[1, 0:2])

    # Create frequency matrix
    unique_pairs_first = sorted(unique_first)
    unique_pairs_second = sorted(unique_second)
    freq_matrix = np.zeros((len(unique_pairs_first), len(unique_pairs_second)))

    for i, aa1 in enumerate(unique_pairs_first):
        for j, aa2 in enumerate(unique_pairs_second):
            pair_key = f"{aa1}-{aa2}"
            freq_matrix[i, j] = pair_counts.get(pair_key, 0)

    # Plot heatmap
    sns.heatmap(
        freq_matrix,
        xticklabels=unique_pairs_second,
        yticklabels=unique_pairs_first,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        ax=ax_heatmap,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='gray'
    )
    ax_heatmap.set_xlabel('Second Anchor (C-terminal)', fontsize=11, fontweight='bold')
    ax_heatmap.set_ylabel('First Anchor (P2)', fontsize=11, fontweight='bold')
    ax_heatmap.set_title('Anchor Pair Frequency Distribution', fontsize=12, fontweight='bold', pad=10)

    # === Middle Right: Top anchor pairs bar chart ===
    ax_bars = fig.add_subplot(gs[1, 2:4])

    top_pair_names = [pair for pair, _ in pair_counts.most_common(12)]
    top_pair_values = [count for _, count in pair_counts.most_common(12)]

    # Create color list for bars based on first anchor
    bar_colors = []
    for pair in top_pair_names:
        first_aa = pair.split('-')[0]
        bar_colors.append(AMINO_ACID_COLORS.get(first_aa, AMINO_ACID_COLORS['X']))

    bars = ax_bars.barh(range(len(top_pair_names)), top_pair_values, color=bar_colors, alpha=0.7, edgecolor='black')
    ax_bars.set_yticks(range(len(top_pair_names)))
    ax_bars.set_yticklabels(top_pair_names, fontsize=10)
    ax_bars.set_xlabel('Count', fontsize=11, fontweight='bold')
    ax_bars.set_title('Most Common Anchor Pairs', fontsize=12, fontweight='bold', pad=10)
    ax_bars.grid(True, alpha=0.3, axis='x')
    ax_bars.invert_yaxis()

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, top_pair_values)):
        ax_bars.text(count, i, f'  {count}', va='center', fontsize=9, fontweight='bold')

    # === Bottom Row: Small multiples for top anchor pairs ===
    n_cols = 4
    n_rows = 2

    for idx, pair in enumerate(top_pairs[:n_cols * n_rows]):
        row = idx // n_cols + 1
        col = idx % n_cols
        ax_small = fig.add_subplot(gs[row + 1, col])

        # Highlight this specific pair
        mask = (pair_labels == pair)

        # Plot all points in gray
        ax_small.scatter(
            embedding[~mask, 0],
            embedding[~mask, 1],
            c='lightgray',
            s=point_size * 0.3,
            alpha=0.2,
            rasterized=True
        )

        # Highlight the specific pair
        if np.any(mask):
            first_aa = pair.split('-')[0]
            color = AMINO_ACID_COLORS.get(first_aa, AMINO_ACID_COLORS['X'])
            ax_small.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[color],
                s=point_size * 1.2,
                alpha=0.8,
                edgecolors='black',
                linewidth=1,
                rasterized=True
            )

        count = pair_counts[pair]
        ax_small.set_title(f'{pair}\n(n={count})', fontsize=10, fontweight='bold')
        ax_small.set_xticks([])
        ax_small.set_yticks([])
        ax_small.grid(True, alpha=0.2)

        # Add border color based on first anchor
        first_aa = pair.split('-')[0]
        border_color = AMINO_ACID_COLORS.get(first_aa, AMINO_ACID_COLORS['X'])
        for spine in ax_small.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

    # Main title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

    # Save plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Improved anchor pair plot saved to {filename}")


def plot_anchor_pairs_interactive_style(
        embedding: np.ndarray,
        peptide_sequences: pd.Series,
        title: str,
        filename: str,
        figsize: tuple = (16, 10),
        point_size: int = 40,
        alpha: float = 0.7
):
    """
    Alternative visualization using bivariate color mapping where both anchors
    are encoded in a single color using RGB channels.

    Args:
        embedding: 2D UMAP embedding coordinates
        peptide_sequences: Series of peptide sequences
        title: Plot title
        filename: Output filename
        figsize: Figure size
        point_size: Size of scatter points
        alpha: Point transparency
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax_main, ax_legend) = plt.subplots(1, 2, figsize=figsize,
                                             gridspec_kw={'width_ratios': [3, 1]})

    # Get anchor pairs
    anchor_pairs = peptide_sequences.apply(get_anchor_pair_amino_acids)
    first_anchors = anchor_pairs.apply(lambda x: x[0])
    second_anchors = anchor_pairs.apply(lambda x: x[1])

    # Create bivariate colors
    colors = []
    for aa1, aa2 in zip(first_anchors, second_anchors):
        color1 = np.array(plt.matplotlib.colors.to_rgb(AMINO_ACID_COLORS.get(aa1, AMINO_ACID_COLORS['X'])))
        color2 = np.array(plt.matplotlib.colors.to_rgb(AMINO_ACID_COLORS.get(aa2, AMINO_ACID_COLORS['X'])))
        # Blend colors: 60% first anchor, 40% second anchor
        blended = 0.6 * color1 + 0.4 * color2
        colors.append(blended)

    # Main scatter plot
    ax_main.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=point_size,
        alpha=alpha,
        edgecolors='white',
        linewidth=0.5,
        rasterized=True
    )

    ax_main.set_xlabel('UMAP Dimension 1', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('UMAP Dimension 2', fontsize=14, fontweight='bold')
    ax_main.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax_main.grid(True, alpha=0.3)

    # Create legend showing color combinations
    unique_pairs = sorted(set(zip(first_anchors, second_anchors)))
    pair_counts = Counter(zip(first_anchors, second_anchors))

    # Sort by frequency
    unique_pairs = sorted(unique_pairs, key=lambda x: pair_counts[x], reverse=True)[:15]

    ax_legend.axis('off')
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, len(unique_pairs) + 1)

    y_pos = len(unique_pairs)
    for aa1, aa2 in unique_pairs:
        color1 = np.array(plt.matplotlib.colors.to_rgb(AMINO_ACID_COLORS.get(aa1, AMINO_ACID_COLORS['X'])))
        color2 = np.array(plt.matplotlib.colors.to_rgb(AMINO_ACID_COLORS.get(aa2, AMINO_ACID_COLORS['X'])))
        blended = 0.6 * color1 + 0.4 * color2

        # Draw color box
        rect = Rectangle((0.05, y_pos - 0.4), 0.15, 0.8,
                         facecolor=blended, edgecolor='black', linewidth=1)
        ax_legend.add_patch(rect)

        # Add label
        count = pair_counts[(aa1, aa2)]
        ax_legend.text(0.25, y_pos, f'{aa1}-{aa2}  (n={count})',
                       va='center', fontsize=10, fontweight='bold')

        y_pos -= 1

    ax_legend.text(0.5, len(unique_pairs) + 0.5, 'Top Anchor Pairs\n(60% P2 + 40% C-term)',
                   ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Bivariate color anchor plot saved to {filename}")


def plot_anchor_pairs_combined(
        embedding: np.ndarray,
        peptide_sequences: pd.Series,
        title_base: str,
        filename_base: str,
        **kwargs
):
    """
    Create both visualization styles.

    Args:
        embedding: 2D UMAP embedding coordinates
        peptide_sequences: Series of peptide sequences
        title_base: Base title for plots
        filename_base: Base filename (without extension)
        **kwargs: Additional arguments passed to plotting functions
    """
    # Main improved plot
    plot_anchor_pairs_with_amino_acid_colors_extended(
        embedding=embedding,
        peptide_sequences=peptide_sequences,
        title=f"{title_base} - Comprehensive Anchor Analysis",
        filename=f"{filename_base}_comprehensive.png",
        **kwargs
    )

    # Bivariate color plot
    plot_anchor_pairs_interactive_style(
        embedding=embedding,
        peptide_sequences=peptide_sequences,
        title=f"{title_base} - Bivariate Color Encoding",
        filename=f"{filename_base}_bivariate.png",
        **{k: v for k, v in kwargs.items() if k in ['figsize', 'point_size', 'alpha']}
    )

    plot_anchor_pairs_heatmap(
        embedding=embedding,
        peptide_sequences=peptide_sequences,
        title=f"{title_base} - Anchor Pair Frequency Heatmap",
        filename=f"{filename_base}_heatmap.png",
        **{k: v for k, v in kwargs.items() if k in ['figsize', 'bins']}
    )


def visualize_inference_results(df_processed, true_labels, predictions, out_dir, dataset_name):
    """
    Visualize inference results: ROC curve, Precision-Recall curve, and Confusion Matrix.

    Parameters:
        df_processed: DataFrame with processed data
        true_labels: True labels
        predictions: Predicted probabilities
        out_dir: Output directory to save plots
        dataset_name: Name of the dataset
    """
    os.makedirs(out_dir, exist_ok=True)

    # ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f'roc_curve_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(true_labels, predictions)
    pr_auc = average_precision_score(true_labels, predictions)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_name}')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f'pr_curve_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Confusion Matrix
    pred_labels = (predictions >= 0.5).astype(int)
    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.savefig(os.path.join(out_dir, f'confusion_matrix_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Inference visualizations saved to {out_dir}")


def visualize_per_allele_metrics(df, true_labels, predictions, out_dir, dataset_name,
                                 allele_col='allele', min_samples=10, top_n=None, threshold=0.5):
    """
    Compute and visualize per-allele performance metrics (AUC, Accuracy, Precision, Recall).

    Parameters:
        df: DataFrame containing allele information
        true_labels: True binary labels (array-like)
        predictions: Predicted probabilities (array-like)
        out_dir: Output directory to save plots
        dataset_name: Name of the dataset
        allele_col: Column name containing allele information
        min_samples: Minimum number of samples required per allele
        top_n: If specified, only show top N alleles by sample count
        threshold: Classification threshold for converting probabilities to labels
    """
    os.makedirs(out_dir, exist_ok=True)

    # Add predictions and true labels to dataframe
    df_copy = df.copy()
    df_copy['true_label'] = true_labels
    df_copy['prediction_prob'] = predictions
    df_copy['prediction_label'] = (predictions >= threshold).astype(int)

    # Compute metrics per allele
    allele_metrics = []
    for allele in df_copy[allele_col].unique():
        allele_mask = df_copy[allele_col] == allele
        allele_true = df_copy.loc[allele_mask, 'true_label']
        allele_pred_prob = df_copy.loc[allele_mask, 'prediction_prob']
        allele_pred_label = df_copy.loc[allele_mask, 'prediction_label']

        n_samples = len(allele_true)

        # Skip alleles with insufficient samples
        if n_samples < min_samples:
            continue

        # Compute confusion matrix elements
        tp = int(np.sum((allele_pred_label == 1) & (allele_true == 1)))
        tn = int(np.sum((allele_pred_label == 0) & (allele_true == 0)))
        fp = int(np.sum((allele_pred_label == 1) & (allele_true == 0)))
        fn = int(np.sum((allele_pred_label == 0) & (allele_true == 1)))

        # Compute metrics
        eps = 1e-12
        accuracy = (tp + tn) / max(n_samples, eps)
        precision = tp / max(tp + fp, eps)
        recall = tp / max(tp + fn, eps)

        # Compute AUC if both classes present
        auc_score = np.nan
        if len(allele_true.unique()) >= 2:
            try:
                fpr, tpr, _ = roc_curve(allele_true, allele_pred_prob)
                auc_score = auc(fpr, tpr)
            except Exception as e:
                print(f"Warning: Could not compute AUC for allele {allele}: {e}")

        allele_metrics.append({
            'allele': allele,
            'n_samples': n_samples,
            'n_positive': int(allele_true.sum()),
            'n_negative': int((1 - allele_true).sum()),
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'AUC': auc_score,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall
        })

    # Create DataFrame and sort by sample count
    metrics_df = pd.DataFrame(allele_metrics)
    if len(metrics_df) == 0:
        print(f"Warning: No alleles with sufficient samples to compute metrics")
        return

    metrics_df = metrics_df.sort_values('n_samples', ascending=False)

    # Filter to top N if specified
    if top_n is not None:
        metrics_df = metrics_df.head(top_n)

    # Plot 1: Heatmap of all metrics per allele
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(8, len(metrics_df) * 0.3)))

    # Confusion matrix counts
    confusion_data = metrics_df[['TP', 'TN', 'FP', 'FN']].set_index(metrics_df['allele'])
    sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title(f'Confusion Matrix Counts per Allele - {dataset_name}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Allele', fontsize=11)
    ax1.set_xlabel('Metric', fontsize=11)

    # Performance metrics
    perf_data = metrics_df[['AUC', 'Accuracy', 'Precision', 'Recall']].set_index(metrics_df['allele'])
    sns.heatmap(perf_data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1, ax=ax2, cbar_kws={'label': 'Score'})
    ax2.set_title(f'Performance Metrics per Allele - {dataset_name}', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Allele', fontsize=11)
    ax2.set_xlabel('Metric', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'per_allele_heatmap_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Grouped bar chart of metrics
    metrics_to_plot = ['AUC', 'Accuracy', 'Precision', 'Recall']
    x = np.arange(len(metrics_df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(14, len(metrics_df) * 0.5), 8))

    for i, metric in enumerate(metrics_to_plot):
        offset = width * (i - 1.5)
        values = metrics_df[metric].fillna(0)
        ax.bar(x + offset, values, width, label=metric, alpha=0.8)

    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random (0.5)')
    ax.set_xlabel('Allele', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(
        f'Per-Allele Performance Metrics - {dataset_name}\n({len(metrics_df)} alleles with ≥{min_samples} samples)',
        fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['allele'], rotation=90, ha='right')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'per_allele_metrics_bars_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Scatter plots of metrics vs sample count
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = metrics_df[metric].fillna(0)

        scatter = ax.scatter(metrics_df['n_samples'], values,
                             c=values, cmap='RdYlGn', vmin=0, vmax=1,
                             s=100, alpha=0.6, edgecolors='black', linewidth=1)

        # Annotate outliers
        for _, row in metrics_df.iterrows():
            val = row[metric]
            if pd.notna(val) and (val > 0.9 or val < 0.6):
                ax.annotate(row['allele'], (row['n_samples'], val),
                            fontsize=7, alpha=0.7, xytext=(5, 5),
                            textcoords='offset points')

        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        mean_val = values.mean()
        ax.axhline(y=mean_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.5,
                   label=f'Mean={mean_val:.3f}')

        ax.set_xlabel('Number of Samples', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} vs Sample Count', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax, label=metric)

    plt.suptitle(f'Per-Allele Metrics vs Sample Count - {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'per_allele_metrics_vs_samples_{dataset_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: Distribution of each metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = metrics_df[metric].dropna()

        if len(values) > 0:
            ax.hist(values, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
            ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Random (0.5)')
            ax.axvline(x=values.mean(), color='blue', linestyle='--', linewidth=2,
                       label=f'Mean={values.mean():.3f}')
            ax.axvline(x=values.median(), color='green', linestyle='--', linewidth=2,
                       label=f'Median={values.median():.3f}')

            ax.set_xlabel(metric, fontsize=11)
            ax.set_ylabel('Number of Alleles', fontsize=11)
            ax.set_title(f'Distribution of {metric}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Distribution of Per-Allele Metrics - {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'per_allele_metrics_distribution_{dataset_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(out_dir, f'per_allele_metrics_{dataset_name}.csv'), index=False)

    # Print summary statistics
    print(f"\n✓ Per-allele metrics visualizations saved to {out_dir}")
    print(f"  - {len(metrics_df)} alleles analyzed")
    print(f"\nMetric Summaries:")
    for metric in metrics_to_plot:
        values = metrics_df[metric].dropna()
        if len(values) > 0:
            print(f"  {metric}:")
            print(f"    Mean:   {values.mean():.3f} ± {values.std():.3f}")
            print(f"    Median: {values.median():.3f}")
            print(f"    Min:    {values.min():.3f}")
            print(f"    Max:    {values.max():.3f}")

    # Find best and worst alleles by AUC
    if not metrics_df['AUC'].isna().all():
        best_idx = metrics_df['AUC'].idxmax()
        worst_idx = metrics_df['AUC'].idxmin()
        print(
            f"\n  Best allele (AUC):  {metrics_df.loc[best_idx, 'allele']} (AUC={metrics_df.loc[best_idx, 'AUC']:.3f})")
        print(
            f"  Worst allele (AUC): {metrics_df.loc[worst_idx, 'allele']} (AUC={metrics_df.loc[worst_idx, 'AUC']:.3f})")


def visualize_anchor_positions_and_binding_pockets(attn_weights, peptide_seq, mhc_seq,
                                                     max_pep_len, max_mhc_len, out_dir,
                                                     sample_idx=0, pooling='max',
                                                     top_k_anchors=4, top_k_pockets=10):
    """
    Use max pooling on attention weights to identify:
    1. Peptide anchor positions (positions with high MHC->peptide attention)
    2. MHC binding pocket regions (positions with high peptide->MHC attention)

    Args:
        attn_weights: Tensor of shape (B, heads, P+M, P+M) with attention scores.
        peptide_seq: Peptide sequence string.
        mhc_seq: MHC sequence string.
        max_pep_len: Maximum peptide length.
        max_mhc_len: Maximum MHC length.
        out_dir: Output directory to save plots.
        sample_idx: Index of sample to analyze (default: 0).
        pooling: Pooling method ('max', 'mean', or 'both').
        top_k_anchors: Number of top anchor positions to highlight.
        top_k_pockets: Number of top binding pocket positions to highlight.

    Returns:
        dict: Contains anchor positions and binding pocket positions with scores.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Convert to numpy
    try:
        attn_weights = attn_weights.numpy()
    except:
        pass

    # Get attention weights for the specified sample
    sample_attn = attn_weights[sample_idx]  # Shape: (heads, P+M, P+M)

    # Get actual sequence lengths
    pep_len = min(len(peptide_seq), max_pep_len)
    mhc_len = min(len(mhc_seq), max_mhc_len)

    # Extract only the relevant parts (no padding)
    relevant_attn = sample_attn[:, :pep_len+mhc_len, :pep_len+mhc_len]

    # Extract cross-attention matrices
    # MHC attending to peptide: rows are MHC positions, columns are peptide positions
    mhc_to_pep = relevant_attn[:, pep_len:pep_len+mhc_len, :pep_len]  # (heads, mhc_len, pep_len)

    # Peptide attending to MHC: rows are peptide positions, columns are MHC positions
    pep_to_mhc = relevant_attn[:, :pep_len, pep_len:pep_len+mhc_len]  # (heads, pep_len, mhc_len)

    # === PEPTIDE ANCHOR POSITIONS ===
    # For each peptide position, aggregate attention from all MHC positions across heads
    if pooling in ['max', 'both']:
        # Max pooling across MHC positions, then across heads
        anchor_scores_max = np.max(np.max(mhc_to_pep, axis=1), axis=0)  # (pep_len,)

    if pooling in ['mean', 'both']:
        # Mean pooling across MHC positions, then across heads
        anchor_scores_mean = np.mean(np.mean(mhc_to_pep, axis=1), axis=0)  # (pep_len,)

    # Use the appropriate scores
    if pooling == 'max':
        anchor_scores = anchor_scores_max
    elif pooling == 'mean':
        anchor_scores = anchor_scores_mean
    else:  # both - use max for primary analysis
        anchor_scores = anchor_scores_max

    # Get top anchor positions
    top_anchor_indices = np.argsort(anchor_scores)[-top_k_anchors:][::-1]
    anchor_results = [
        {'position': int(idx), 'amino_acid': peptide_seq[idx],
         'score': float(anchor_scores[idx]), 'position_1indexed': int(idx + 1)}
        for idx in top_anchor_indices
    ]

    # === MHC BINDING POCKET POSITIONS ===
    # For each MHC position, aggregate attention from all peptide positions across heads
    if pooling in ['max', 'both']:
        pocket_scores_max = np.max(np.max(pep_to_mhc, axis=1), axis=0)  # (mhc_len,)

    if pooling in ['mean', 'both']:
        pocket_scores_mean = np.mean(np.mean(pep_to_mhc, axis=1), axis=0)  # (mhc_len,)

    # Use the appropriate scores
    if pooling == 'max':
        pocket_scores = pocket_scores_max
    elif pooling == 'mean':
        pocket_scores = pocket_scores_mean
    else:  # both
        pocket_scores = pocket_scores_max

    # Get top binding pocket positions
    top_pocket_indices = np.argsort(pocket_scores)[-top_k_pockets:][::-1]
    pocket_results = [
        {'position': int(idx), 'amino_acid': mhc_seq[idx],
         'score': float(pocket_scores[idx]), 'position_1indexed': int(idx + 1)}
        for idx in top_pocket_indices
    ]

    # === VISUALIZATION 1: Peptide Anchor Positions Bar Chart ===
    fig, ax = plt.subplots(figsize=(max(10, pep_len * 0.5), 6))
    positions = np.arange(pep_len)
    colors = ['red' if i in top_anchor_indices else 'skyblue' for i in range(pep_len)]

    bars = ax.bar(positions, anchor_scores, color=colors, edgecolor='black', linewidth=1)

    # Highlight top anchors
    for idx in top_anchor_indices:
        ax.text(idx, anchor_scores[idx], f'{peptide_seq[idx]}\n{anchor_scores[idx]:.3f}',
               ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_xlabel('Peptide Position', fontsize=12)
    ax.set_ylabel(f'Attention Score ({pooling} pooling)', fontsize=12)
    ax.set_title(f'Peptide Anchor Positions (Sample {sample_idx})\nTop {top_k_anchors} highlighted in red',
                fontsize=14, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{i+1}\n{peptide_seq[i]}' for i in range(pep_len)], fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'peptide_anchor_positions_sample_{sample_idx}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # === VISUALIZATION 2: MHC Binding Pocket Positions Bar Chart ===
    fig, ax = plt.subplots(figsize=(max(14, mhc_len * 0.3), 6))
    positions = np.arange(mhc_len)
    colors = ['red' if i in top_pocket_indices else 'lightcoral' for i in range(mhc_len)]

    bars = ax.bar(positions, pocket_scores, color=colors, edgecolor='black', linewidth=0.5)

    # Highlight top pockets
    for idx in top_pocket_indices:
        ax.text(idx, pocket_scores[idx], f'{pocket_scores[idx]:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=8, rotation=0)

    ax.set_xlabel('MHC Position', fontsize=12)
    ax.set_ylabel(f'Attention Score ({pooling} pooling)', fontsize=12)
    ax.set_title(f'MHC Binding Pocket Regions (Sample {sample_idx})\nTop {top_k_pockets} highlighted in red',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{i + 1}\n{mhc_seq[i]}' for i in range(mhc_len)], fontsize=7, rotation=90)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'mhc_binding_pockets_sample_{sample_idx}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # === VISUALIZATION 3: Combined heatmap with both pooling methods (if both) ===
    if pooling == 'both':
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))

        # Peptide anchors - max pooling
        axes[0, 0].bar(range(pep_len), anchor_scores_max,
                      color=['red' if i in top_anchor_indices else 'skyblue' for i in range(pep_len)])
        axes[0, 0].set_title('Peptide Anchors (Max Pooling)', fontweight='bold')
        axes[0, 0].set_xlabel('Peptide Position')
        axes[0, 0].set_ylabel('Max Attention Score')
        axes[0, 0].set_xticks(range(pep_len))
        axes[0, 0].set_xticklabels([f'{i+1}\n{peptide_seq[i]}' for i in range(pep_len)], fontsize=8)
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Peptide anchors - mean pooling
        top_anchor_indices_mean = np.argsort(anchor_scores_mean)[-top_k_anchors:][::-1]
        axes[0, 1].bar(range(pep_len), anchor_scores_mean,
                      color=['orange' if i in top_anchor_indices_mean else 'lightblue' for i in range(pep_len)])
        axes[0, 1].set_title('Peptide Anchors (Mean Pooling)', fontweight='bold')
        axes[0, 1].set_xlabel('Peptide Position')
        axes[0, 1].set_ylabel('Mean Attention Score')
        axes[0, 1].set_xticks(range(pep_len))
        axes[0, 1].set_xticklabels([f'{i+1}\n{peptide_seq[i]}' for i in range(pep_len)], fontsize=8)
        axes[0, 1].grid(axis='y', alpha=0.3)

        # MHC pockets - max pooling
        axes[1, 0].bar(range(mhc_len), pocket_scores_max,
                      color=['red' if i in top_pocket_indices else 'lightcoral' for i in range(mhc_len)])
        axes[1, 0].set_title('MHC Binding Pockets (Max Pooling)', fontweight='bold')
        axes[1, 0].set_xlabel('MHC Position')
        axes[1, 0].set_ylabel('Max Attention Score')
        axes[1, 0].set_xticks(range(0, mhc_len, 5))
        axes[1, 0].set_xticklabels([f'{i+1}' for i in range(0, mhc_len, 5)], fontsize=8)
        axes[1, 0].grid(axis='y', alpha=0.3)

        # MHC pockets - mean pooling
        top_pocket_indices_mean = np.argsort(pocket_scores_mean)[-top_k_pockets:][::-1]
        axes[1, 1].bar(range(mhc_len), pocket_scores_mean,
                      color=['orange' if i in top_pocket_indices_mean else 'peachpuff' for i in range(mhc_len)])
        axes[1, 1].set_title('MHC Binding Pockets (Mean Pooling)', fontweight='bold')
        axes[1, 1].set_xlabel('MHC Position')
        axes[1, 1].set_ylabel('Mean Attention Score')
        axes[1, 1].set_xticks(range(0, mhc_len, 5))
        axes[1, 1].set_xticklabels([f'{i+1}' for i in range(0, mhc_len, 5)], fontsize=8)
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.suptitle(f'Anchor & Binding Pocket Analysis - Max vs Mean Pooling (Sample {sample_idx})',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'anchor_pocket_comparison_sample_{sample_idx}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    # === VISUALIZATION 4: 2D Heatmaps of Both Cross-Attention Directions ===
    # Average attention across heads for clearer visualization
    avg_pep_to_mhc = np.mean(pep_to_mhc, axis=0)  # (pep_len, mhc_len)
    avg_mhc_to_pep = np.mean(mhc_to_pep, axis=0)  # (mhc_len, pep_len)

    # Heatmap 1: Peptide -> MHC attention (horizontal: more columns than rows)
    fig1, ax1 = plt.subplots(figsize=(max(16, mhc_len * 0.5), max(6, pep_len * 0.5)))

    im1 = ax1.imshow(avg_pep_to_mhc, aspect='auto', cmap='viridis', interpolation='nearest')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Attention Score', fontsize=12)

    # Mark top peptide anchor positions with red lines
    for idx in top_anchor_indices:
        ax1.axhline(y=idx, color='red', linewidth=2, linestyle='--', alpha=0.7)

    # Mark top MHC binding pocket positions with yellow lines
    for idx in top_pocket_indices:
        ax1.axvline(x=idx, color='yellow', linewidth=2, linestyle='--', alpha=0.7)

    ax1.set_xlabel('MHC Position', fontsize=12)
    ax1.set_ylabel('Peptide Position', fontsize=12)
    ax1.set_title(
        f'Peptide→MHC Attention\nRed: Top {top_k_anchors} peptide anchors | Yellow: Top {top_k_pockets} MHC pockets',
        fontsize=12, fontweight='bold')
    ax1.set_xticks(range(mhc_len))
    ax1.set_yticks(range(pep_len))
    ax1.set_xticklabels([f'{i + 1}\n{mhc_seq[i]}' for i in range(mhc_len)], fontsize=7, rotation=90)
    ax1.set_yticklabels([f'P{i + 1}:{peptide_seq[i]}' for i in range(pep_len)], fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'peptide_to_mhc_attention_sample_{sample_idx}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # Heatmap 2: MHC -> Peptide attention (vertical: more rows than columns)
    fig2, ax2 = plt.subplots(figsize=(max(6, pep_len * 0.5), max(16, mhc_len * 0.5)))

    im2 = ax2.imshow(avg_mhc_to_pep, aspect='auto', cmap='plasma', interpolation='nearest')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Attention Score', fontsize=12)

    # Mark top MHC binding pocket positions with yellow lines
    for idx in top_pocket_indices:
        ax2.axhline(y=idx, color='yellow', linewidth=2, linestyle='--', alpha=0.7)

    # Mark top peptide anchor positions with red lines
    for idx in top_anchor_indices:
        ax2.axvline(x=idx, color='red', linewidth=2, linestyle='--', alpha=0.7)

    ax2.set_xlabel('Peptide Position', fontsize=12)
    ax2.set_ylabel('MHC Position', fontsize=12)
    ax2.set_title(
        f'MHC→Peptide Attention\nYellow: Top {top_k_pockets} MHC pockets | Red: Top {top_k_anchors} peptide anchors',
        fontsize=12, fontweight='bold')
    ax2.set_xticks(range(pep_len))
    ax2.set_yticks(range(mhc_len))
    ax2.set_xticklabels([f'P{i + 1}\n{peptide_seq[i]}' for i in range(pep_len)], fontsize=9, rotation=90)
    ax2.set_yticklabels([f'{i + 1}:{mhc_seq[i]}' for i in range(mhc_len)], fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'mhc_to_peptide_attention_sample_{sample_idx}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # Print summary
    print(f"\n✓ Anchor and binding pocket analysis complete for sample {sample_idx}")
    print(f"\nTop {top_k_anchors} Peptide Anchor Positions:")
    for result in anchor_results:
        print(f"  Position {result['position_1indexed']}: {result['amino_acid']} (score: {result['score']:.4f})")

    print(f"\nTop {top_k_pockets} MHC Binding Pocket Positions:")
    for result in pocket_results:
        print(f"  Position {result['position_1indexed']}: {result['amino_acid']} (score: {result['score']:.4f})")

    return {
        'anchor_positions': anchor_results,
        'binding_pockets': pocket_results,
        'anchor_scores_all': anchor_scores.tolist(),
        'pocket_scores_all': pocket_scores.tolist()
    }


# ========================================================================
# PERMUTATION AND ABLATION STUDY VISUALIZATIONS
# ========================================================================

def visualize_permutation_importance(pep_results, mhc_results, pos_results, blosum_results,
                                     baseline_score, out_dir, metric='auc'):
    """
    Visualize permutation importance results.

    Args:
        pep_results: DataFrame with peptide permutation results
        mhc_results: DataFrame with MHC permutation results
        pos_results: DataFrame with per-position permutation results
        blosum_results: DataFrame with BLOSUM feature permutation results
        baseline_score: Baseline performance score
        out_dir: Output directory for plots
        metric: Performance metric name
    """
    os.makedirs(out_dir, exist_ok=True)

    # Create comprehensive permutation importance plot
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Peptide and MHC permutation comparison
    ax1 = fig.add_subplot(gs[0, :2])

    # Combine peptide and MHC results
    combined_data = []
    for _, row in pep_results.iterrows():
        combined_data.append({'component': 'Peptide', 'importance': row['importance']})
    for _, row in mhc_results.iterrows():
        combined_data.append({'component': 'MHC', 'importance': row['importance']})

    combined_df = pd.DataFrame(combined_data)

    sns.boxplot(data=combined_df, x='component', y='importance', ax=ax1, palette='Set2')
    sns.stripplot(data=combined_df, x='component', y='importance', ax=ax1,
                 color='black', alpha=0.5, size=4)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No importance')
    ax1.set_title(f'Permutation Importance: Peptide vs MHC\n(Baseline {metric.upper()}: {baseline_score:.4f})',
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel(f'{metric.upper()} Drop', fontsize=12)
    ax1.set_xlabel('Component', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Summary statistics table
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')

    summary_data = [
        ['Component', 'Mean Importance', 'Std'],
        ['Peptide', f"{pep_results['importance'].mean():.4f}", f"{pep_results['importance'].std():.4f}"],
        ['MHC', f"{mhc_results['importance'].mean():.4f}", f"{mhc_results['importance'].std():.4f}"],
    ]

    table = ax2.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax2.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    # 3. Position importance bar plot
    ax3 = fig.add_subplot(gs[1, :])

    positions = pos_results['position'].values
    importances = pos_results['importance'].values
    errors = pos_results.get('std_score', np.zeros_like(importances))

    colors = ['#ff6b6b' if imp > importances.mean() else '#4ecdc4' for imp in importances]
    bars = ax3.bar(positions, importances, yerr=errors, capsize=5, alpha=0.7,
                  color=colors, edgecolor='black', linewidth=1.5)

    # Highlight anchor positions (P2 = position 1, P-omega = last position)
    if len(positions) > 0:
        ax3.axvline(x=1, color='red', linestyle='--', alpha=0.7, linewidth=2, label='P2 anchor')
        ax3.axvline(x=positions[-1], color='orange', linestyle='--', alpha=0.7,
                   linewidth=2, label='P-omega anchor')

    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.axhline(y=importances.mean(), color='green', linestyle=':', alpha=0.7,
               label=f'Mean importance: {importances.mean():.4f}')

    ax3.set_xlabel('Peptide Position', fontsize=12, fontweight='bold')
    ax3.set_ylabel(f'{metric.upper()} Drop', fontsize=12, fontweight='bold')
    ax3.set_title('Per-Position Permutation Importance', fontsize=14, fontweight='bold')
    ax3.set_xticks(positions)
    ax3.set_xticklabels([f'P{i+1}' for i in positions])
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. BLOSUM feature importance
    ax4 = fig.add_subplot(gs[2, :2])

    blosum_importances = blosum_results['importance'].values
    blosum_features = blosum_results['feature_idx'].values
    blosum_errors = blosum_results.get('std_score', np.zeros_like(blosum_importances))

    # Sort by importance
    sorted_indices = np.argsort(blosum_importances)[::-1]
    top_n = 15
    top_indices = sorted_indices[:top_n]

    y_pos = np.arange(top_n)
    ax4.barh(y_pos, blosum_importances[top_indices],
            xerr=blosum_errors[top_indices], capsize=3,
            color=plt.cm.viridis(blosum_importances[top_indices] / blosum_importances.max()),
            edgecolor='black', linewidth=1)

    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([f'Feature {blosum_features[i]}' for i in top_indices])
    ax4.invert_yaxis()
    ax4.set_xlabel(f'{metric.upper()} Drop', fontsize=12, fontweight='bold')
    ax4.set_title(f'Top {top_n} BLOSUM62 Feature Importances', fontsize=14, fontweight='bold')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax4.grid(True, alpha=0.3, axis='x')

    # 5. Distribution of all importances
    ax5 = fig.add_subplot(gs[2, 2])

    all_importances = np.concatenate([
        pep_results['importance'].values,
        mhc_results['importance'].values,
        pos_results['importance'].values,
        blosum_results['importance'].values
    ])

    ax5.hist(all_importances, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No importance')
    ax5.axvline(x=all_importances.mean(), color='green', linestyle=':',
               linewidth=2, label=f'Mean: {all_importances.mean():.4f}')
    ax5.set_xlabel(f'{metric.upper()} Drop', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax5.set_title('Overall Importance Distribution', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    plt.suptitle('Permutation Feature Importance Analysis', fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(os.path.join(out_dir, 'permutation_importance_comprehensive.png'),
               dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Permutation importance visualization saved to {out_dir}")


def visualize_ablation_results(input_results, pos_results, anchor_results,
                               baseline_score, out_dir, metric='auc'):
    """
    Visualize ablation study results.

    Args:
        input_results: DataFrame with input ablation results (peptide/MHC)
        pos_results: DataFrame with per-position ablation results
        anchor_results: DataFrame with anchor position ablation results
        baseline_score: Baseline performance score
        out_dir: Output directory for plots
        metric: Performance metric name
    """
    os.makedirs(out_dir, exist_ok=True)

    # Create comprehensive ablation results plot
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Input component ablation
    ax1 = fig.add_subplot(gs[0, 0])

    components = input_results['component'].values
    importances = input_results['importance'].values

    colors = ['#e74c3c', '#3498db']
    bars = ax1.bar(range(len(components)), importances, color=colors,
                  edgecolor='black', linewidth=2, alpha=0.8)

    ax1.set_xticks(range(len(components)))
    ax1.set_xticklabels(['Peptide\nInput', 'MHC\nInput'], fontsize=11)
    ax1.set_ylabel(f'{metric.upper()} Drop', fontsize=12, fontweight='bold')
    ax1.set_title('Input Component Ablation', fontsize=13, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, importances)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom' if val > 0 else 'top',
                fontweight='bold', fontsize=10)

    # 2. Anchor position ablation
    ax2 = fig.add_subplot(gs[0, 1])

    anchors = anchor_results['anchor'].values
    anchor_importances = anchor_results['importance'].values

    colors_anchor = ['#f39c12', '#e67e22', '#c0392b']
    bars = ax2.bar(range(len(anchors)), anchor_importances, color=colors_anchor[:len(anchors)],
                  edgecolor='black', linewidth=2, alpha=0.8)

    ax2.set_xticks(range(len(anchors)))
    ax2.set_xticklabels(anchors, fontsize=11)
    ax2.set_ylabel(f'{metric.upper()} Drop', fontsize=12, fontweight='bold')
    ax2.set_title('Anchor Position Ablation', fontsize=13, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, anchor_importances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom' if val > 0 else 'top',
                fontweight='bold', fontsize=10)

    # 3. Summary comparison
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')

    summary_data = [
        ['Component', f'{metric.upper()} Drop', 'Score'],
        ['Baseline', '0.0000', f'{baseline_score:.4f}'],
        ['Peptide Ablated', f"{input_results[input_results['component']=='peptide_input']['importance'].values[0]:.4f}",
         f"{input_results[input_results['component']=='peptide_input']['score'].values[0]:.4f}"],
        ['MHC Ablated', f"{input_results[input_results['component']=='mhc_input']['importance'].values[0]:.4f}",
         f"{input_results[input_results['component']=='mhc_input']['score'].values[0]:.4f}"],
    ]

    table = ax3.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax3.set_title('Ablation Summary', fontsize=12, fontweight='bold', pad=20)

    # 4. Per-position ablation (full plot)
    ax4 = fig.add_subplot(gs[1, :])

    positions = pos_results['position'].values
    pos_importances = pos_results['importance'].values

    # Color based on importance magnitude
    norm = plt.Normalize(vmin=pos_importances.min(), vmax=pos_importances.max())
    colors_pos = plt.cm.RdYlGn_r(norm(pos_importances))

    bars = ax4.bar(positions, pos_importances, color=colors_pos,
                  edgecolor='black', linewidth=1.5, alpha=0.85)

    # Highlight key positions
    if len(positions) > 0:
        ax4.axvline(x=1, color='red', linestyle='--', alpha=0.7, linewidth=2.5, label='P2 anchor')
        if len(positions) > 8:  # Only show P-omega if we have enough positions
            ax4.axvline(x=positions[-1], color='orange', linestyle='--', alpha=0.7,
                       linewidth=2.5, label='P-omega anchor')

    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.axhline(y=pos_importances.mean(), color='blue', linestyle=':', alpha=0.7,
               linewidth=2, label=f'Mean: {pos_importances.mean():.4f}')

    ax4.set_xlabel('Peptide Position', fontsize=13, fontweight='bold')
    ax4.set_ylabel(f'{metric.upper()} Drop', fontsize=13, fontweight='bold')
    ax4.set_title('Per-Position Ablation Importance', fontsize=14, fontweight='bold')
    ax4.set_xticks(positions)
    ax4.set_xticklabels([f'P{i+1}' for i in positions])
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add colorbar for position importance
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax4, orientation='vertical', pad=0.01, aspect=30)
    cbar.set_label('Importance Magnitude', fontsize=10, fontweight='bold')

    plt.suptitle('Ablation Study Results', fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(os.path.join(out_dir, 'ablation_results_comprehensive.png'),
               dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Ablation results visualization saved to {out_dir}")


def visualize_position_importance(pos_results, baseline_score, max_pep_len,
                                  out_dir, study_type='permutation', metric='auc'):
    """
    Create detailed position-specific importance visualization.

    Args:
        pos_results: DataFrame with per-position results
        baseline_score: Baseline performance score
        max_pep_len: Maximum peptide length
        out_dir: Output directory
        study_type: 'permutation' or 'ablation'
        metric: Performance metric name
    """
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    positions = pos_results['position'].values
    importances = pos_results['importance'].values

    # 1. Bar plot with error bars
    ax1 = axes[0, 0]
    if 'std_score' in pos_results.columns:
        errors = pos_results['std_score'].values
    else:
        errors = np.zeros_like(importances)

    colors = plt.cm.viridis(importances / (importances.max() + 1e-8))
    bars = ax1.bar(positions, importances, yerr=errors, capsize=4,
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

    # Mark important positions
    threshold = importances.mean() + importances.std()
    for i, (pos, imp) in enumerate(zip(positions, importances)):
        if imp > threshold:
            ax1.text(pos, imp + errors[i], f'{imp:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.axhline(y=importances.mean(), color='red', linestyle='--', alpha=0.7,
               label=f'Mean: {importances.mean():.4f}')
    ax1.set_xlabel('Position', fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'{metric.upper()} Drop', fontsize=11, fontweight='bold')
    ax1.set_title(f'Position Importance ({study_type.capitalize()})', fontsize=12, fontweight='bold')
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f'P{i+1}' for i in positions], rotation=0)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Line plot showing trend
    ax2 = axes[0, 1]
    ax2.plot(positions, importances, marker='o', linewidth=2.5, markersize=8,
            color='#2c3e50', markerfacecolor='#e74c3c', markeredgecolor='black',
            markeredgewidth=1.5, alpha=0.8)

    if len(importances) > 3:
        # Add trend line
        z = np.polyfit(positions, importances, 2)
        p = np.poly1d(z)
        ax2.plot(positions, p(positions), "--", alpha=0.7, linewidth=2,
                color='#3498db', label='Trend (polynomial)')

    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.fill_between(positions, 0, importances, alpha=0.2, color='#3498db')
    ax2.set_xlabel('Position', fontsize=11, fontweight='bold')
    ax2.set_ylabel(f'{metric.upper()} Drop', fontsize=11, fontweight='bold')
    ax2.set_title('Position Importance Trend', fontsize=12, fontweight='bold')
    ax2.set_xticks(positions)
    ax2.set_xticklabels([f'P{i+1}' for i in positions])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Heatmap view
    ax3 = axes[1, 0]
    importance_matrix = importances.reshape(1, -1)

    im = ax3.imshow(importance_matrix, cmap='RdYlGn_r', aspect='auto',
                   interpolation='nearest')
    ax3.set_yticks([0])
    ax3.set_yticklabels(['Importance'])
    ax3.set_xticks(range(len(positions)))
    ax3.set_xticklabels([f'P{i+1}' for i in positions])
    ax3.set_title('Position Importance Heatmap', fontsize=12, fontweight='bold')

    # Add text annotations
    for i, val in enumerate(importances):
        ax3.text(i, 0, f'{val:.3f}', ha='center', va='center',
                color='white' if val > importances.mean() else 'black',
                fontweight='bold', fontsize=8)

    plt.colorbar(im, ax=ax3, label=f'{metric.upper()} Drop')

    # 4. Cumulative importance
    ax4 = axes[1, 1]

    # Sort by importance
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    cumulative = np.cumsum(sorted_importances)
    cumulative_pct = (cumulative / cumulative[-1]) * 100 if cumulative[-1] != 0 else cumulative

    ax4.plot(range(len(cumulative)), cumulative_pct, marker='s', linewidth=2.5,
            markersize=7, color='#16a085', markeredgecolor='black', markeredgewidth=1)
    ax4.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% threshold')
    ax4.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% threshold')
    ax4.fill_between(range(len(cumulative)), 0, cumulative_pct, alpha=0.3, color='#16a085')

    ax4.set_xlabel('Number of Top Positions', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cumulative Importance (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Cumulative Position Importance', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Position-Specific {study_type.capitalize()} Analysis\n(Baseline {metric.upper()}: {baseline_score:.4f})',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(os.path.join(out_dir, f'position_importance_{study_type}.png'),
               dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Position importance visualization saved to {out_dir}")


def visualize_feature_importance_heatmap(blosum_results, out_dir):
    """
    Create heatmap visualization for BLOSUM62 feature importance.

    Args:
        blosum_results: DataFrame with BLOSUM feature importance results
        out_dir: Output directory
    """
    os.makedirs(out_dir, exist_ok=True)

    # Amino acid labels for BLOSUM62 (23 dimensions including gap)
    aa_labels = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B', 'Z', 'X']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Feature importance as heatmap
    ax1 = axes[0]

    importances = blosum_results['importance'].values
    importance_matrix = importances.reshape(-1, 1)

    im = ax1.imshow(importance_matrix.T, cmap='RdYlGn_r', aspect='auto')
    ax1.set_xticks(range(len(aa_labels)))
    ax1.set_xticklabels(aa_labels, fontsize=10)
    ax1.set_yticks([0])
    ax1.set_yticklabels(['Importance'])
    ax1.set_title('BLOSUM62 Feature Importance Heatmap', fontsize=13, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, orientation='horizontal', pad=0.1)
    cbar.set_label('Importance (AUC Drop)', fontsize=11, fontweight='bold')

    # 2. Top features bar plot
    ax2 = axes[1]

    # Sort and get top 10
    sorted_indices = np.argsort(importances)[::-1]
    top_n = min(10, len(importances))
    top_indices = sorted_indices[:top_n]

    top_labels = [aa_labels[i] if i < len(aa_labels) else f'F{i}' for i in top_indices]
    top_importances = importances[top_indices]

    colors = plt.cm.Reds(top_importances / top_importances.max())
    bars = ax2.barh(range(top_n), top_importances, color=colors,
                   edgecolor='black', linewidth=1.5)

    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(top_labels, fontsize=11, fontweight='bold')
    ax2.invert_yaxis()
    ax2.set_xlabel('Importance (AUC Drop)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Top {top_n} Most Important BLOSUM62 Features', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_importances)):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f' {val:.4f}', ha='left', va='center', fontweight='bold', fontsize=9)

    plt.suptitle('BLOSUM62 Feature Importance Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(os.path.join(out_dir, 'blosum_feature_importance.png'),
               dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ BLOSUM feature importance visualization saved to {out_dir}")