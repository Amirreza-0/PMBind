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
            'ConcatBarcode': wrap_layer(ConcatBarcode),
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
        filename=os.path.join(out_dir, f"umap_{latent_type}_by_anchor_pair_reduced.png"),
        legend_name='Anchor Pair (Reduced)', figsize=figsize, point_size=point_size, legend_=True,
        legend_font_size=legend_font_size // 2, cbar_font_size=cbar_font_size // 2
    )

    # Plot by Anchor Pair with Amino Acid Colors (new enhanced method)
    plot_anchor_pairs_with_amino_acid_colors(
        embedding=embedding,
        peptide_sequences=df['long_mer'],
        title=f'UMAP of {latent_type.capitalize()} Latents by Anchor Pairs\nColored by 1st Anchor AA (face) and 2nd Anchor AA (outline)',
        filename=os.path.join(out_dir, f"umap_{latent_type}_by_anchor_pair_aa_colors.png"),
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
    figsize: tuple = (15, 12),
    point_size: int = 20,
    edge_width: float = 1.5,
    alpha: float = 0.8
):
    """
    Plot UMAP embedding with points colored by first anchor amino acid
    and outlined by second anchor amino acid color.

    Args:
        embedding: 2D UMAP embedding coordinates
        peptide_sequences: Series of peptide sequences
        title: Plot title
        filename: Output filename
        figsize: Figure size
        point_size: Size of scatter points
        edge_width: Width of outline
        alpha: Point transparency
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    # Get anchor pairs for all peptides
    anchor_pairs = peptide_sequences.apply(get_anchor_pair_amino_acids)
    first_anchors = anchor_pairs.apply(lambda x: x[0])
    second_anchors = anchor_pairs.apply(lambda x: x[1])

    # Get unique amino acids for legend
    unique_first = sorted(set(first_anchors.values))
    unique_second = sorted(set(second_anchors.values))

    # Plot points grouped by first anchor amino acid (face color)
    for aa1 in unique_first:
        mask1 = (first_anchors == aa1)
        if not np.any(mask1):
            continue

        # For this first anchor AA, plot points with different edge colors for second anchor
        for aa2 in unique_second:
            mask2 = (second_anchors == aa2)
            combined_mask = mask1 & mask2

            if not np.any(combined_mask):
                continue

            face_color = AMINO_ACID_COLORS.get(aa1, AMINO_ACID_COLORS['X'])
            edge_color = AMINO_ACID_COLORS.get(aa2, AMINO_ACID_COLORS['X'])

            ax.scatter(
                embedding[combined_mask, 0],
                embedding[combined_mask, 1],
                c=[face_color],
                edgecolors=[edge_color],
                s=point_size,
                alpha=alpha,
                linewidth=edge_width,
                rasterized=True
            )

    # Create legends
    # Legend for face colors (first anchor)
    first_legend_elements = []
    for aa in unique_first:
        color = AMINO_ACID_COLORS.get(aa, AMINO_ACID_COLORS['X'])
        first_legend_elements.append(
            plt.scatter([], [], c=[color], s=point_size, alpha=alpha,
                       label=f'{aa} (1st anchor)')
        )

    first_legend = ax.legend(
        handles=first_legend_elements,
        title='First Anchor (Face Color)',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=10,
        title_fontsize=12
    )
    ax.add_artist(first_legend)

    # Legend for edge colors (second anchor)
    second_legend_elements = []
    for aa in unique_second:
        color = AMINO_ACID_COLORS.get(aa, AMINO_ACID_COLORS['X'])
        second_legend_elements.append(
            plt.scatter([], [], c='white', edgecolors=[color], s=point_size,
                       linewidth=edge_width, label=f'{aa} (2nd anchor)')
        )

    # Calculate position for second legend to avoid overlap
    # Position it lower based on the number of items in the first legend
    second_legend_y_position = max(0.1, 0.95 - len(unique_first) * 0.04)

    second_legend = ax.legend(
        handles=second_legend_elements,
        title='Second Anchor (Edge Color)',
        bbox_to_anchor=(1.02, second_legend_y_position),
        loc='upper left',
        fontsize=10,
        title_fontsize=12
    )

    # Set labels and title
    ax.set_xlabel('UMAP Dimension 1', fontsize=14)
    ax.set_ylabel('UMAP Dimension 2', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Save plot
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Anchor pair plot saved to {filename}")


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


def visualize_per_allele_auc(df, true_labels, predictions, out_dir, dataset_name,
                              allele_col='allele', min_samples=10, top_n=None):
    """
    Compute and visualize per-allele AUC scores.

    Parameters:
        df: DataFrame containing allele information
        true_labels: True binary labels (array-like)
        predictions: Predicted probabilities (array-like)
        out_dir: Output directory to save plots
        dataset_name: Name of the dataset
        allele_col: Column name containing allele information
        min_samples: Minimum number of samples required per allele to compute AUC
        top_n: If specified, only show top N alleles by sample count
    """
    os.makedirs(out_dir, exist_ok=True)

    # Add predictions and true labels to dataframe
    df_copy = df.copy()
    df_copy['true_label'] = true_labels
    df_copy['prediction'] = predictions

    # Compute AUC per allele
    allele_metrics = []
    for allele in df_copy[allele_col].unique():
        allele_mask = df_copy[allele_col] == allele
        allele_true = df_copy.loc[allele_mask, 'true_label']
        allele_pred = df_copy.loc[allele_mask, 'prediction']

        n_samples = len(allele_true)

        # Skip alleles with insufficient samples or only one class
        if n_samples < min_samples:
            continue
        if len(allele_true.unique()) < 2:
            continue

        try:
            fpr, tpr, _ = roc_curve(allele_true, allele_pred)
            allele_auc = auc(fpr, tpr)

            allele_metrics.append({
                'allele': allele,
                'auc': allele_auc,
                'n_samples': n_samples,
                'n_positive': allele_true.sum(),
                'n_negative': (1 - allele_true).sum()
            })
        except Exception as e:
            print(f"Warning: Could not compute AUC for allele {allele}: {e}")
            continue

    # Create DataFrame and sort by AUC
    metrics_df = pd.DataFrame(allele_metrics)
    if len(metrics_df) == 0:
        print(f"Warning: No alleles with sufficient samples to compute AUC")
        return

    metrics_df = metrics_df.sort_values('auc', ascending=False)

    # Filter to top N if specified
    if top_n is not None:
        metrics_df = metrics_df.nlargest(top_n, 'n_samples')

    # Plot 1: Bar chart of per-allele AUC
    plt.figure(figsize=(max(12, len(metrics_df) * 0.4), 8))
    colors = plt.cm.RdYlGn(metrics_df['auc'].values)
    bars = plt.bar(range(len(metrics_df)), metrics_df['auc'], color=colors)

    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Random (AUC=0.5)')
    plt.axhline(y=metrics_df['auc'].mean(), color='blue', linestyle='--',
                linewidth=2, label=f'Mean AUC={metrics_df["auc"].mean():.3f}')

    plt.xlabel('Allele', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.title(f'Per-Allele AUC - {dataset_name}\n({len(metrics_df)} alleles with ≥{min_samples} samples)',
              fontsize=14, fontweight='bold')
    plt.xticks(range(len(metrics_df)), metrics_df['allele'], rotation=90, ha='right')
    plt.ylim([0, 1.05])
    plt.legend(loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'per_allele_auc_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Scatter plot of AUC vs sample count
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(metrics_df['n_samples'], metrics_df['auc'],
                         c=metrics_df['auc'], cmap='RdYlGn',
                         s=100, alpha=0.6, edgecolors='black', linewidth=1)
    plt.colorbar(scatter, label='AUC')

    # Annotate outliers (very high or very low AUC)
    for idx, row in metrics_df.iterrows():
        if row['auc'] > 0.9 or row['auc'] < 0.6:
            plt.annotate(row['allele'], (row['n_samples'], row['auc']),
                        fontsize=8, alpha=0.7, xytext=(5, 5),
                        textcoords='offset points')

    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Random')
    plt.xlabel('Number of Samples', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.title(f'Per-Allele AUC vs Sample Count - {dataset_name}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'per_allele_auc_vs_samples_{dataset_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Distribution of AUC scores
    plt.figure(figsize=(10, 6))
    plt.hist(metrics_df['auc'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Random (AUC=0.5)')
    plt.axvline(x=metrics_df['auc'].mean(), color='blue', linestyle='--',
                linewidth=2, label=f'Mean AUC={metrics_df["auc"].mean():.3f}')
    plt.axvline(x=metrics_df['auc'].median(), color='green', linestyle='--',
                linewidth=2, label=f'Median AUC={metrics_df["auc"].median():.3f}')
    plt.xlabel('AUC', fontsize=12)
    plt.ylabel('Number of Alleles', fontsize=12)
    plt.title(f'Distribution of Per-Allele AUC - {dataset_name}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'per_allele_auc_distribution_{dataset_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(out_dir, f'per_allele_metrics_{dataset_name}.csv'), index=False)

    print(f"✓ Per-allele AUC visualizations saved to {out_dir}")
    print(f"  - {len(metrics_df)} alleles analyzed")
    print(f"  - Mean AUC: {metrics_df['auc'].mean():.3f} ± {metrics_df['auc'].std():.3f}")
    print(f"  - Median AUC: {metrics_df['auc'].median():.3f}")
    print(f"  - Best allele: {metrics_df.iloc[0]['allele']} (AUC={metrics_df.iloc[0]['auc']:.3f})")
    print(f"  - Worst allele: {metrics_df.iloc[-1]['allele']} (AUC={metrics_df.iloc[-1]['auc']:.3f})")


def visualize_anchor_positions_and_binding_pockets(attn_weights, peptide_seq, mhc_seq,
                                                     max_pep_len, max_mhc_len, out_dir,
                                                     sample_idx=0, pooling='max',
                                                     top_k_anchors=3, top_k_pockets=10):
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
        ax.text(idx, pocket_scores[idx], f'{mhc_seq[idx]}',
               ha='center', va='bottom', fontweight='bold', fontsize=8, rotation=0)

    ax.set_xlabel('MHC Position', fontsize=12)
    ax.set_ylabel(f'Attention Score ({pooling} pooling)', fontsize=12)
    ax.set_title(f'MHC Binding Pocket Regions (Sample {sample_idx})\nTop {top_k_pockets} highlighted in red',
                fontsize=14, fontweight='bold')
    ax.set_xticks(positions[::5])  # Show every 5th position to avoid crowding
    ax.set_xticklabels([f'{i+1}' for i in range(0, mhc_len, 5)], fontsize=9)
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

    # === VISUALIZATION 4: 2D Heatmap of Peptide-MHC Cross-Attention with Anchor/Pocket Highlights ===
    # Average attention across heads for clearer visualization
    avg_pep_to_mhc = np.mean(pep_to_mhc, axis=0)  # (pep_len, mhc_len)

    fig, ax = plt.subplots(figsize=(max(12, mhc_len * 0.4), max(8, pep_len * 0.6)))

    # Create heatmap
    im = ax.imshow(avg_pep_to_mhc, aspect='auto', cmap='viridis', interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Score', fontsize=12)

    # Mark top peptide anchor positions with red lines
    for idx in top_anchor_indices:
        ax.axhline(y=idx, color='red', linewidth=2, linestyle='--', alpha=0.7)

    # Mark top MHC binding pocket positions with red lines
    for idx in top_pocket_indices:
        ax.axvline(x=idx, color='yellow', linewidth=2, linestyle='--', alpha=0.7)

    # Labels
    ax.set_xlabel('MHC Position', fontsize=12)
    ax.set_ylabel('Peptide Position', fontsize=12)
    ax.set_title(f'Peptide→MHC Cross-Attention with Anchor & Pocket Highlights (Sample {sample_idx})\n' +
                f'Red lines: Top {top_k_anchors} peptide anchors | Yellow lines: Top {top_k_pockets} MHC pockets',
                fontsize=14, fontweight='bold')

    # Set ticks
    ax.set_xticks(range(0, mhc_len, max(1, mhc_len // 20)))
    ax.set_yticks(range(pep_len))
    ax.set_xticklabels([f'{i+1}' for i in range(0, mhc_len, max(1, mhc_len // 20))], fontsize=9)
    ax.set_yticklabels([f'P{i+1}:{peptide_seq[i]}' for i in range(pep_len)], fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'cross_attention_with_highlights_sample_{sample_idx}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

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