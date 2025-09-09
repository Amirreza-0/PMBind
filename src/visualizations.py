# visualize tensorflow model

import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from utils import AttentionLayer, PositionalEncoding, AnchorPositionExtractor, SplitLayer, ConcatMask, ConcatBarcode, \
    MaskedEmbedding
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import matplotlib.colors as mcolors
from utils import reduced_anchor_pair, cn_terminal_amino_acids, peptide_properties_biopython
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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


def visualize_cross_attention_weights(cross_attn_scores, peptide_seq, mhc_seq):
    """
    Visualize cross-attention weights between peptide and MHC sequences.

    Args:
        cross_attn_scores: Tensor of shape (B, N_peptide, N_mhc) with attention scores.
        peptide_seq: List of peptide sequences.
        mhc_seq: List of MHC sequences.
        top_n: Number of top attention scores to visualize.

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
import numpy as np
import matplotlib.pyplot as plt


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

            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels, fontsize=cbar_font_size)

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

                cbar.set_ticks(tick_positions)
                cbar.ax.set_yticklabels(tick_labels, fontsize=cbar_font_size)
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

    # Plot by Reduced Anchor Pair
    anchor_pair_labels = df['long_mer'].apply(reduced_anchor_pair).astype('category')
    unique_anchor_pairs = sorted(anchor_pair_labels.unique())
    colors = sns.color_palette("hls", n_colors=len(unique_anchor_pairs))
    anchor_color_map = {pair: color for pair, color in zip(unique_anchor_pairs, colors)}
    if 'Short' in anchor_color_map:
        anchor_color_map['Short'] = [0.7, 0.7, 0.7, 0.5]

    _plot_umap(
        embedding=embedding, labels=anchor_pair_labels, color_map=anchor_color_map,
        title=f'UMAP of {latent_type.capitalize()} Latents by Reduced Anchor Pairs\n({len(unique_anchor_pairs)} unique pairs)',
        filename=os.path.join(out_dir, f"umap_{latent_type}_by_anchor_pair.png"),
        legend_name='Anchor Pair (Reduced)', figsize=figsize, point_size=point_size, legend_=True,
        legend_font_size=legend_font_size // 2, cbar_font_size=cbar_font_size // 2
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

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot losses
    if 'train_loss' in history:
        axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    if 'cls_loss' in history:
        axes[0, 0].plot(epochs, history['cls_loss'], label='CLS Loss', color='red')
    if 'pep_loss' in history:
        axes[0, 0].plot(epochs, history['pep_loss'], label='PEP Loss', color='green')
    if 'mhc_loss' in history:
        axes[0, 0].plot(epochs, history['mhc_loss'], label='MHC Loss', color='orange')
    axes[0, 0].set_title('Losses')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot AUC
    if 'train_auc' in history:
        axes[0, 1].plot(epochs, history['train_auc'], label='Train AUC', color='blue')
    if 'val_auc' in history:
        axes[0, 1].plot(epochs, history['val_auc'], label='Val AUC', color='orange')
    axes[0, 1].set_title('AUC')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot Accuracy
    if 'train_acc' in history:
        axes[1, 0].plot(epochs, history['train_acc'], label='Train Acc', color='blue')
    if 'val_acc' in history:
        axes[1, 0].plot(epochs, history['val_acc'], label='Val Acc', color='orange')
    axes[1, 0].set_title('Accuracy')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # If there's val_loss, plot it, else maybe plot something else or leave empty
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