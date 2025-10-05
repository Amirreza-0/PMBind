#!/usr/bin/env python
"""
Simple Latent Interpretability Analysis Script

This script analyzes whether the model's latent space is interpretable by:
1. Running inference on train+test data
2. Clustering the latent representations
3. Comparing latent distances to amino acid distances (BLOSUM62)
4. Showing anchor pair visualizations

Usage:
    python src/run_interpolation.py --fold 1 --model_dir results/model_fold1 --output_dir results/analysis_fold1
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Analysis imports
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr, spearmanr
import umap

# Local imports
from utils import BLOSUM62


def run_inference(fold, model_dir, output_dir):
    """
    Step 1: Run inference on train+test data using existing pipeline.
    """
    print(f"\n=== STEP 1: Running Inference ===")
    print(f"Fold: {fold}, Model: {model_dir}")

    # Get script directory and construct absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Setup paths (now using absolute paths)
    data_root = os.path.join(project_root, "data/cross_validation_dataset")
    embedding_dir = os.path.join(project_root, "results/ESM/esm3-open/PMGen_whole_seq_/")
    allele_seq_path = os.path.join(project_root, "data/alleles/aligned_PMGen_class_1.csv")

    fold_dir = os.path.join(data_root, "mhc1/cv_folds")
    train_path = os.path.join(fold_dir, f"fold_{fold:02d}_train.parquet")
    test_path = os.path.join(data_root, "mhc1/test_set_rarest_alleles.parquet")

    # Create joint train+test dataset
    print("Creating joint train+test dataset...")
    df_train = pd.read_parquet(train_path)
    df_train['source'] = 'train'
    df_test = pd.read_parquet(test_path)
    df_test['source'] = 'test'
    df_joint = pd.concat([df_train, df_test], ignore_index=True)

    # Save joint dataset
    joint_dir = os.path.join(output_dir, "inference_joint")
    os.makedirs(joint_dir, exist_ok=True)
    joint_data_path = os.path.join(joint_dir, "joint_data.parquet")
    df_joint.to_parquet(joint_data_path)

    print(f"Joint dataset: {len(df_train)} train + {len(df_test)} test = {len(df_joint)} total")

    # Run inference
    print("Running inference...")
    infer_script = os.path.join(script_dir, "infer.py")

    cmd = [
        sys.executable, infer_script,
        "--model_weights_path", os.path.join(model_dir, "best_model.weights.h5"),
        "--config_path", os.path.join(model_dir, "run_config.json"),
        "--df_path", joint_data_path,
        "--out_dir", joint_dir,
        "--name", "joint",
        "--allele_seq_path", allele_seq_path,
        "--embedding_key_path", os.path.join(embedding_dir, "mhc1_encodings.csv"),
        "--embedding_npz_path", os.path.join(embedding_dir, "mhc1_encodings.npz"),
        "--source_col", "source"
    ]

    subprocess.run(cmd, check=True)
    print("✓ Inference completed!")

    return joint_dir


def load_data(joint_dir, use_pooled=True):
    """
    Step 2: Load inference results and latent representations.
    """
    print(f"\n=== STEP 2: Loading Data ===")

    # Load results DataFrame
    results_path = os.path.join(joint_dir, "inference_results_joint.csv")
    df = pd.read_csv(results_path)
    print(f"Loaded results: {df.shape}")

    # Load latents from .h5 files
    latent_type = "pooled" if use_pooled else "seq"
    latents_path = os.path.join(joint_dir, f"latents_{latent_type}_joint.h5")

    with h5py.File(latents_path, 'r') as f:
        latents = f['latents'][:]
    print(f"Loaded {latent_type} latents: {latents.shape}")

    # Extract data
    sequences = df['long_mer'].values
    labels = df['assigned_label'].values
    sources = df['source'].values

    print(f"Train samples: {sum(sources == 'train')}")
    print(f"Test samples: {sum(sources == 'test')}")

    return sequences, labels, sources, latents


def cluster_latents(latents, n_clusters=None):
    """
    Step 3: Cluster the latent representations using K-means.
    """
    print(f"\n=== STEP 3: Clustering Latents ===")

    # Flatten latents if needed (for sequence latents)
    if len(latents.shape) > 2:
        latents_flat = np.mean(latents, axis=1)
        print(f"Flattened latents: {latents.shape} -> {latents_flat.shape}")
    else:
        latents_flat = latents

    # Find optimal number of clusters if not specified
    if n_clusters is None:
        print("Finding optimal number of clusters...")
        best_score = -1
        best_k = 5
        for k in range(2, 21):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = kmeans.fit_predict(latents_flat)
            score = silhouette_score(latents_flat, labels)
            if score > best_score:
                best_score = score
                best_k = k
        n_clusters = best_k
        print(f"Optimal clusters: {n_clusters} (silhouette: {best_score:.3f})")

    # Perform final clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latents_flat)
    score = silhouette_score(latents_flat, cluster_labels)
    print(f"✓ Clustered into {n_clusters} clusters (silhouette: {score:.3f})")

    return cluster_labels, n_clusters, latents_flat


def calculate_distances(sequences, sources, cluster_labels, latents_flat, top_p=10):
    """
    Step 4: Calculate distances between test and train samples.
    """
    print(f"\n=== STEP 4: Calculating Distances ===")

    train_mask = sources == 'train'
    test_mask = sources == 'test'
    n_clusters = len(np.unique(cluster_labels))
    n_test = sum(test_mask)

    print(f"Computing distances for {n_test} test samples across {n_clusters} clusters")
    print(f"Using top-{top_p} nearest neighbors")

    # Initialize distance matrices
    latent_distances = np.zeros((n_test, n_clusters))
    aa_distances = np.zeros((n_test, n_clusters))

    # Get train/test data
    train_sequences = sequences[train_mask]
    test_sequences = sequences[test_mask]
    train_clusters = cluster_labels[train_mask]
    train_latents = latents_flat[train_mask]
    test_latents = latents_flat[test_mask]

    # Calculate distances for each test sample to each cluster
    for test_idx, (test_seq, test_latent) in enumerate(tqdm(zip(test_sequences, test_latents),
                                                           desc="Test samples", total=len(test_sequences))):
        for cluster_id in range(n_clusters):
            # Get train samples in this cluster
            cluster_mask = train_clusters == cluster_id
            if not np.any(cluster_mask):
                latent_distances[test_idx, cluster_id] = np.inf
                aa_distances[test_idx, cluster_id] = np.inf
                continue

            cluster_train_seqs = train_sequences[cluster_mask]
            cluster_train_latents = train_latents[cluster_mask]

            # Calculate amino acid distances using BLOSUM62
            aa_dists = [blosum62_distance(test_seq, train_seq) for train_seq in cluster_train_seqs]

            # Calculate latent distances
            lat_dists = [euclidean(test_latent, train_latent) for train_latent in cluster_train_latents]

            # Use top-p nearest distances
            top_aa_dists = sorted(aa_dists)[:min(top_p, len(aa_dists))]
            top_lat_dists = sorted(lat_dists)[:min(top_p, len(lat_dists))]

            aa_distances[test_idx, cluster_id] = np.mean(top_aa_dists)
            latent_distances[test_idx, cluster_id] = np.mean(top_lat_dists)

    print("✓ Distance calculations completed!")
    return latent_distances, aa_distances


def blosum62_distance(seq1, seq2):
    """
    Calculate BLOSUM62 distance between two sequences using:
    D_ij = (B_ii + B_jj) / 2 - B_ij
    """
    max_len = max(len(seq1), len(seq2))
    seq1_padded = seq1.ljust(max_len, '-')
    seq2_padded = seq2.ljust(max_len, '-')

    total_distance = 0
    valid_positions = 0

    for aa1, aa2 in zip(seq1_padded, seq2_padded):
        if aa1 in BLOSUM62 and aa2 in BLOSUM62:
            aa1_idx = list(BLOSUM62.keys()).index(aa1)
            aa2_idx = list(BLOSUM62.keys()).index(aa2)

            B_ii = BLOSUM62[aa1][aa1_idx]
            B_jj = BLOSUM62[aa2][aa2_idx]
            B_ij = BLOSUM62[aa1][aa2_idx]

            distance = (B_ii + B_jj) / 2 - B_ij
            total_distance += distance
            valid_positions += 1

    return total_distance / valid_positions if valid_positions > 0 else float('inf')


def analyze_correlations(latent_distances, aa_distances):
    """
    Step 5: Analyze correlations between latent and amino acid distances.
    """
    print(f"\n=== STEP 5: Analyzing Correlations ===")

    # Flatten distance matrices
    lat_flat = latent_distances.flatten()
    aa_flat = aa_distances.flatten()

    # Remove infinite distances
    valid_mask = np.isfinite(lat_flat) & np.isfinite(aa_flat)
    lat_valid = lat_flat[valid_mask]
    aa_valid = aa_flat[valid_mask]

    # Calculate correlations
    pearson_corr, _ = pearsonr(lat_valid, aa_valid)
    spearman_corr, _ = spearmanr(lat_valid, aa_valid)

    results = {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'n_valid_pairs': len(lat_valid)
    }

    print(f"Pearson correlation: {pearson_corr:.3f}")
    print(f"Spearman correlation: {spearman_corr:.3f}")
    print(f"Valid data points: {len(lat_valid):,}")

    return results


def create_visualizations(sequences, sources, cluster_labels, latents_flat,
                        latent_distances, aa_distances, correlation_results, output_dir):
    """
    Step 6: Create visualizations of the analysis results.
    """
    print(f"\n=== STEP 6: Creating Visualizations ===")

    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # 1. UMAP of latents with anchor pair colors
    print("Creating UMAP visualization...")
    create_umap_plot(sequences, sources, cluster_labels, latents_flat, viz_dir)

    # 2. Distance correlation scatter plot
    print("Creating correlation plots...")
    create_correlation_plots(latent_distances, aa_distances, correlation_results, viz_dir)

    # 3. Distance distributions
    print("Creating distance distributions...")
    create_distance_plots(latent_distances, aa_distances, viz_dir)

    # 4. Interpolation visualization (test samples to nearest train samples)
    print("Creating interpolation visualization...")
    create_interpolation_plot(sequences, sources, latents_flat, viz_dir)

    print(f"✓ Visualizations saved to {viz_dir}")


def create_umap_plot(sequences, sources, cluster_labels, latents_flat, viz_dir):
    """Create UMAP visualization with anchor pair colors."""
    # Create UMAP embedding
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(latents_flat)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Plot 1: Train vs Test
    colors = ['blue' if s == 'train' else 'red' for s in sources]
    axes[0].scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.6, s=20)
    axes[0].set_title('Latent Space: Train vs Test')
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')

    # Plot 2: Clusters
    scatter = axes[1].scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6, s=20)
    axes[1].set_title('Latent Space: Clusters')
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=axes[1], label='Cluster')

    # Plot 3: Anchor pair colors
    plot_anchor_pairs(embedding, sequences, axes[2])

    # Plot 4: Combined - clusters with test overlay
    scatter = axes[3].scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap='tab10', alpha=0.4, s=15)
    test_mask = sources == 'test'
    axes[3].scatter(embedding[test_mask, 0], embedding[test_mask, 1], c='red', marker='x', s=50, alpha=0.8, label='Test')
    axes[3].set_title('Clusters with Test Overlay')
    axes[3].set_xlabel('UMAP 1')
    axes[3].set_ylabel('UMAP 2')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'umap_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_anchor_pairs(embedding, sequences, ax):
    """Plot points colored by anchor pair amino acids."""
    # Get anchor pairs (P2 and P-omega positions)
    anchor_pairs = []
    for seq in sequences:
        if len(seq) >= 2:
            p2 = seq[1].upper()  # Second amino acid
            p_omega = seq[-1].upper()  # Last amino acid
            anchor_pairs.append((p2, p_omega))
        else:
            anchor_pairs.append(('X', 'X'))

    # Color mapping for amino acids
    aa_colors = {
        'A': '#FF6B6B', 'C': '#4ECDC4', 'D': '#45B7D1', 'E': '#96CEB4',
        'F': '#FFEAA7', 'G': '#DDA0DD', 'H': '#98D8C8', 'I': '#F7DC6F',
        'K': '#BB8FCE', 'L': '#85C1E9', 'M': '#F8C471', 'N': '#82E0AA',
        'P': '#F1948A', 'Q': '#85C1E9', 'R': '#D7BDE2', 'S': '#A9DFBF',
        'T': '#FAD7A0', 'V': '#AED6F1', 'W': '#D5A6BD', 'Y': '#F9E79F',
        'X': '#808080'
    }

    # Plot points with face color = P2, edge color = P-omega
    for i, (p2, p_omega) in enumerate(anchor_pairs):
        face_color = aa_colors.get(p2, aa_colors['X'])
        edge_color = aa_colors.get(p_omega, aa_colors['X'])

        ax.scatter(embedding[i, 0], embedding[i, 1],
                  c=[face_color], edgecolors=[edge_color],
                  s=20, alpha=0.8, linewidth=1)

    ax.set_title('Anchor Pair Colors\n(Face: P2, Edge: P-omega)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')


def create_correlation_plots(latent_distances, aa_distances, correlation_results, viz_dir):
    """Create correlation scatter plot."""
    lat_flat = latent_distances.flatten()
    aa_flat = aa_distances.flatten()
    valid_mask = np.isfinite(lat_flat) & np.isfinite(aa_flat)

    plt.figure(figsize=(8, 6))
    plt.scatter(lat_flat[valid_mask], aa_flat[valid_mask], alpha=0.5, s=10)
    plt.xlabel('Latent Distance')
    plt.ylabel('Amino Acid Distance (BLOSUM62)')
    plt.title(f'Distance Correlation (r = {correlation_results["pearson"]:.3f})')

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'correlation_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_distance_plots(latent_distances, aa_distances, viz_dir):
    """Create distance distribution plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Latent distances
    lat_flat = latent_distances.flatten()
    lat_flat = lat_flat[np.isfinite(lat_flat)]
    axes[0].hist(lat_flat, bins=30, alpha=0.7, color='blue')
    axes[0].set_title('Latent Distance Distribution')
    axes[0].set_xlabel('Distance')
    axes[0].set_ylabel('Frequency')

    # AA distances
    aa_flat = aa_distances.flatten()
    aa_flat = aa_flat[np.isfinite(aa_flat)]
    axes[1].hist(aa_flat, bins=30, alpha=0.7, color='red')
    axes[1].set_title('Amino Acid Distance Distribution')
    axes[1].set_xlabel('Distance')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'distance_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_interpolation_plot(sequences, sources, latents_flat, viz_dir, n_neighbors=5, max_samples=50):
    """
    Create interpolation visualization showing how test samples relate to nearest train samples.

    Args:
        sequences: Array of peptide sequences
        sources: Array indicating 'train' or 'test' for each sample
        latents_flat: Flattened latent representations
        viz_dir: Output directory
        n_neighbors: Number of nearest neighbors to show
        max_samples: Maximum number of test samples to visualize
    """
    from sklearn.neighbors import NearestNeighbors

    # Create UMAP embedding for visualization
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(latents_flat)

    # Get train and test masks
    train_mask = sources == 'train'
    test_mask = sources == 'test'

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    # Limit test samples for visualization
    if len(test_indices) > max_samples:
        test_indices = np.random.choice(test_indices, max_samples, replace=False)

    # Find nearest neighbors in latent space
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(latents_flat[train_mask])

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot 1: Interpolation lines in UMAP space
    ax1 = axes[0]

    # Plot all training points
    ax1.scatter(embedding[train_mask, 0], embedding[train_mask, 1],
               c='lightblue', s=20, alpha=0.5, label='Train', zorder=1)

    # For each test sample, draw lines to nearest neighbors
    for test_idx in test_indices:
        test_latent = latents_flat[test_idx:test_idx+1]
        distances, neighbor_indices = nbrs.kneighbors(test_latent)

        # Map back to original indices
        train_neighbor_indices = train_indices[neighbor_indices[0]]

        # Draw lines from test to neighbors
        for neighbor_idx in train_neighbor_indices:
            ax1.plot([embedding[test_idx, 0], embedding[neighbor_idx, 0]],
                    [embedding[test_idx, 1], embedding[neighbor_idx, 1]],
                    'gray', alpha=0.3, linewidth=0.5, zorder=2)

        # Plot the test sample on top
        ax1.scatter(embedding[test_idx, 0], embedding[test_idx, 1],
                   c='red', s=100, marker='*', edgecolors='black',
                   linewidth=1, zorder=3)

    ax1.set_title(f'Latent Space Interpolation\n(Test samples with {n_neighbors} nearest train neighbors)')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.legend()

    # Plot 2: Distance heatmap showing interpolation weights
    ax2 = axes[1]

    # Calculate distances for visualization
    distance_matrix = []
    test_labels = []

    for i, test_idx in enumerate(test_indices[:20]):  # Limit to 20 for readability
        test_latent = latents_flat[test_idx:test_idx+1]
        distances, neighbor_indices = nbrs.kneighbors(test_latent)

        # Convert distances to similarity weights (inverse distance)
        weights = 1.0 / (distances[0] + 1e-6)
        weights = weights / weights.sum()  # Normalize to sum to 1

        distance_matrix.append(weights)
        test_labels.append(f"Test {i+1}\n{sequences[test_idx][:8]}...")

    distance_matrix = np.array(distance_matrix)

    im = ax2.imshow(distance_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_yticks(range(len(test_labels)))
    ax2.set_yticklabels(test_labels, fontsize=8)
    ax2.set_xlabel(f'Top {n_neighbors} Nearest Train Neighbors', fontsize=12)
    ax2.set_title('Interpolation Weights\n(Normalized inverse distances)', fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Weight', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'interpolation_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Interpolation plot shows {len(test_indices)} test samples with nearest train neighbors")


def save_results(correlation_results, latent_distances, aa_distances, output_dir):
    """
    Step 7: Save analysis results.
    """
    print(f"\n=== STEP 7: Saving Results ===")

    # Save correlation results
    results_summary = {
        'interpretability_analysis': {
            'pearson_correlation': float(correlation_results['pearson']),
            'spearman_correlation': float(correlation_results['spearman']),
            'n_valid_pairs': int(correlation_results['n_valid_pairs']),
            'interpretation': get_interpretation(correlation_results['pearson'])
        }
    }

    with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Save distance matrices
    np.save(os.path.join(output_dir, 'latent_distances.npy'), latent_distances)
    np.save(os.path.join(output_dir, 'aa_distances.npy'), aa_distances)

    print(f"✓ Results saved to {output_dir}")
    return results_summary


def get_interpretation(pearson_score):
    """Get interpretation of correlation score."""
    if pearson_score > 0.5:
        return "STRONG: Model latents strongly correlate with amino acid similarities"
    elif pearson_score > 0.3:
        return "MODERATE: Model latents moderately correlate with amino acid similarities"
    elif pearson_score > 0.1:
        return "WEAK: Model latents weakly correlate with amino acid similarities"
    else:
        return "POOR: Model latents do not correlate well with amino acid similarities"


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(description='Simple Latent Interpretability Analysis')
    parser.add_argument('--fold', type=int, required=True, help='Cross-validation fold number')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory with trained model')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--top_p', type=int, default=10, help='Top-p nearest neighbors (default: 10)')
    parser.add_argument('--n_clusters', type=int, default=None, help='Number of clusters (auto if None)')
    parser.add_argument('--use_pooled', action='store_true', default=True, help='Use pooled latents')
    parser.add_argument('--skip_inference', action='store_true', help='Skip inference (use existing results)')

    args = parser.parse_args()

    print("="*60)
    print("LATENT INTERPRETABILITY ANALYSIS")
    print("="*60)

    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Step 1: Run inference (unless skipped)
        if not args.skip_inference:
            joint_dir = run_inference(args.fold, args.model_dir, args.output_dir)
        else:
            joint_dir = os.path.join(args.output_dir, "inference_joint")
            print(f"\nSkipping inference - using existing results in {joint_dir}")

        # Step 2: Load data
        sequences, labels, sources, latents = load_data(joint_dir, args.use_pooled)

        # Step 3: Cluster latents
        cluster_labels, n_clusters, latents_flat = cluster_latents(latents, args.n_clusters)

        # Step 4: Calculate distances
        latent_distances, aa_distances = calculate_distances(
            sequences, sources, cluster_labels, latents_flat, args.top_p)

        # Step 5: Analyze correlations
        correlation_results = analyze_correlations(latent_distances, aa_distances)

        # Step 6: Create visualizations
        create_visualizations(sequences, sources, cluster_labels, latents_flat, latent_distances,
                            aa_distances, correlation_results, args.output_dir)

        # Step 7: Save results
        results_summary = save_results(correlation_results, latent_distances, aa_distances, args.output_dir)

        # Final summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Pearson Correlation: {correlation_results['pearson']:.3f}")
        print(f"Spearman Correlation: {correlation_results['spearman']:.3f}")
        print(f"Interpretation: {results_summary['interpretability_analysis']['interpretation']}")
        print(f"Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"\nERROR: Analysis failed - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()