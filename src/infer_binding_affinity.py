#!/usr/bin/env python
"""
infer_binding_affinity.py: Inference script for binding affinity data with measurement values.

This script performs inference on binding affinity data and analyzes the correlation between
model predictions and actual binding affinity measurements. It generates visualizations showing
three distinct regions:
- Bottom left: Non-binders (low affinity, low prediction)
- Middle: Noisy samples (mixed affinity and predictions)
- Top right: Strong binders (high affinity, high prediction)

Usage:
    python infer_binding_affinity.py \
        --model_weights_path <path_to_model> \
        --config_path <path_to_config> \
        --df_path <path_to_parquet_with_measurements> \
        --measurement_col <column_name_with_measurements> \
        --out_dir <output_directory> \
        --name <dataset_name> \
        --allele_seq_path <path_to_allele_sequences> \
        --embedding_key_path <path_to_embedding_keys> \
        --embedding_npz_path <path_to_embeddings> \
        [--batch_size <batch_size>] \
        [--invert_measurements]  # Use if lower values = stronger binding (e.g., IC50)

Author: Generated for PMBind Project
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pyarrow.parquet as pq
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

# Local imports - reuse functions from infer.py
from utils import (
    seq_to_onehot, get_embed_key, clean_key, seq_to_blossom62,
    NORM_TOKEN, MASK_TOKEN, PAD_TOKEN, PAD_VALUE,
    load_embedding_db
)
from models import pmbind_multitask_plus as pmbind

# Global variables
EMB_DB_p = None
ESM_DIM = None
MHC_CLASS = 1
EMB_NORM_METHOD = "robust_zscore"


class OptimizedDataGenerator(keras.utils.Sequence):
    """Optimized data generator for inference (reused from infer.py)."""

    def __init__(self, df, seq_map, embed_map, max_pep_len, max_mhc_len, batch_size,
                 normalization_method=None):
        super().__init__()
        self.df = df
        self.seq_map = seq_map
        self.embed_map = embed_map
        self.max_pep_len = max_pep_len
        self.max_mhc_len = max_mhc_len
        self.batch_size = batch_size
        self.long_mer_arr = df['long_mer'].to_numpy()
        self.emb_key_arr = df['_emb_key'].to_numpy()
        self.cleaned_key_arr = df['_cleaned_key'].to_numpy()
        self.mhc_seq_arr = df['_mhc_seq'].to_numpy()
        self.label_arr = df['assigned_label'].to_numpy() if 'assigned_label' in df.columns else np.zeros(len(df))
        self.indices = np.arange(len(df))
        self.normalization_method = normalization_method

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.indices))
        return self._generate_batch(self.indices[start_idx:end_idx])

    def _get_embedding(self, emb_key, cleaned_key):
        """Get embedding and apply normalization."""
        try:
            emb = EMB_DB_p[emb_key]
        except KeyError:
            raise KeyError(f"embedding not found for emb_key {emb_key}, cleaned_key: {cleaned_key}")
        return self._normalize_embedding(emb)

    def _normalize_embedding(self, emb):
        """Apply robust normalization to handle extreme values."""
        if self.normalization_method == "robust_zscore":
            mean = emb.mean()
            std = emb.std()
            emb_norm = (emb - mean) / (std + 1e-8)
            emb_norm = np.clip(emb_norm, -5, 5)
            return emb_norm
        else:
            return emb

    def _generate_batch(self, batch_indices):
        n = len(batch_indices)
        data = {
            "pep_blossom62": np.zeros((n, self.max_pep_len, 23), np.float32),
            "pep_mask": np.full((n, self.max_pep_len), PAD_TOKEN, dtype=np.float32),
            "mhc_emb": np.zeros((n, self.max_mhc_len, ESM_DIM), np.float32),
            "mhc_mask": np.full((n, self.max_mhc_len), PAD_TOKEN, dtype=np.float32),
            "pep_ohe_target": np.zeros((n, self.max_pep_len, 21), np.float32),
            "mhc_ohe_target": np.zeros((n, self.max_mhc_len, 21), np.float32),
            "labels": np.zeros((n, 1), np.float32),
            "sample_weights": np.ones((n, 1), np.float32)
        }

        for i, master_idx in enumerate(batch_indices):
            pep_seq = self.long_mer_arr[master_idx].upper()
            emb_key = self.emb_key_arr[master_idx]
            cleaned_key = self.cleaned_key_arr[master_idx]
            mhc_seq = self.mhc_seq_arr[master_idx]
            pep_len = len(pep_seq)

            # Peptide features
            data["pep_blossom62"][i] = seq_to_blossom62(pep_seq, max_seq_len=self.max_pep_len)
            data["pep_ohe_target"][i] = seq_to_onehot(pep_seq, max_seq_len=self.max_pep_len)
            data["pep_mask"][i, :pep_len] = NORM_TOKEN

            # MHC features
            emb = self._get_embedding(emb_key, cleaned_key)
            L = emb.shape[0]
            data["mhc_emb"][i, :L] = emb
            data["mhc_ohe_target"][i] = seq_to_onehot(mhc_seq, max_seq_len=self.max_mhc_len)
            is_padding = np.all(data["mhc_ohe_target"][i, :] == 0, axis=-1)
            data["mhc_mask"][i, ~is_padding] = NORM_TOKEN

            data["labels"][i, 0] = float(self.label_arr[master_idx])

        # Convert to tensors
        tensor_data = {k: tf.convert_to_tensor(v) for k, v in data.items()}
        return tensor_data


def preprocess_df(df, seq_map, embed_map):
    """Preprocess dataframe to add MHC sequence and embedding keys (from infer.py)."""
    df['_cleaned_key'] = df.apply(
        lambda r: r.get('mhc_embedding_key', r['allele'].replace(' ', '').replace('*', '').replace(':', '')),
        axis=1
    )
    df['_emb_key'] = df['_cleaned_key'].apply(lambda k: get_embed_key(clean_key(k), embed_map))
    df['_mhc_seq'] = df['_emb_key'].apply(lambda k: seq_map.get(k, ''))
    return df


def load_model_and_config(model_weights_path, config_path, embedding_npz_path):
    """Load model and configuration (adapted from infer.py)."""
    global ESM_DIM, MHC_CLASS, EMB_DB_p

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    MHC_CLASS = config.get("MHC_CLASS", 1)

    # Load embeddings
    print("Loading MHC embedding database...")
    EMB_DB_p = load_embedding_db(embedding_npz_path)
    ESM_DIM = int(next(iter(EMB_DB_p.values())).shape[1])
    print(f"✓ Loaded embedding database with ESM_DIM={ESM_DIM}")

    # Load model
    if model_weights_path.endswith('.keras'):
        print(f"Loading full model from {model_weights_path}...")
        model = keras.models.load_model(model_weights_path, compile=False)
        print("✓ Full model loaded successfully.")

        # Extract dimensions from loaded model
        max_pep_len = model.input[0].shape[1]
        max_mhc_len = model.input[2].shape[1]
        embed_dim = model.get_layer("pep_dense_embed").output.shape[-1]

        try:
            attn_layer = model.get_layer("pmhc_2d_masked_attention2")
            heads = attn_layer.heads
        except:
            try:
                attn_layer = model.get_layer("pmhc_2d_masked_attention")
                heads = attn_layer.heads
            except:
                heads = config.get("HEADS", 8)
    else:
        # For weights-only files, build model from config
        max_pep_len = config.get("MAX_PEP_LEN", 15)
        max_mhc_len = config.get("MAX_MHC_LEN", 400)
        embed_dim = config.get("EMBED_DIM", 32)
        heads = config.get("HEADS", 8)

        print(f"Building model and loading weights from {model_weights_path}...")
        model = pmbind(
            max_pep_len=max_pep_len,
            max_mhc_len=max_mhc_len,
            emb_dim=embed_dim,
            heads=heads,
            noise_std=0.0,
            drop_out_rate=0.0,
            l2_reg=config.get("L2_REG", 0.0),
            ESM_dim=ESM_DIM
        )

        # Build model with dummy data (need to create a small dummy dataset)
        print("Building model with dummy data...")
        dummy_data = {
            "pep_blossom62": tf.zeros((1, max_pep_len, 23), dtype=tf.float32),
            "pep_mask": tf.ones((1, max_pep_len), dtype=tf.float32),
            "mhc_emb": tf.zeros((1, max_mhc_len, ESM_DIM), dtype=tf.float32),
            "mhc_mask": tf.ones((1, max_mhc_len), dtype=tf.float32),
            "pep_ohe_target": tf.zeros((1, max_pep_len, 21), dtype=tf.float32),
            "mhc_ohe_target": tf.zeros((1, max_mhc_len, 21), dtype=tf.float32),
            "labels": tf.zeros((1, 1), dtype=tf.float32),
            "sample_weights": tf.ones((1, 1), dtype=tf.float32)
        }
        model(dummy_data, training=False)
        model.load_weights(model_weights_path)
        print("✓ Model weights loaded successfully.")

    print(f"Model configuration: max_pep_len={max_pep_len}, max_mhc_len={max_mhc_len}, "
          f"embed_dim={embed_dim}, heads={heads}")

    return model, config, max_pep_len, max_mhc_len


def run_inference(model, df, seq_map, embed_map, max_pep_len, max_mhc_len, batch_size):
    """Run inference on the dataset (adapted from infer.py)."""
    print("\nRunning inference...")

    infer_gen = OptimizedDataGenerator(
        df, seq_map, embed_map, max_pep_len, max_mhc_len, batch_size,
        normalization_method=EMB_NORM_METHOD
    )

    all_predictions = []
    all_labels = []

    for batch in tqdm(infer_gen, desc="Inference", file=sys.stdout):
        outputs = model(batch, training=False)
        all_predictions.append(outputs["cls_ypred"].numpy())
        all_labels.append(batch["labels"].numpy())

    all_predictions = np.concatenate(all_predictions).squeeze()
    all_labels = np.concatenate(all_labels).squeeze()

    df["prediction_score"] = all_predictions
    df["prediction_label"] = (all_predictions >= 0.5).astype(int)

    print(f"✓ Inference complete on {len(df)} samples")

    return df


def visualize_binding_affinity_correlation(df, measurement_col, out_dir, name, invert_measurements=False):
    """
    Generate comprehensive correlation plots between predictions and binding affinity measurements.

    Args:
        df: DataFrame with 'prediction_score' and measurement column
        measurement_col: Name of the column containing binding affinity measurements
        out_dir: Output directory for visualizations
        name: Dataset name for titles
        invert_measurements: If True, invert measurements (e.g., for IC50 where lower = stronger)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Prepare data
    predictions = df['prediction_score'].values
    measurements = df[measurement_col].values

    # Invert if needed (e.g., IC50: lower values = stronger binding)
    if invert_measurements:
        measurements = 1 / (measurements + 1e-10)  # Invert with small epsilon to avoid division by zero
        measurement_label = f"1 / {measurement_col}"
    else:
        measurement_label = measurement_col

    # Normalize measurements to [0, 1] for better visualization
    measurements_norm = (measurements - measurements.min()) / (measurements.max() - measurements.min() + 1e-10)

    # Calculate correlations
    pearson_r, pearson_p = pearsonr(predictions, measurements_norm)
    spearman_r, spearman_p = spearmanr(predictions, measurements_norm)

    print(f"\n{'='*60}")
    print(f"Correlation Analysis Results")
    print(f"{'='*60}")
    print(f"Pearson correlation:  r = {pearson_r:.4f}, p-value = {pearson_p:.4e}")
    print(f"Spearman correlation: ρ = {spearman_r:.4f}, p-value = {spearman_p:.4e}")
    print(f"{'='*60}\n")

    # Define regions
    # Bottom left: low affinity, low prediction (true negatives)
    # Middle: mixed (noisy samples)
    # Top right: high affinity, high prediction (true positives)

    bottom_left = (predictions < 0.33) & (measurements_norm < 0.33)
    top_right = (predictions > 0.67) & (measurements_norm > 0.67)
    middle = ~(bottom_left | top_right)

    print(f"Region Statistics:")
    print(f"  Bottom-left (Non-binders):  {bottom_left.sum():>6} samples ({100*bottom_left.sum()/len(df):.1f}%)")
    print(f"  Middle (Noisy):             {middle.sum():>6} samples ({100*middle.sum()/len(df):.1f}%)")
    print(f"  Top-right (Strong binders): {top_right.sum():>6} samples ({100*top_right.sum()/len(df):.1f}%)")

    # ============================================================================
    # Main correlation plot with regions and marginal distributions
    # ============================================================================

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 3, 1], width_ratios=[1, 3, 1],
                          hspace=0.05, wspace=0.05)

    # Main scatter plot
    ax_main = fig.add_subplot(gs[1, 1])

    # Plot regions with different colors
    scatter_kws = {'alpha': 0.4, 's': 20, 'edgecolors': 'none'}

    ax_main.scatter(measurements_norm[bottom_left], predictions[bottom_left],
                   c='#3498db', label=f'Non-binders (n={bottom_left.sum()})', **scatter_kws)
    ax_main.scatter(measurements_norm[middle], predictions[middle],
                   c='#f39c12', label=f'Noisy samples (n={middle.sum()})', **scatter_kws)
    ax_main.scatter(measurements_norm[top_right], predictions[top_right],
                   c='#e74c3c', label=f'Strong binders (n={top_right.sum()})', **scatter_kws)

    # Add region boundaries
    ax_main.axhline(y=0.33, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax_main.axhline(y=0.67, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax_main.axvline(x=0.33, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax_main.axvline(x=0.67, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    # Add diagonal line
    ax_main.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1.5, label='Perfect correlation')

    # Add regression line
    z = np.polyfit(measurements_norm, predictions, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 1, 100)
    ax_main.plot(x_line, p(x_line), 'r-', alpha=0.5, linewidth=2, label=f'Linear fit (y={z[0]:.2f}x+{z[1]:.2f})')

    # Annotations for regions
    ax_main.text(0.15, 0.15, 'Non-binders\n(True Negatives)',
                ha='center', va='center', fontsize=10, color='#3498db', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='#3498db'))
    ax_main.text(0.85, 0.85, 'Strong Binders\n(True Positives)',
                ha='center', va='center', fontsize=10, color='#e74c3c', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='#e74c3c'))
    ax_main.text(0.5, 0.5, 'Noisy Region\n(Mixed)',
                ha='center', va='center', fontsize=10, color='#f39c12', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='#f39c12'))

    # Add correlation text
    textstr = f'Pearson: r = {pearson_r:.3f} (p={pearson_p:.2e})\nSpearman: ρ = {spearman_r:.3f} (p={spearman_p:.2e})'
    ax_main.text(0.05, 0.95, textstr, transform=ax_main.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax_main.set_xlabel(f'Normalized {measurement_label}', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Model Prediction Probability', fontsize=12, fontweight='bold')
    ax_main.set_xlim(-0.05, 1.05)
    ax_main.set_ylim(-0.05, 1.05)
    ax_main.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax_main.grid(True, alpha=0.2)

    # Top marginal distribution (measurements)
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    ax_top.hist(measurements_norm[bottom_left], bins=50, alpha=0.5, color='#3498db', density=True)
    ax_top.hist(measurements_norm[middle], bins=50, alpha=0.5, color='#f39c12', density=True)
    ax_top.hist(measurements_norm[top_right], bins=50, alpha=0.5, color='#e74c3c', density=True)
    ax_top.set_ylabel('Density', fontsize=10)
    ax_top.set_title(f'Binding Affinity vs Model Predictions\nDataset: {name}',
                    fontsize=14, fontweight='bold', pad=15)
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, alpha=0.2, axis='y')

    # Right marginal distribution (predictions)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    ax_right.hist(predictions[bottom_left], bins=50, alpha=0.5, color='#3498db',
                 orientation='horizontal', density=True)
    ax_right.hist(predictions[middle], bins=50, alpha=0.5, color='#f39c12',
                 orientation='horizontal', density=True)
    ax_right.hist(predictions[top_right], bins=50, alpha=0.5, color='#e74c3c',
                 orientation='horizontal', density=True)
    ax_right.set_xlabel('Density', fontsize=10)
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, alpha=0.2, axis='x')

    # Bottom marginal: distribution comparison by region
    ax_bottom = fig.add_subplot(gs[2, 1], sharex=ax_main)
    regions_data = [measurements_norm[bottom_left], measurements_norm[middle], measurements_norm[top_right]]
    bp = ax_bottom.boxplot(regions_data, vert=False, labels=['Non-binders', 'Noisy', 'Strong binders'],
                           patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], ['#3498db', '#f39c12', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax_bottom.set_xlabel(f'Normalized {measurement_label}', fontsize=10)
    ax_bottom.grid(True, alpha=0.2, axis='x')
    ax_bottom.tick_params(labelbottom=True)

    # Right marginal: distribution comparison by region
    ax_right_bottom = fig.add_subplot(gs[1, 0], sharey=ax_main)
    regions_pred = [predictions[bottom_left], predictions[middle], predictions[top_right]]
    bp2 = ax_right_bottom.boxplot(regions_pred, labels=['Non-\nbinders', 'Noisy', 'Strong\nbinders'],
                                  patch_artist=True, widths=0.6)
    for patch, color in zip(bp2['boxes'], ['#3498db', '#f39c12', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax_right_bottom.set_ylabel('Model Prediction Probability', fontsize=10)
    ax_right_bottom.grid(True, alpha=0.2, axis='y')

    # Hide unused corners
    fig.add_subplot(gs[0, 0]).axis('off')
    fig.add_subplot(gs[0, 2]).axis('off')
    fig.add_subplot(gs[2, 0]).axis('off')
    fig.add_subplot(gs[2, 2]).axis('off')

    plt.savefig(os.path.join(out_dir, f'binding_affinity_correlation_{name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Main correlation plot saved")

    # ============================================================================
    # Additional: 2D density plot
    # ============================================================================

    fig, ax = plt.subplots(figsize=(10, 8))

    # 2D histogram
    h = ax.hist2d(measurements_norm, predictions, bins=50, cmap='YlOrRd', cmin=1)
    plt.colorbar(h[3], ax=ax, label='Count')

    # Add region boundaries
    ax.axhline(y=0.33, color='white', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=0.67, color='white', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=0.33, color='white', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=0.67, color='white', linestyle='--', alpha=0.5, linewidth=1.5)

    # Add diagonal
    ax.plot([0, 1], [0, 1], 'w--', alpha=0.5, linewidth=2)

    ax.set_xlabel(f'Normalized {measurement_label}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Prediction Probability', fontsize=12, fontweight='bold')
    ax.set_title(f'Density Plot: Binding Affinity vs Predictions\n{name}',
                fontsize=14, fontweight='bold')

    # Add correlation text
    textstr = f'Pearson r = {pearson_r:.3f}\nSpearman ρ = {spearman_r:.3f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'binding_affinity_density_{name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Density plot saved")

    # ============================================================================
    # Save statistics
    # ============================================================================

    stats_dict = {
        'dataset': name,
        'n_samples': len(df),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'n_nonbinders': int(bottom_left.sum()),
        'n_noisy': int(middle.sum()),
        'n_strong_binders': int(top_right.sum()),
        'mean_pred_nonbinders': float(predictions[bottom_left].mean()) if bottom_left.sum() > 0 else 0,
        'mean_pred_noisy': float(predictions[middle].mean()) if middle.sum() > 0 else 0,
        'mean_pred_strong_binders': float(predictions[top_right].mean()) if top_right.sum() > 0 else 0,
        'mean_affinity_nonbinders': float(measurements_norm[bottom_left].mean()) if bottom_left.sum() > 0 else 0,
        'mean_affinity_noisy': float(measurements_norm[middle].mean()) if middle.sum() > 0 else 0,
        'mean_affinity_strong_binders': float(measurements_norm[top_right].mean()) if top_right.sum() > 0 else 0,
    }

    stats_df = pd.DataFrame([stats_dict])
    stats_path = os.path.join(out_dir, f'correlation_statistics_{name}.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"✓ Statistics saved to {stats_path}")

    return stats_dict


def main():
    parser = argparse.ArgumentParser(
        description="Inference on binding affinity data with correlation analysis"
    )
    parser.add_argument("--model_weights_path", required=True, help="Path to model weights or .keras file")
    parser.add_argument("--config_path", required=True, help="Path to model configuration JSON")
    parser.add_argument("--df_path", required=True, help="Path to parquet file with binding affinity data")
    parser.add_argument("--measurement_col", required=True,
                       help="Column name containing binding affinity measurements")
    parser.add_argument("--out_dir", required=True, help="Output directory for results")
    parser.add_argument("--name", required=True, help="Dataset name for labeling outputs")
    parser.add_argument("--allele_seq_path", required=True, help="Path to allele sequences CSV")
    parser.add_argument("--embedding_key_path", required=True, help="Path to embedding keys CSV")
    parser.add_argument("--embedding_npz_path", required=True, help="Path to embeddings NPZ file")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference (default: 256)")
    parser.add_argument("--invert_measurements", action='store_true',
                       help="Invert measurements (use for IC50 where lower = stronger binding)")

    args = parser.parse_args()

    # Configure GPU
    if gpus := tf.config.list_physical_devices('GPU'):
        try:
            [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
            print(f"✓ Memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU error: {e}")

    # Enable mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f'✓ Mixed precision enabled: {policy}')

    # Load model and configuration
    model, config, max_pep_len, max_mhc_len = load_model_and_config(
        args.model_weights_path, args.config_path, args.embedding_npz_path
    )

    # Load sequence mappings
    print("\nLoading sequence mappings...")
    seq_map = {clean_key(k): v for k, v in
               pd.read_csv(args.allele_seq_path, index_col="allele")["mhc_sequence"].to_dict().items()}
    embed_map = pd.read_csv(args.embedding_key_path, index_col="key")["mhc_sequence"].to_dict()
    print("✓ Sequence mappings loaded")

    # Load data
    print(f"\nLoading data from {args.df_path}...")
    df = pq.ParquetFile(args.df_path).read().to_pandas()
    print(f"✓ Loaded {len(df)} samples")

    # Check measurement column exists
    if args.measurement_col not in df.columns:
        print(f"\nERROR: Measurement column '{args.measurement_col}' not found in data!")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # Filter out rows with missing measurements
    df_clean = df.dropna(subset=[args.measurement_col])
    print(f"✓ Filtered to {len(df_clean)} samples with valid measurements")

    # Preprocess
    print("\nPreprocessing data...")
    df_clean = preprocess_df(df_clean, seq_map, embed_map)
    print("✓ Data preprocessing complete")

    # Run inference
    df_results = run_inference(model, df_clean, seq_map, embed_map, max_pep_len, max_mhc_len, args.batch_size)

    # Save inference results
    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, f'inference_results_{args.name}.csv')
    df_results.to_csv(results_path, index=False)
    print(f"\n✓ Inference results saved to {results_path}")

    # Generate correlation visualizations
    print("\nGenerating correlation visualizations...")
    stats = visualize_binding_affinity_correlation(
        df_results, args.measurement_col, args.out_dir, args.name, args.invert_measurements
    )

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {args.out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()