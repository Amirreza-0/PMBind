#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mixture of Experts (MoE) Peptide-MHC Binding Prediction
========================================================
This notebook trains separate expert models for groups of similar MHC alleles,
using pseudo sequence clustering to determine allele groupings.

Key Features:
- SubtractLayer for pMHC interaction capture
- Dual-axis attention pooling for anchor finding
- Likelihood-based core selection training
- Hierarchical MHC clustering based on pseudo sequences
- Unseen allele holdout testing
"""

# %% [markdown]
# # 1. Imports and Configuration

# %%
import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, accuracy_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Optional
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using {len(gpus)} GPU(s)")
else:
    print("No GPU found, using CPU")

# Disable interactive plotting for batch mode
import matplotlib
matplotlib.use('Agg')

# %% [markdown]
# # 2. Constants and Hyperparameters

# %%
# ============================================================================
# HYPERPARAMETERS - Modify these as needed
# ============================================================================
NUM_EXPERTS = 5  # Number of expert models (MHC allele clusters)
MAX_PEP_LEN = 15  # Maximum peptide length
MAX_MHC_LEN = 34  # Maximum MHC pseudo sequence length
EMBED_DIM = 64  # Embedding dimension
LATENT_DIM = 32  # Latent dimension
NUM_HEADS = 4  # Attention heads
BATCH_SIZE = 1024
EPOCHS = 5
LEARNING_RATE = 5e-4
DROPOUT_RATE = 0.3
L2_REG = 0.003
PATIENCE = 3  # Early stopping patience
NOISE_STD = 0.1
CORE_TEMPERATURE = 1.0

# Subset for quick pipeline testing (set to None or 0 to use full data)
SUBSET_SIZE = None  # e.g., 10000 for quick test, None for full data

# Unseen test alleles (for holdout evaluation)
UNSEEN_ALLELES = [
    "BOLA-300101", "H-2-KK", "HLA-A30040101",
    "HLA-B2701", "HLA-C17010102", "MAMU-B06601"
]

# Random seed for reproducibility
RANDOM_SEED = 999

# ============================================================================
# OUTPUT DIRECTORY - All outputs saved here
# ============================================================================
from datetime import datetime
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"/scratch-scc/users/u15472/PMBind/src/PY_runs/py6_moe/outputs/run_{RUN_TIMESTAMP}"

# ============================================================================
# CHECKPOINT RESUMPTION - Set to existing run directory to continue training
# ============================================================================
# Set RESUME_FROM_CHECKPOINT to an existing run directory to continue training
# e.g., RESUME_FROM_CHECKPOINT = "/scratch-scc/users/u15472/PMBind/src/PY_runs/py6_moe/outputs/run_20241220_120000"
RESUME_FROM_CHECKPOINT = "/scratch-scc/users/u15472/PMBind/src/PY_runs/py6_moe/outputs/run_20251221_165016" #None  # Set to checkpoint directory path to resume

# If resuming, use the checkpoint directory instead
if RESUME_FROM_CHECKPOINT and os.path.exists(RESUME_FROM_CHECKPOINT):
    OUTPUT_DIR = RESUME_FROM_CHECKPOINT
    print(f"Resuming from checkpoint: {OUTPUT_DIR}")
else:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================================
# AMINO ACID ENCODINGS
# ============================================================================
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_23 = "ARNDCQEGHILKMFPSTWYVBZX"
PAD_TOKEN = -2.0
MASK_TOKEN = -1.0
NORM_TOKEN = 1.0
PAD_VALUE = 0.0
MASK_VALUE = 0.0

# Physicochemical properties (z-score normalized)
PHYSICOCHEMICAL_PROPERTIES = {
    "A": [-0.7115, -1.3796, -0.3237,  1.7644, -0.0249, -1.7046, -1.5572, -0.3873, -0.7860, -0.7889, -0.9536, -0.1956,  0.0770, -0.1703],
    "C": [-1.4936,  0.5087, -0.3881, -1.1136, -0.0977, -0.5661, -0.8431, -0.3873, -0.7860, -0.7889, -1.0949, -1.1785, -0.4755, -0.9901],
    "D": [ 0.9952,  0.3345, -1.7827, -0.2679, -2.0926, -0.1392, -0.6552, -0.3873, -0.7860,  2.8401,  1.0251,  1.6568, -1.8416,  0.8496],
    "E": [ 1.3145, -1.5410,  0.7795,  0.1454, -0.4864,  0.3589,  0.0212, -0.3873, -0.7860,  1.6305,  0.8452,  1.3922, -1.5743,  0.9259],
    "F": [-1.1431, -0.6188,  0.9861, -0.4213,  0.3478,  0.9993,  0.9983,  2.5820, -0.7860, -0.7889, -1.5060, -1.2919, -0.2319, -1.6955],
    "G": [-0.4962,  1.7772,  0.7061,  1.1811,  1.4511, -2.2027, -2.2712, -0.3873, -0.7860, -0.7889,  1.4490,  0.1446,  0.0591,  0.4207],
    "H": [ 0.2526, -0.4339, -0.7929, -1.6181,  0.0205,  0.6435,  0.3595, -0.3873,  0.5531,  0.4208, -1.3904,  0.6739,  1.0214,  0.0394],
    "I": [-1.3855, -0.5728,  1.1059,  0.4566,  0.6176, -0.2104,  0.5850, -0.3873, -0.7860, -0.7889,  0.3955, -1.2919,  0.0888, -1.3905],
    "K": [ 1.8074, -0.5878,  0.3083, -0.2879,  1.1733,  0.3233,  0.9983, -0.3873,  1.8922, -0.7889,  0.4469,  1.0141,  2.2985,  1.1546],
    "L": [-1.1567, -1.0430, -0.7090,  1.4266, -0.5365, -0.2104,  0.5850, -0.3873, -0.7860, -0.7889, -0.8508, -1.4053,  0.0651, -1.2951],
    "M": [-0.7864, -1.6169,  1.1498, -1.0969,  0.8821,  0.4300,  0.5850, -0.3873, -0.7860, -0.7889, -1.7501, -1.1029, -0.0775, -0.9139],
    "N": [ 0.8860,  0.8966,  0.6906, -0.1679,  0.6957, -0.1748, -0.4673, -0.3873,  0.5531,  0.4208,  0.4083,  1.1276, -0.2735,  1.2309],
    "P": [ 0.0997,  2.2356, -0.7704,  0.4877, -0.8570, -0.7796, -0.6928, -0.3873, -0.7860, -0.7889,  0.9994, -0.2334,  0.2552,  0.2205],
    "Q": [ 0.8714, -0.1795, -1.4577, -0.5391, -1.1649,  0.3233,  0.2091, -0.3873,  0.5531,  0.4208,  0.7938,  0.7117, -0.1309,  1.1546],
    "R": [ 1.5027, -0.0470,  0.7919,  0.5088,  2.0074,  1.3194,  1.4869, -0.3873,  3.2313, -0.7889,  1.2563,  0.7117,  2.9044,  1.4692],
    "S": [-0.3340,  1.5068, -2.3337,  0.7644, -1.6952, -1.1354, -1.3317, -0.3873,  0.5531,  0.4208,  0.9737,  0.2203, -0.1131,  0.3063],
    "T": [-0.1301,  0.3601,  1.1468,  1.0288,  0.9495, -0.6373, -0.5801, -0.3873,  0.5531,  0.4208,  0.1642, -0.0066, -0.1606,  0.1156],
    "V": [-1.4874, -0.2864, -0.2293,  1.4000, -0.7702, -0.7084, -0.1291, -0.3873, -0.7860, -0.7889, -0.5810, -1.0273,  0.0532, -0.9806],
    "W": [-0.7157,  0.0214,  0.3776, -2.3448, -0.0503,  2.3868,  2.0506,  2.5820,  0.5531, -0.7889, -1.6217, -1.2163,  0.0116, -1.5716],
    "Y": [ 0.1736,  0.8987,  1.5881, -0.9113,  1.0824,  1.5685,  1.2238,  2.5820,  0.5531,  0.4208, -0.1441, -0.9139, -0.1250, -0.9043],
    "B": [ 0.9411,  0.6155, -0.5463, -0.2179, -0.6988, -0.1570, -0.5613, -0.3873, -0.1164,  1.6305,  0.7167,  1.3922, -1.0576,  1.0403],
    "Z": [ 1.0929, -0.8603, -0.3391, -0.1968, -0.8257,  0.3411,  0.1152, -0.3873, -0.1164,  1.0256,  0.8195,  1.0519, -0.8497,  1.0403],
    "X": [-0.0969,  0.0118,  0.0422,  0.0199,  0.0726, -0.0681, -0.1291, -0.3873, -0.1164, -0.1841, -0.4011, -0.2334,  0.0770, -0.0560],
}

ENCODING_DIM = len(PHYSICOCHEMICAL_PROPERTIES["A"])
AA_TO_INT = {a: i for i, a in enumerate(AA)}
PAD_INDEX_OHE = len(AA)


# %% [markdown]
# # 3. Utility Functions

# %%
def clean_allele_key(allele: str) -> str:
    """Clean allele name by removing special characters."""
    if allele is None:
        return ""
    return allele.strip().replace("*", "").replace(":", "").replace(" ", "").replace("/", "_").upper()


def normalize_allele_for_lookup(allele: str) -> str:
    """
    Normalize allele name for lookup, keeping the HLA-X format but removing * and :.
    E.g., 'HLA-A*01:01' -> 'HLA-A0101'
    """
    if allele is None:
        return ""
    allele = allele.strip()
    # Remove * and : but keep hyphens for HLA-A, HLA-B format
    return allele.replace("*", "").replace(":", "").replace(" ", "").upper()


def find_matching_mhc_key(allele: str, mhc_keys: set, mhc_key_to_original: Dict[str, str] = None) -> Optional[str]:
    """
    Find a matching MHC key for an allele, handling different precision levels.
    
    E.g., 'HLA-A*01:01' should match 'HLA-A01010101' (4-field format)
    
    Args:
        allele: The allele to look up (e.g., 'HLA-A*01:01')
        mhc_keys: Set of available MHC keys (already normalized)
        mhc_key_to_original: Optional mapping from normalized key to original key
        
    Returns:
        Matching key or None
    """
    # First try exact match with clean_allele_key
    cleaned = clean_allele_key(allele)
    if cleaned in mhc_keys:
        return cleaned
    
    # Try normalized format (keeping HLA-A format)
    normalized = normalize_allele_for_lookup(allele)
    normalized_cleaned = clean_allele_key(normalized)
    if normalized_cleaned in mhc_keys:
        return normalized_cleaned
    
    # Try prefix matching for different precision levels
    # HLA-A*01:01 -> look for keys starting with HLAA0101
    for key in mhc_keys:
        if key.startswith(cleaned) or cleaned.startswith(key):
            return key
    
    return None


def seq_to_encoding(sequence: str, max_len: int) -> np.ndarray:
    """Convert sequence to physicochemical encoding."""
    arr = np.full((max_len, ENCODING_DIM), PAD_VALUE, dtype=np.float32)
    default_vec = PHYSICOCHEMICAL_PROPERTIES.get('X', [0.0] * ENCODING_DIM)
    for i, aa in enumerate(sequence.upper()[:max_len]):
        arr[i] = PHYSICOCHEMICAL_PROPERTIES.get(aa, default_vec)
    return arr


def seq_to_onehot(sequence: str, max_len: int) -> np.ndarray:
    """Convert sequence to one-hot encoding for reconstruction target."""
    arr = np.zeros((max_len, 21), dtype=np.float32)
    for i, aa in enumerate(sequence.upper()[:max_len]):
        idx = AA_TO_INT.get(aa, PAD_INDEX_OHE)
        if idx < 21:
            arr[i, idx] = 1.0
    return arr


def create_mask(sequence: str, max_len: int) -> np.ndarray:
    """Create mask array (NORM_TOKEN for valid, PAD_TOKEN for padding)."""
    mask = np.full(max_len, PAD_TOKEN, dtype=np.float32)
    seq_len = min(len(sequence), max_len)
    mask[:seq_len] = NORM_TOKEN
    return mask


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
    """Compute comprehensive classification metrics."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    y_pred_binary_flat = y_pred_binary.flatten()

    metrics = {
        'auc': roc_auc_score(y_true_flat, y_pred_flat) if len(np.unique(y_true_flat)) > 1 else 0.0,
        'auprc': average_precision_score(y_true_flat, y_pred_flat) if len(np.unique(y_true_flat)) > 1 else 0.0,
        'mcc': matthews_corrcoef(y_true_flat, y_pred_binary_flat),
        'accuracy': accuracy_score(y_true_flat, y_pred_binary_flat),
        'precision': precision_score(y_true_flat, y_pred_binary_flat, zero_division=0),
        'recall': recall_score(y_true_flat, y_pred_binary_flat, zero_division=0),
        'f1': f1_score(y_true_flat, y_pred_binary_flat, zero_division=0),
    }

    # AUC at FPR 0.1
    try:
        fpr, tpr, _ = roc_curve(y_true_flat, y_pred_flat)
        idx = np.where(fpr <= 0.1)[0]
        if len(idx) > 0:
            fpr_partial = fpr[idx]
            tpr_partial = tpr[idx]
            if fpr_partial[-1] < 0.1 and idx[-1] + 1 < len(fpr):
                tpr_at_01 = np.interp(0.1, [fpr[idx[-1]], fpr[idx[-1] + 1]],
                                      [tpr[idx[-1]], tpr[idx[-1] + 1]])
                fpr_partial = np.append(fpr_partial, 0.1)
                tpr_partial = np.append(tpr_partial, tpr_at_01)
            metrics['auc_01'] = np.trapz(tpr_partial, fpr_partial) / 0.1
        else:
            metrics['auc_01'] = 0.0
    except:
        metrics['auc_01'] = 0.0

    return metrics


# %% [markdown]
# # 4. Data Loading and Preprocessing

# %%
def load_and_prepare_data(
        data_parquet_path: str,
        mhc_info_csv_path: str,
        benchmark_parquet_path: Optional[str] = None,
        subset_size: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data and prepare train/val/test splits.

    Args:
        data_parquet_path: Path to main dataset parquet
        mhc_info_csv_path: Path to MHC info CSV
        benchmark_parquet_path: Path to benchmark parquet (optional)
        subset_size: If set, take only this many samples for quick testing

    Returns:
        train_df, val_df, unseen_test_df, benchmark_df
    """
    print("Loading data...")

    # Load main dataset
    df = pd.read_parquet(data_parquet_path)
    print(f"  Main dataset: {len(df)} samples")

    # Load MHC info
    mhc_info = pd.read_csv(mhc_info_csv_path)
    print(f"  MHC info: {len(mhc_info)} alleles")

    # Clean allele keys - strip whitespace first
    df['allele'] = df['allele'].str.strip()
    df['allele_key'] = df['allele'].apply(clean_allele_key)
    mhc_info['allele_key'] = mhc_info['key'].apply(clean_allele_key)
    
    # Build lookup structures for flexible matching
    mhc_keys_set = set(mhc_info['allele_key'].unique())
    mhc_key_to_seq = dict(zip(mhc_info['allele_key'], mhc_info['mhc_sequence']))
    mhc_key_to_3di = dict(zip(mhc_info['allele_key'], mhc_info['pseudo_3di']))
    
    # Track samples before merge
    samples_before_merge = len(df)
    unique_alleles_in_data = set(df['allele_key'].unique())
    unique_alleles_in_mhc = set(mhc_info['allele_key'].unique())

    # Merge MHC sequences
    df = df.merge(
        mhc_info[['allele_key', 'mhc_sequence', 'pseudo_3di']].drop_duplicates('allele_key'),
        on='allele_key',
        how='left'
    )

    # Report missing MHC sequences BEFORE dropping
    missing_mask = df['mhc_sequence'].isna()
    missing_count = missing_mask.sum()
    if missing_count > 0:
        missing_alleles = df[missing_mask]['allele_key'].unique()[:10]
        print(f"  Warning: {missing_count} samples missing MHC sequence")
        print(f"    Sample missing allele keys: {list(missing_alleles)}")
        print(f"    Will attempt flexible matching...")
        
        # Try flexible matching for missing alleles
        for idx in df[missing_mask].index:
            allele = df.loc[idx, 'allele']
            matched_key = find_matching_mhc_key(allele, mhc_keys_set)
            if matched_key:
                df.loc[idx, 'mhc_sequence'] = mhc_key_to_seq.get(matched_key)
                df.loc[idx, 'pseudo_3di'] = mhc_key_to_3di.get(matched_key)
                df.loc[idx, 'allele_key'] = matched_key
        
        # Report remaining missing
        still_missing = df['mhc_sequence'].isna().sum()
        if still_missing > 0:
            print(f"    Still missing after flexible matching: {still_missing} samples, dropping...")
        else:
            print(f"    All missing sequences resolved via flexible matching!")

    # Handle missing sequences
    missing_seq = df['mhc_sequence'].isna().sum()
    if missing_seq > 0:
        df = df.dropna(subset=['mhc_sequence'])
    
    samples_after_merge = len(df)
    print(f"  Samples after MHC merge: {samples_after_merge} ({samples_after_merge/samples_before_merge*100:.1f}% retained)")

    # Rename columns for consistency
    if 'long_mer' in df.columns:
        df = df.rename(columns={'long_mer': 'peptide'})

    print(f"  After merge: {len(df)} samples")

    # Apply subset for quick testing
    if subset_size is not None and subset_size > 0 and len(df) > subset_size:
        print(f"  Applying subset: taking {subset_size} samples for quick testing")
        df = df.sample(n=subset_size, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"  After subset: {len(df)} samples")

    # Load benchmark data with flexible allele matching
    benchmark_df = pd.DataFrame()
    if benchmark_parquet_path and os.path.exists(benchmark_parquet_path):
        benchmark_df = pd.read_parquet(benchmark_parquet_path)
        if 'long_mer' in benchmark_df.columns:
            benchmark_df = benchmark_df.rename(columns={'long_mer': 'peptide'})
        benchmark_df['allele'] = benchmark_df['allele'].str.strip()
        benchmark_df['allele_key'] = benchmark_df['allele'].apply(clean_allele_key)
        benchmark_df = benchmark_df.merge(
            mhc_info[['allele_key', 'mhc_sequence', 'pseudo_3di']].drop_duplicates('allele_key'),
            on='allele_key',
            how='left'
        )
        
        # Try flexible matching for missing benchmark alleles
        missing_bench_mask = benchmark_df['mhc_sequence'].isna()
        if missing_bench_mask.sum() > 0:
            print(f"  Benchmark: {missing_bench_mask.sum()} samples with missing MHC, trying flexible match...")
            for idx in benchmark_df[missing_bench_mask].index:
                allele = benchmark_df.loc[idx, 'allele']
                matched_key = find_matching_mhc_key(allele, mhc_keys_set)
                if matched_key:
                    benchmark_df.loc[idx, 'mhc_sequence'] = mhc_key_to_seq.get(matched_key)
                    benchmark_df.loc[idx, 'pseudo_3di'] = mhc_key_to_3di.get(matched_key)
                    benchmark_df.loc[idx, 'allele_key'] = matched_key
        
        benchmark_df = benchmark_df.dropna(subset=['mhc_sequence'])
        print(f"  Benchmark: {len(benchmark_df)} samples")

        # Remove benchmark pairs from main dataset
        benchmark_pairs = set(zip(benchmark_df['allele_key'], benchmark_df['peptide']))
        before_removal = len(df)
        df = df[~df.apply(lambda x: (x['allele_key'], x['peptide']) in benchmark_pairs, axis=1)]
        print(f"  Removed {before_removal - len(df)} benchmark pairs from training data")

    # Split unseen test alleles - use cleaned keys for matching
    unseen_allele_keys = [clean_allele_key(a) for a in UNSEEN_ALLELES]
    unseen_mask = df['allele_key'].isin(unseen_allele_keys)
    unseen_test_df = df[unseen_mask].copy()
    df = df[~unseen_mask].copy()
    
    # Report which unseen alleles were found vs expected
    found_unseen = set(unseen_test_df['allele_key'].unique())
    expected_unseen = set(unseen_allele_keys)
    missing_unseen = expected_unseen - found_unseen
    if missing_unseen:
        print(f"  Warning: Missing unseen test alleles: {missing_unseen}")
    print(f"  Unseen test alleles: {len(unseen_test_df)} samples ({unseen_test_df['allele_key'].nunique()}/{len(UNSEEN_ALLELES)} alleles)")

    # Train/validation split (stratified by label)
    train_df, val_df = train_test_split(
        df, test_size=0.15, stratify=df['assigned_label'],
        random_state=RANDOM_SEED
    )

    print(f"\nFinal splits:")
    print(f"  Train: {len(train_df)} ({train_df['assigned_label'].mean():.2%} positive)")
    print(f"  Val: {len(val_df)} ({val_df['assigned_label'].mean():.2%} positive)")
    print(f"  Unseen test: {len(unseen_test_df)} ({unseen_test_df['assigned_label'].mean():.2%} positive)")
    print(f"  Benchmark: {len(benchmark_df)}")

    return train_df, val_df, unseen_test_df, benchmark_df, mhc_info


# %% [markdown]
# # 5. MHC Clustering

# %%
def cluster_mhc_alleles(
        mhc_info: pd.DataFrame,
        allele_keys: List[str],
        n_clusters: int,
        use_pseudo_3di: bool = False,  # Default to mhc_sequence for better clustering
        method: str = 'kmeans'  # 'kmeans' or 'hierarchical'
) -> Dict[str, int]:
    """
    Cluster MHC alleles based on pseudo sequence similarity.

    Args:
        mhc_info: DataFrame with MHC information
        allele_keys: List of allele keys to cluster
        n_clusters: Number of clusters (experts)
        use_pseudo_3di: Whether to use pseudo_3di or mhc_sequence
        method: Clustering method ('kmeans' for balanced clusters, 'hierarchical')

    Returns:
        Dictionary mapping allele_key to cluster_id
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    
    print(f"\nClustering {len(allele_keys)} MHC alleles into {n_clusters} groups...")

    # Get unique alleles present in data
    mhc_subset = mhc_info[mhc_info['allele_key'].isin(allele_keys)].drop_duplicates('allele_key')

    if len(mhc_subset) < n_clusters:
        print(f"  Warning: Only {len(mhc_subset)} unique alleles, reducing clusters")
        n_clusters = max(1, len(mhc_subset))

    # Get sequences for clustering - prefer mhc_sequence for better diversity
    seq_col = 'pseudo_3di' if use_pseudo_3di and 'pseudo_3di' in mhc_subset.columns else 'mhc_sequence'
    print(f"  Using '{seq_col}' for clustering with method='{method}'")

    # Handle missing sequences
    mhc_subset = mhc_subset.dropna(subset=[seq_col])

    if len(mhc_subset) == 0:
        print("  Error: No valid sequences for clustering")
        return {k: 0 for k in allele_keys}

    allele_list = mhc_subset['allele_key'].tolist()
    sequences = mhc_subset[seq_col].tolist()

    # Encode sequences as feature vectors using physicochemical properties
    def encode_sequence(seq, max_len=50):
        """Encode sequence as flattened physicochemical feature vector."""
        features = []
        for i, aa in enumerate(seq.upper()[:max_len]):
            props = PHYSICOCHEMICAL_PROPERTIES.get(aa, PHYSICOCHEMICAL_PROPERTIES.get('X'))
            features.extend(props)
        # Pad to fixed length
        while len(features) < max_len * ENCODING_DIM:
            features.extend([0.0] * ENCODING_DIM)
        return features[:max_len * ENCODING_DIM]

    # Create feature matrix
    max_seq_len = max(len(s) for s in sequences)
    max_seq_len = min(max_seq_len, 50)  # Cap at 50 for efficiency
    
    feature_matrix = np.array([encode_sequence(seq, max_seq_len) for seq in sequences])
    
    # Standardize features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    print(f"  Feature matrix shape: {feature_matrix_scaled.shape}")

    if method == 'kmeans':
        # K-means produces more balanced clusters
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=RANDOM_SEED, 
            n_init=20,  # Multiple initializations for stability
            max_iter=500
        )
        clusters = kmeans.fit_predict(feature_matrix_scaled)
    else:
        # Hierarchical with Ward linkage for more balanced clusters
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'  # Ward produces more balanced clusters than average
        )
        clusters = clustering.fit_predict(feature_matrix_scaled)

    # Create mapping
    allele_to_cluster = {allele: int(cluster) for allele, cluster in zip(allele_list, clusters)}

    # Assign to nearest cluster for any missing alleles
    for key in allele_keys:
        if key not in allele_to_cluster:
            allele_to_cluster[key] = 0

    # Print cluster distribution
    cluster_counts = {}
    for c in allele_to_cluster.values():
        cluster_counts[c] = cluster_counts.get(c, 0) + 1
    
    # Sort by cluster ID for cleaner output
    cluster_counts = dict(sorted(cluster_counts.items()))
    print(f"  Cluster distribution: {cluster_counts}")
    
    # Print balance statistics
    counts = list(cluster_counts.values())
    print(f"  Cluster sizes: min={min(counts)}, max={max(counts)}, "
          f"std={np.std(counts):.1f}, mean={np.mean(counts):.1f}")

    return allele_to_cluster


# %% [markdown]
# # 6. Custom Layers

# %%
@tf.keras.utils.register_keras_serializable(package='MoE')
class SubtractLayer(keras.layers.Layer):
    """
    Computes pairwise differences between peptide and MHC features.
    Output shape: (B, M, P*D) representing MHC-peptide interactions.
    """

    def __init__(self, mask_token=-1., pad_token=-2., **kwargs):
        super().__init__(**kwargs)
        self.mask_token = mask_token
        self.pad_token = pad_token

    def call(self, peptide, pep_mask, mhc, mhc_mask):
        B = tf.shape(peptide)[0]
        P = tf.shape(peptide)[1]
        D = tf.shape(peptide)[2]
        M = tf.shape(mhc)[1]

        pep_mask = tf.cast(pep_mask, peptide.dtype)
        mhc_mask = tf.cast(mhc_mask, mhc.dtype)
        pep_mask = tf.where(pep_mask == self.pad_token, 0., 1.)
        mhc_mask = tf.where(mhc_mask == self.pad_token, 0., 1.)

        # Expand and compute difference: (B, M, P, D)
        peptide_exp = peptide[:, tf.newaxis, :, :]  # (B, 1, P, D)
        mhc_exp = mhc[:, :, tf.newaxis, :]  # (B, M, 1, D)
        diff = mhc_exp - peptide_exp  # (B, M, P, D)

        # Flatten to (B, M, P*D)
        result = tf.reshape(diff, (B, M, P * D))

        # Create combined mask
        pep_mask_exp = pep_mask[:, tf.newaxis, :, tf.newaxis]  # (B, 1, P, 1)
        mhc_mask_exp = mhc_mask[:, :, tf.newaxis, tf.newaxis]  # (B, M, 1, 1)
        combined_mask = pep_mask_exp * mhc_mask_exp  # (B, M, P, 1)
        combined_mask = tf.broadcast_to(combined_mask, (B, M, P, D))
        combined_mask = tf.reshape(combined_mask, (B, M, P * D))

        return result * combined_mask

    def get_config(self):
        config = super().get_config()
        config.update({'mask_token': self.mask_token, 'pad_token': self.pad_token})
        return config


@tf.keras.utils.register_keras_serializable(package='MoE')
class DualAxisAttentionPooling(keras.layers.Layer):
    """
    Dual-axis attention pooling for finding binding anchors.
    Axis 1: Position attention (which positions matter)
    Axis 2: Feature attention (which features matter)
    """

    def __init__(self, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        latent_dim = input_shape[-1]

        # Position attention
        self.pos_dropout = layers.Dropout(self.dropout_rate)
        self.pos_attn = layers.Dense(1, activation=None, name='position_attention')

        # Feature attention
        self.feat_dropout = layers.Dropout(self.dropout_rate)
        self.feat_attn = layers.Dense(1, activation=None, name='feature_attention')

        super().build(input_shape)

    def call(self, x, mask=None, training=None):
        # x: (B, P, L)

        # Position attention
        x_dropped = self.pos_dropout(x, training=training)
        pos_scores = self.pos_attn(x_dropped)  # (B, P, 1)

        if mask is not None:
            mask_exp = tf.cast(mask != PAD_TOKEN, x.dtype)[:, :, tf.newaxis]
            pos_scores = pos_scores + (1.0 - mask_exp) * (-1e9)

        pos_weights = tf.nn.softmax(pos_scores, axis=1)  # (B, P, 1)
        pos_attended = x * pos_weights + x  # Residual

        # Feature attention (transpose to attend over features)
        x_permuted = tf.transpose(pos_attended, [0, 2, 1])  # (B, L, P)
        x_permuted = self.feat_dropout(x_permuted, training=training)
        feat_scores = self.feat_attn(x_permuted)  # (B, L, 1)
        feat_weights = tf.nn.softmax(feat_scores, axis=1)  # (B, L, 1)
        feat_attended = x_permuted * feat_weights + x_permuted  # Residual

        # Output
        output = tf.transpose(feat_attended, [0, 2, 1])  # (B, P, L)
        pooled = tf.reduce_mean(output, axis=1)  # (B, L)

        return output, pooled, pos_weights

    def get_config(self):
        config = super().get_config()
        config.update({'dropout_rate': self.dropout_rate})
        return config


@tf.keras.utils.register_keras_serializable(package='MoE')
class PositionalEncoding(keras.layers.Layer):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim, max_len=100, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.max_len = max_len

    def build(self, input_shape):
        pos = np.arange(self.max_len)[:, np.newaxis]
        i = np.arange(self.embed_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000., (2 * (i // 2)) / self.embed_dim)
        angle_rads = pos * angle_rates

        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.zeros((self.max_len, self.embed_dim))
        pos_encoding[:, 0::2] = sines
        pos_encoding[:, 1::2] = cosines

        self.pos_encoding = tf.constant(pos_encoding[np.newaxis, :, :], dtype=tf.float32)
        super().build(input_shape)

    def call(self, x, mask=None):
        seq_len = tf.shape(x)[1]
        pe = self.pos_encoding[:, :seq_len, :]
        if mask is not None:
            mask_exp = tf.cast(mask != PAD_TOKEN, x.dtype)[:, :, tf.newaxis]
            pe = pe * mask_exp
        return x + tf.cast(pe, x.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim, 'max_len': self.max_len})
        return config


@tf.keras.utils.register_keras_serializable(package='MoE')
class MaskedEmbedding(keras.layers.Layer):
    """Zero out masked/padded positions."""

    def __init__(self, mask_token=-1., pad_token=-2., **kwargs):
        super().__init__(**kwargs)
        self.mask_token = mask_token
        self.pad_token = pad_token

    def call(self, x, mask):
        mask_float = tf.cast(mask, tf.float32)
        valid_mask = tf.where(
            (mask_float == self.pad_token) | (mask_float == self.mask_token),
            0., 1.
        )
        return x * valid_mask[:, :, tf.newaxis]

    def get_config(self):
        config = super().get_config()
        config.update({'mask_token': self.mask_token, 'pad_token': self.pad_token})
        return config


# %% [markdown]
# # 7. Expert Model Architecture

# %%
def build_expert_model(
        max_pep_len: int = MAX_PEP_LEN,
        max_mhc_len: int = MAX_MHC_LEN,
        pep_dim: int = ENCODING_DIM,
        mhc_dim: int = ENCODING_DIM,
        embed_dim: int = EMBED_DIM,
        latent_dim: int = LATENT_DIM,
        num_heads: int = NUM_HEADS,
        dropout_rate: float = DROPOUT_RATE,
        l2_reg: float = L2_REG,
        noise_std: float = NOISE_STD,
        return_logits: bool = True,
        name: str = "expert"
) -> keras.Model:
    """
    Build a single expert model for pMHC binding prediction.

    Architecture:
    1. Input embedding with positional encoding
    2. SubtractLayer for pMHC interaction
    3. Self-attention on interaction features
    4. Dual-axis attention pooling
    5. Classification head (binding prediction)
    6. Reconstruction head (auxiliary task)
    """
    # Inputs
    pep_in = layers.Input((max_pep_len, pep_dim), name="pep_emb")
    pep_mask_in = layers.Input((max_pep_len,), name="pep_mask")
    mhc_in = layers.Input((max_mhc_len, mhc_dim), name="mhc_emb")
    mhc_mask_in = layers.Input((max_mhc_len,), name="mhc_mask")
    pep_target_in = layers.Input((max_pep_len, 21), name="pep_ohe_target")

    # === Peptide Processing ===
    pep = MaskedEmbedding(name=f"{name}_pep_mask_embed")(pep_in, pep_mask_in)
    pep = PositionalEncoding(pep_dim, max_pep_len * 2, name=f"{name}_pep_pos")(pep, pep_mask_in)
    pep = layers.GaussianNoise(noise_std, name=f"{name}_pep_noise")(pep)
    pep = layers.SpatialDropout1D(dropout_rate, name=f"{name}_pep_dropout")(pep)

    # === MHC Processing ===
    mhc = MaskedEmbedding(name=f"{name}_mhc_mask_embed")(mhc_in, mhc_mask_in)
    mhc = PositionalEncoding(mhc_dim, max_mhc_len * 2, name=f"{name}_mhc_pos")(mhc, mhc_mask_in)
    mhc = layers.GaussianNoise(noise_std, name=f"{name}_mhc_noise")(mhc)
    mhc = layers.SpatialDropout1D(dropout_rate, name=f"{name}_mhc_dropout")(mhc)

    # === SubtractLayer for Interaction ===
    # Output: (B, M, P*D)
    interaction = SubtractLayer(name=f"{name}_subtract")(pep, pep_mask_in, mhc, mhc_mask_in)
    interaction = layers.LayerNormalization(name=f"{name}_interact_norm")(interaction)

    # === Self-Attention on Interaction ===
    interaction_proj = layers.Dense(
        embed_dim, activation='gelu',
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name=f"{name}_interact_proj"
    )(interaction)

    # Multi-head self-attention
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads,
        dropout=dropout_rate,
        name=f"{name}_mha"
    )(interaction_proj, interaction_proj)
    attn_out = layers.Add(name=f"{name}_attn_residual")([interaction_proj, attn_out])
    attn_out = layers.LayerNormalization(name=f"{name}_attn_norm")(attn_out)

    # === Reshape for Peptide-centric View ===
    # (B, M, embed_dim) -> (B, M, P, embed_dim/P) -> mean over M -> (B, P, dim)
    # Alternative: project directly to latent sequence
    attn_pooled = layers.GlobalAveragePooling1D(name=f"{name}_attn_pool")(attn_out)
    latent_proj = layers.Dense(
        latent_dim * max_pep_len,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name=f"{name}_latent_proj"
    )(attn_pooled)  # (B, latent_dim * P)
    latent_seq = layers.Reshape((max_pep_len, latent_dim), name=f"{name}_latent_reshape")(latent_proj)

    # === Dual-Axis Attention Pooling ===
    attn_output, pooled_latent, pos_weights = DualAxisAttentionPooling(
        dropout_rate=dropout_rate, name=f"{name}_dual_attn"
    )(latent_seq, pep_mask_in)

    # === Classification Head ===
    cls_head = layers.Dropout(dropout_rate * 1.5, name=f"{name}_cls_dropout1")(pooled_latent)
    cls_head = layers.Dense(
        latent_dim,
        activation='gelu',
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        name=f"{name}_cls_dense1"
    )(cls_head)
    cls_head = layers.Dropout(dropout_rate, name=f"{name}_cls_dropout2")(cls_head)

    binding_activation = None if return_logits else 'sigmoid'
    binding_pred = layers.Dense(
        1, activation=binding_activation,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
        name=f"{name}_binding_pred",
        dtype='float32'
    )(cls_head)

    # === Reconstruction Head (Auxiliary) ===
    recon_head = layers.SpatialDropout1D(dropout_rate, name=f"{name}_recon_dropout")(latent_seq)
    recon_head = layers.Dense(embed_dim, activation='gelu', name=f"{name}_recon_dense")(recon_head)
    recon_pred = layers.Dense(21, activation='softmax', name=f"{name}_recon_pred")(recon_head)

    # Concatenate target and prediction for loss computation
    pep_recon_out = layers.Concatenate(name=f"{name}_recon_concat")([pep_target_in, recon_pred])

    # === Core Selection Outputs (for likelihood training) ===
    core_logits = layers.Dense(
        1, activation=None, name=f"{name}_core_logits", dtype='float32'
    )(pooled_latent)

    model = keras.Model(
        inputs=[pep_in, pep_mask_in, mhc_in, mhc_mask_in, pep_target_in],
        outputs={
            'binding_pred': binding_pred,
            'pep_recon': pep_recon_out,
            'latent_vector': pooled_latent,
            'latent_seq': latent_seq,
            'core_logits': core_logits,
            'pos_weights': pos_weights,
        },
        name=f"Expert_{name}"
    )

    return model


# %% [markdown]
# # 8. Loss Functions and Training Utilities

# %%
@tf.function
def masked_categorical_crossentropy(y_true_pred, mask, pad_token=PAD_TOKEN):
    """Compute masked reconstruction loss."""
    y_true, y_pred = tf.split(y_true_pred, 2, axis=-1)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

    mask_float = tf.cast(tf.not_equal(mask, pad_token), tf.float32)
    log_pred = tf.math.log(y_pred)
    log_pred = tf.where(tf.math.is_finite(log_pred), log_pred, tf.zeros_like(log_pred))

    loss = -tf.reduce_sum(y_true * log_pred, axis=-1)
    loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))

    masked_loss = loss * mask_float
    total_loss = tf.reduce_sum(masked_loss)
    total_weight = tf.reduce_sum(mask_float)

    return tf.math.divide_no_nan(total_loss, total_weight)

@tf.function
def likelihood_core_loss(
    model, peptide, mhc, labels, pep_mask, mhc_mask,
    max_pep_len, temperature=1.0, min_core_len=8, training=True,
    class_weights=None
):
    """
    Likelihood-based loss for core selection and binding prediction.
    Marginalizes over possible binding cores.
    
    Args:
    model: The expert model
    peptide: Peptide embeddings (B, P, D)
    mhc: MHC embeddings (B, M, D)
    labels: Binary labels (B, 1)
    pep_mask: Peptide mask (B, P)
    mhc_mask: MHC mask (B, M)
    max_pep_len: Maximum peptide length
    temperature: Softmax temperature for core selection
    min_core_len: Minimum core length to consider
    training: Whether in training mode
    class_weights: Optional dict with {0: weight0, 1: weight1} for class balancing
    """
    B = tf.shape(peptide)[0]
    P = tf.shape(peptide)[1]
    D = tf.shape(peptide)[2]
    M = tf.shape(mhc)[1]

    cores_list = []
    masks_list = []
    valid_flags_list = []

    # Extract all possible cores
    for k in range(min_core_len, max_pep_len + 1):
        num_windows = max_pep_len - k + 1
        if num_windows <= 0:
            continue

        for i in range(num_windows):
            # Extract core window
            core = peptide[:, i:i + k, :]

            # A window is valid only if all positions are non-pad
            window_mask = pep_mask[:, i:i + k]
            is_valid = tf.reduce_all(tf.not_equal(window_mask, PAD_TOKEN), axis=1)
            valid_flags_list.append(is_valid)

            # Pad to max_pep_len
            paddings = [[0, 0], [0, max_pep_len - k], [0, 0]]
            padded_core = tf.pad(core, paddings, constant_values=PAD_VALUE)
            cores_list.append(padded_core)

            # Create mask
            mask_vec = tf.concat([
                tf.fill([k], NORM_TOKEN),
                tf.fill([max_pep_len - k], PAD_TOKEN)
            ], axis=0)
            mask_batch = tf.tile(tf.expand_dims(mask_vec, 0), [B, 1])
            masks_list.append(mask_batch)

    if not cores_list:
        return tf.constant(0.0), tf.zeros((B, 0)), tf.zeros((B, 0))

    # Stack
    cores_stack = tf.stack(cores_list, axis=1)  # (B, N, P, D)
    masks_stack = tf.stack(masks_list, axis=1)  # (B, N, P)
    valid_flags = tf.stack(valid_flags_list, axis=1)  # (B, N)
    valid_float = tf.cast(valid_flags, tf.float32)
    has_valid = tf.reduce_any(valid_flags, axis=1)  # (B,)

    N = tf.shape(cores_stack)[1]

    # Flatten for batch processing
    cores_flat = tf.reshape(cores_stack, [B * N, max_pep_len, D])
    masks_flat = tf.reshape(masks_stack, [B * N, max_pep_len])

    mhc_tiled = tf.tile(tf.expand_dims(mhc, 1), [1, N, 1, 1])
    mhc_flat = tf.reshape(mhc_tiled, [B * N, M, D])

    mhc_mask_tiled = tf.tile(tf.expand_dims(mhc_mask, 1), [1, N, 1])
    mhc_mask_flat = tf.reshape(mhc_mask_tiled, [B * N, M])

    dummy_target = tf.zeros((B * N, max_pep_len, 21), dtype=tf.float32)

    # Forward pass
    outputs = model([cores_flat, masks_flat, mhc_flat, mhc_mask_flat, dummy_target], training=training)

    core_logits = tf.reshape(outputs['core_logits'], [B, N])
    binding_logits = tf.reshape(outputs['binding_pred'], [B, N])

    # Core selection distribution
    core_logits_masked = tf.where(valid_flags, core_logits, tf.fill(tf.shape(core_logits), -1e9))
    attention_weights = tf.nn.softmax(core_logits_masked / temperature, axis=1)
    attention_weights = tf.where(
        tf.expand_dims(has_valid, 1),
        attention_weights,
        tf.zeros_like(attention_weights)
    )
    # Renormalize in case some windows were zeroed out
    attn_sums = tf.reduce_sum(attention_weights, axis=1, keepdims=True)
    attention_weights = tf.math.divide_no_nan(attention_weights, attn_sums)

    # Per-core binding probability
    binding_probs = tf.nn.sigmoid(binding_logits)
    binding_probs = binding_probs * valid_float

    # Marginal binding probability
    pred_prob = tf.reduce_sum(attention_weights * binding_probs, axis=1)
    # If no valid windows, keep prediction at zero confidence
    pred_prob = tf.where(has_valid, pred_prob, tf.zeros_like(pred_prob))
    pred_prob = tf.clip_by_value(pred_prob, 1e-6, 1.0 - 1e-6)

    # BCE loss
    label_flat = tf.cast(tf.reshape(labels, [B]), tf.float32)
    loss = tf.keras.backend.binary_crossentropy(label_flat, pred_prob)

    # Apply class weights if provided
    if class_weights is not None:
        w0 = tf.cast(class_weights[0], tf.float32)
        w1 = tf.cast(class_weights[1], tf.float32)
        sample_weights = tf.where(label_flat > 0.5, w1, w0)
        loss = loss * sample_weights

    # Down-weight positive samples with no valid cores so we do not force
    # high confidence when every window is invalid; keep non-binders active.
    validity_weight = tf.where(
        has_valid,
        tf.ones_like(label_flat),
        tf.where(label_flat > 0.5, tf.zeros_like(label_flat), tf.ones_like(label_flat))
    )
    loss = loss * validity_weight

    return tf.reduce_mean(loss), attention_weights, binding_probs


class ExpertTrainer:
    """Trainer class for a single expert model."""

    def __init__(self, model, learning_rate=LEARNING_RATE, cls_weight=1.0, recon_weight=0.5,
                 use_likelihood=True, core_temperature=CORE_TEMPERATURE, min_core_len=8):
        self.model = model
        self.cls_weight = cls_weight
        self.recon_weight = recon_weight
        self.use_likelihood = use_likelihood
        self.core_temperature = core_temperature
        self.min_core_len = min_core_len

        # Optimizer with learning rate schedule
        self.lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=learning_rate,
            first_decay_steps=1000,
            alpha=0.1
        )
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_schedule)

        # Metrics
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.val_loss = keras.metrics.Mean(name='val_loss')
        self.train_auc = keras.metrics.AUC(name='train_auc')
        self.val_auc = keras.metrics.AUC(name='val_auc')
        
        # For MCC computation (collected per epoch)
        self.train_preds = []
        self.train_labels = []

    def train_step(self, batch, class_weights=None):
        """Single training step with optional likelihood-based core loss."""
        pep_emb = batch['pep_emb']
        pep_mask = batch['pep_mask']
        mhc_emb = batch['mhc_emb']
        mhc_mask = batch['mhc_mask']
        pep_target = batch['pep_ohe_target']
        labels = batch['labels']

        with tf.GradientTape() as tape:
            outputs = self.model(
                [pep_emb, pep_mask, mhc_emb, mhc_mask, pep_target],
                training=True
            )

            # Reconstruction loss
            recon_loss = masked_categorical_crossentropy(outputs['pep_recon'], pep_mask)

            if self.use_likelihood:
                # Likelihood-based core selection loss
                likelihood_loss, attention_weights, binding_probs = likelihood_core_loss(
                    model=self.model,
                    peptide=pep_emb,
                    mhc=mhc_emb,
                    labels=labels,
                    pep_mask=pep_mask,
                    mhc_mask=mhc_mask,
                    max_pep_len=MAX_PEP_LEN,
                    temperature=self.core_temperature,
                    min_core_len=self.min_core_len,
                    training=True
                )
                
                # Also compute direct loss for weighting if class weights provided
                # direct_loss = tf.keras.losses.binary_crossentropy(
                #     labels, outputs['binding_pred'], from_logits=True
                # )
                # if class_weights is not None:
                #     w0 = tf.cast(class_weights[0], tf.float32)
                #     w1 = tf.cast(class_weights[1], tf.float32)
                #     weights = tf.where(labels > 0.5, w1, w0)
                #     direct_loss = direct_loss * tf.squeeze(weights)
                # direct_loss = tf.reduce_mean(direct_loss)
                
                # Combine likelihood loss with direct loss (weighted)
                # cls_loss = 0.7 * likelihood_loss + 0.3 * direct_loss
                cls_loss = likelihood_loss
            else:
                # Classification loss (direct only)
                direct_loss = tf.keras.losses.binary_crossentropy(
                    labels, outputs['binding_pred'], from_logits=True
                )

                if class_weights is not None:
                    # Cast class weights to float32 to match loss dtype
                    w0 = tf.cast(class_weights[0], tf.float32)
                    w1 = tf.cast(class_weights[1], tf.float32)
                    weights = tf.where(labels > 0.5, w1, w0)
                    direct_loss = direct_loss * tf.squeeze(weights)

                cls_loss = tf.reduce_mean(direct_loss)

            # Total loss
            total_loss = self.cls_weight * cls_loss + self.recon_weight * recon_loss

        # Gradient update
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Update metrics
        probs = tf.nn.sigmoid(outputs['binding_pred'])
        self.train_loss.update_state(total_loss)
        self.train_auc.update_state(labels, probs)
        
        # Collect for MCC computation
        self.train_preds.extend(probs.numpy().flatten())
        self.train_labels.extend(labels.numpy().flatten())

        return total_loss, recon_loss, cls_loss, probs

    @tf.function
    def val_step(self, batch):
        """Single validation step."""
        pep_emb = batch['pep_emb']
        pep_mask = batch['pep_mask']
        mhc_emb = batch['mhc_emb']
        mhc_mask = batch['mhc_mask']
        pep_target = batch['pep_ohe_target']
        labels = batch['labels']

        outputs = self.model(
            [pep_emb, pep_mask, mhc_emb, mhc_mask, pep_target],
            training=False
        )

        recon_loss = masked_categorical_crossentropy(outputs['pep_recon'], pep_mask)
        direct_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
            labels, outputs['binding_pred'], from_logits=True
        ))
        total_loss = self.cls_weight * direct_loss + self.recon_weight * recon_loss

        probs = tf.nn.sigmoid(outputs['binding_pred'])
        self.val_loss.update_state(total_loss)
        self.val_auc.update_state(labels, probs)

        return total_loss, probs

    def reset_metrics(self):
        self.train_loss.reset_state()
        self.val_loss.reset_state()
        self.train_auc.reset_state()
        self.val_auc.reset_state()
        self.train_preds = []
        self.train_labels = []
    
    def compute_train_mcc(self):
        """Compute training MCC from collected predictions."""
        if len(self.train_preds) == 0:
            return 0.0
        preds = np.array(self.train_preds)
        labels = np.array(self.train_labels)
        pred_binary = (preds > 0.5).astype(int)
        return matthews_corrcoef(labels, pred_binary)


# %% [markdown]
# # 9. Data Generator

# %%
class DataGenerator(keras.utils.Sequence):
    """Data generator for batching and preprocessing."""

    def __init__(self, df, batch_size, shuffle=True, augment=False):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(df))

        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]
        return self._generate_batch(batch_indices)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _generate_batch(self, indices):
        n = len(indices)

        data = {
            'pep_emb': np.zeros((n, MAX_PEP_LEN, ENCODING_DIM), np.float32),
            'pep_mask': np.full((n, MAX_PEP_LEN), PAD_TOKEN, np.float32),
            'mhc_emb': np.zeros((n, MAX_MHC_LEN, ENCODING_DIM), np.float32),
            'mhc_mask': np.full((n, MAX_MHC_LEN), PAD_TOKEN, np.float32),
            'pep_ohe_target': np.zeros((n, MAX_PEP_LEN, 21), np.float32),
            'labels': np.zeros((n, 1), np.float32),
        }

        for i, idx in enumerate(indices):
            row = self.df.iloc[idx]
            pep_seq = str(row['peptide']).upper()
            mhc_seq = str(row['mhc_sequence']).upper()

            # Encode
            data['pep_emb'][i] = seq_to_encoding(pep_seq, MAX_PEP_LEN)
            data['pep_mask'][i] = create_mask(pep_seq, MAX_PEP_LEN)
            data['pep_ohe_target'][i] = seq_to_onehot(pep_seq, MAX_PEP_LEN)
            data['mhc_emb'][i] = seq_to_encoding(mhc_seq, MAX_MHC_LEN)
            data['mhc_mask'][i] = create_mask(mhc_seq, MAX_MHC_LEN)
            data['labels'][i, 0] = float(row['assigned_label'])

        # Optional augmentation (masking)
        if self.augment:
            data = self._apply_masking(data)

        return {k: tf.convert_to_tensor(v) for k, v in data.items()}

    def _apply_masking(self, data):
        """Apply random masking for data augmentation."""
        for i in range(len(data['pep_emb'])):
            # Peptide masking (15% of valid positions)
            valid_pos = np.where(data['pep_mask'][i] == NORM_TOKEN)[0]
            if len(valid_pos) > 2:
                n_mask = max(1, int(len(valid_pos) * 0.15))
                mask_pos = np.random.choice(valid_pos, n_mask, replace=False)
                data['pep_mask'][i, mask_pos] = MASK_TOKEN
                data['pep_emb'][i, mask_pos] = MASK_VALUE

            # MHC masking (15% of valid positions)
            valid_pos = np.where(data['mhc_mask'][i] == NORM_TOKEN)[0]
            if len(valid_pos) > 5:
                n_mask = max(2, int(len(valid_pos) * 0.15))
                mask_pos = np.random.choice(valid_pos, n_mask, replace=False)
                data['mhc_mask'][i, mask_pos] = MASK_TOKEN
                data['mhc_emb'][i, mask_pos] = MASK_VALUE

        return data


# %% [markdown]
# # 10. Mixture of Experts Training

# %%
def load_checkpoint_state(checkpoint_dir: str) -> Optional[Dict]:
    """
    Load checkpoint state from a directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        Dictionary with checkpoint state or None if no checkpoint found
    """
    checkpoint_file = os.path.join(checkpoint_dir, "checkpoints", "training_state.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            state = json.load(f)
        print(f"Loaded checkpoint state from {checkpoint_file}")
        return state
    return None


def save_checkpoint_state(checkpoint_dir: str, state: Dict):
    """
    Save checkpoint state to a directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        state: Dictionary with checkpoint state
    """
    os.makedirs(os.path.join(checkpoint_dir, "checkpoints"), exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, "checkpoints", "training_state.json")
    with open(checkpoint_file, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"Saved checkpoint state to {checkpoint_file}")


class MixtureOfExpertsTrainer:
    """
    Trains multiple expert models, each specialized for a cluster of MHC alleles.
    Supports checkpoint resumption for continuing training.
    """

    def __init__(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            mhc_info: pd.DataFrame,
            num_experts: int = NUM_EXPERTS,
            batch_size: int = BATCH_SIZE,
            epochs: int = EPOCHS,
            patience: int = PATIENCE,
            resume_from: Optional[str] = None
    ):
        self.train_df = train_df
        self.val_df = val_df
        self.mhc_info = mhc_info
        self.num_experts = num_experts
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.resume_from = resume_from
        
        # Try to load checkpoint state
        self.checkpoint_state = None
        if resume_from and os.path.exists(resume_from):
            self.checkpoint_state = load_checkpoint_state(resume_from)
            if self.checkpoint_state:
                print(f"Found checkpoint: experts completed = {self.checkpoint_state.get('completed_experts', [])}")

        # Load allele_to_cluster from checkpoint if available
        if self.checkpoint_state and 'allele_to_cluster' in self.checkpoint_state:
            self.allele_to_cluster = self.checkpoint_state['allele_to_cluster']
            print(f"Loaded allele_to_cluster from checkpoint ({len(self.allele_to_cluster)} alleles)")
        elif resume_from:
            # Try loading from saved mapping file
            mapping_file = os.path.join(resume_from, 'allele_to_cluster.json')
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    self.allele_to_cluster = json.load(f)
                print(f"Loaded allele_to_cluster from {mapping_file}")
            else:
                # Cluster alleles fresh
                all_alleles = list(set(train_df['allele_key'].unique()) | set(val_df['allele_key'].unique()))
                self.allele_to_cluster = cluster_mhc_alleles(mhc_info, all_alleles, num_experts)
        else:
            # Cluster alleles fresh
            all_alleles = list(set(train_df['allele_key'].unique()) | set(val_df['allele_key'].unique()))
            self.allele_to_cluster = cluster_mhc_alleles(mhc_info, all_alleles, num_experts)

        # Split data by cluster
        self.train_by_cluster = self._split_by_cluster(train_df)
        self.val_by_cluster = self._split_by_cluster(val_df)

        # Build experts
        self.experts = {}
        self.trainers = {}
        
        # Track completed epochs per expert (for continuing training)
        self.completed_epochs = {}  # cluster_id -> num_epochs_completed
        if self.checkpoint_state and 'history' in self.checkpoint_state:
            # Determine how many epochs each expert has completed from history
            for cid_str, hist in self.checkpoint_state['history'].items():
                cid = int(cid_str)
                self.completed_epochs[cid] = len(hist.get('train_loss', []))
            print(f"Completed epochs per expert: {self.completed_epochs}")

        for cluster_id in range(num_experts):
            if cluster_id in self.train_by_cluster and len(self.train_by_cluster[cluster_id]) > 0:
                expert = build_expert_model(name=f"cluster{cluster_id}")
                
                # Try to load weights from checkpoint
                if resume_from:
                    model_path = os.path.join(resume_from, "models", f"expert_{cluster_id}_best_model.keras")
                    weight_path = os.path.join(resume_from, "models", f"expert_{cluster_id}_best.weights.h5")
                    if os.path.exists(model_path):
                        try:
                            expert = keras.models.load_model(model_path, custom_objects={
                                'SubtractLayer': SubtractLayer,
                                'DualAxisAttentionPooling': DualAxisAttentionPooling,
                                'PositionalEncoding': PositionalEncoding,
                                'MaskedEmbedding': MaskedEmbedding,
                            })
                            print(f"  Loaded expert {cluster_id} from {model_path}")
                        except Exception as e:
                            print(f"  Warning: Could not load model for expert {cluster_id}: {e}")
                            if os.path.exists(weight_path):
                                try:
                                    expert.load_weights(weight_path)
                                    print(f"  Loaded weights for expert {cluster_id} from {weight_path}")
                                except Exception as e2:
                                    print(f"  Warning: Could not load weights either: {e2}")
                    elif os.path.exists(weight_path):
                        try:
                            expert.load_weights(weight_path)
                            print(f"  Loaded weights for expert {cluster_id} from {weight_path}")
                        except Exception as e:
                            print(f"  Warning: Could not load weights for expert {cluster_id}: {e}")
                
                self.experts[cluster_id] = expert
                self.trainers[cluster_id] = ExpertTrainer(
                    expert, 
                    use_likelihood=True,
                    core_temperature=CORE_TEMPERATURE,
                    min_core_len=8
                )
                print(f"Expert {cluster_id}: {len(self.train_by_cluster[cluster_id])} train, "
                      f"{len(self.val_by_cluster.get(cluster_id, []))} val samples")

        # Initialize history, loading from checkpoint if available
        self.history = {cid: {'train_loss': [], 'train_auc': [], 'train_mcc': [],
                               'val_loss': [], 'val_auc': [], 'val_mcc': []}
                        for cid in self.experts.keys()}
        if self.checkpoint_state and 'history' in self.checkpoint_state:
            for cid_str, hist in self.checkpoint_state['history'].items():
                cid = int(cid_str)
                if cid in self.history:
                    self.history[cid] = hist
                    print(f"  Loaded history for expert {cid}: {len(hist.get('train_loss', []))} epochs")

    def _split_by_cluster(self, df):
        """Split DataFrame by cluster assignment."""
        clusters = {}
        df = df.copy()
        # Use .get() with default 0 to handle any alleles not in the mapping
        df['cluster'] = df['allele_key'].map(lambda x: self.allele_to_cluster.get(x, 0))

        for cid in range(self.num_experts):
            cluster_df = df[df['cluster'] == cid]
            if len(cluster_df) > 0:
                clusters[cid] = cluster_df

        return clusters
    
    def _save_checkpoint(self):
        """Save checkpoint state for resumption."""
        state = {
            'completed_epochs': self.completed_epochs,
            'history': {str(k): v for k, v in self.history.items()},
            'allele_to_cluster': self.allele_to_cluster,
            'num_experts': self.num_experts,
            'epochs': self.epochs,
        }
        save_checkpoint_state(OUTPUT_DIR, state)

    def train(self):
        """Train all experts with checkpoint support."""
        print("\n" + "=" * 60)
        print("Starting Mixture of Experts Training")
        if self.completed_epochs:
            print(f"Resuming - completed epochs per expert: {self.completed_epochs}")
        print(f"Target epochs: {self.epochs}")
        print("=" * 60)

        for cluster_id, expert in self.experts.items():
            # Check how many epochs this expert has already completed
            start_epoch = self.completed_epochs.get(cluster_id, 0)
            
            # Skip if already trained for requested number of epochs
            if start_epoch >= self.epochs:
                print(f"\n{'=' * 60}")
                print(f"Skipping Expert {cluster_id} (already completed {start_epoch}/{self.epochs} epochs)")
                print(f"{'=' * 60}")
                continue
            
            print(f"\n{'=' * 60}")
            print(f"Training Expert {cluster_id} (epochs {start_epoch + 1} to {self.epochs})")
            print(f"{'=' * 60}")

            train_df = self.train_by_cluster[cluster_id]
            val_df = self.val_by_cluster.get(cluster_id, pd.DataFrame())

            if len(val_df) == 0:
                # Use part of training for validation
                train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=RANDOM_SEED)

            # Compute class weights (handle single-class case)
            labels = train_df['assigned_label'].values
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                # Single class in this cluster - use equal weights
                print(f"  Warning: Expert {cluster_id} has only class {unique_labels[0]}, using equal weights")
                class_weights = {0: 1.0, 1: 1.0}
            else:
                class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=labels)
                class_weights = {0: class_weights[0], 1: class_weights[1]}

            # Create generators
            train_gen = DataGenerator(train_df, self.batch_size, shuffle=True, augment=True)
            val_gen = DataGenerator(val_df, self.batch_size, shuffle=False, augment=False)

            trainer = self.trainers[cluster_id]
            
            # Initialize best_mcc from previous training if resuming
            if start_epoch > 0 and cluster_id in self.history and self.history[cluster_id]['val_mcc']:
                best_mcc = max(self.history[cluster_id]['val_mcc'])
                print(f"  Resuming from epoch {start_epoch}, previous best MCC: {best_mcc:.4f}")
            else:
                best_mcc = -1
            patience_counter = 0

            for epoch in range(start_epoch, self.epochs):
                trainer.reset_metrics()
                epoch_start = time.time()

                # Training
                pbar = tqdm(train_gen, desc=f"Epoch {epoch + 1}/{self.epochs} (Expert {cluster_id})")
                for batch in pbar:
                    loss, recon_loss, cls_loss, _ = trainer.train_step(batch, class_weights)
                    pbar.set_postfix({
                        'loss': f'{trainer.train_loss.result():.4f}',
                        'auc': f'{trainer.train_auc.result():.4f}'
                    })

                # Compute train MCC
                train_mcc = trainer.compute_train_mcc()

                # Validation
                val_preds = []
                val_labels = []
                for batch in val_gen:
                    _, probs = trainer.val_step(batch)
                    val_preds.extend(probs.numpy().flatten())
                    val_labels.extend(batch['labels'].numpy().flatten())

                # Compute metrics
                val_preds = np.array(val_preds)
                val_labels = np.array(val_labels)
                metrics = compute_metrics(val_labels, val_preds)
                epoch_time = time.time() - epoch_start

                # Log all metrics
                self.history[cluster_id]['train_loss'].append(float(trainer.train_loss.result()))
                self.history[cluster_id]['train_auc'].append(float(trainer.train_auc.result()))
                self.history[cluster_id]['train_mcc'].append(train_mcc)
                self.history[cluster_id]['val_loss'].append(float(trainer.val_loss.result()))
                self.history[cluster_id]['val_auc'].append(metrics['auc'])
                self.history[cluster_id]['val_mcc'].append(metrics['mcc'])

                # Print detailed validation metrics
                print(f"\n  {'─' * 50}")
                print(f"  Epoch {epoch + 1}/{self.epochs} Summary (took {epoch_time:.1f}s)")
                print(f"  {'─' * 50}")
                print(f"  Train: loss={trainer.train_loss.result():.4f}, AUC={trainer.train_auc.result():.4f}, MCC={train_mcc:.4f}")
                print(f"  Valid: loss={trainer.val_loss.result():.4f}, AUC={metrics['auc']:.4f}, MCC={metrics['mcc']:.4f}")
                print(f"         Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
                print(f"  {'─' * 50}")

                # Early stopping
                if metrics['mcc'] > best_mcc:
                    best_mcc = metrics['mcc']
                    patience_counter = 0
                    # Save best weights to output directory
                    weight_path = os.path.join(OUTPUT_DIR, "models", f"expert_{cluster_id}_best.weights.h5")
                    expert.save_weights(weight_path)
                    # Save full model as well
                    model_path = os.path.join(OUTPUT_DIR, "models", f"expert_{cluster_id}_best_model.keras")
                    expert.save(model_path)
                    print(f"  Saved best model (MCC={best_mcc:.4f}) to {model_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"  Early stopping at epoch {epoch + 1}")
                        break

                train_gen.on_epoch_end()
                
                # Update completed epochs and save checkpoint after each epoch
                self.completed_epochs[cluster_id] = epoch + 1
                self._save_checkpoint()

            # Load best weights
            try:
                weight_path = os.path.join(OUTPUT_DIR, "models", f"expert_{cluster_id}_best.weights.h5")
                expert.load_weights(weight_path)
            except:
                pass

            print(f"Expert {cluster_id} best MCC: {best_mcc:.4f}")
            print(f"  Completed {self.completed_epochs.get(cluster_id, 0)}/{self.epochs} epochs")

        return self.history

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions using appropriate expert for each sample."""
        df = df.copy()
        df['cluster'] = df['allele_key'].map(lambda x: self.allele_to_cluster.get(x, 0))

        predictions = np.zeros(len(df))

        for cluster_id, expert in self.experts.items():
            cluster_mask = df['cluster'] == cluster_id
            if not cluster_mask.any():
                continue

            cluster_df = df[cluster_mask]
            gen = DataGenerator(cluster_df, self.batch_size, shuffle=False, augment=False)

            preds = []
            for batch in gen:
                outputs = expert([
                    batch['pep_emb'], batch['pep_mask'],
                    batch['mhc_emb'], batch['mhc_mask'],
                    batch['pep_ohe_target']
                ], training=False)
                probs = tf.nn.sigmoid(outputs['binding_pred']).numpy().flatten()
                preds.extend(probs)

            predictions[cluster_mask.values] = preds[:cluster_mask.sum()]

        # Handle samples with no matching expert (use expert 0)
        no_expert_mask = ~df['cluster'].isin(self.experts.keys())
        if no_expert_mask.any() and 0 in self.experts:
            fallback_df = df[no_expert_mask]
            gen = DataGenerator(fallback_df, self.batch_size, shuffle=False, augment=False)

            preds = []
            for batch in gen:
                outputs = self.experts[0]([
                    batch['pep_emb'], batch['pep_mask'],
                    batch['mhc_emb'], batch['mhc_mask'],
                    batch['pep_ohe_target']
                ], training=False)
                probs = tf.nn.sigmoid(outputs['binding_pred']).numpy().flatten()
                preds.extend(probs)

            predictions[no_expert_mask.values] = preds[:no_expert_mask.sum()]

        return predictions

    def evaluate(self, df: pd.DataFrame, name: str = "Test") -> Dict[str, float]:
        """Evaluate on a dataset."""
        if len(df) == 0:
            print(f"{name}: No samples")
            return {}

        predictions = self.predict(df)
        labels = df['assigned_label'].values

        metrics = compute_metrics(labels, predictions)

        print(f"\n{name} Results ({len(df)} samples):")
        print(f"  AUC:      {metrics['auc']:.4f}")
        print(f"  AUPRC:    {metrics['auprc']:.4f}")
        print(f"  MCC:      {metrics['mcc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1:       {metrics['f1']:.4f}")
        print(f"  AUC@0.1:  {metrics['auc_01']:.4f}")

        return metrics


# %% [markdown]
# # 11. Visualization

# %%
def plot_training_history(history: Dict, save_path: str = None):
    """Plot training history for all experts."""
    n_experts = len(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = plt.cm.tab10(np.linspace(0, 1, n_experts))

    for i, (cluster_id, hist) in enumerate(history.items()):
        if len(hist['train_loss']) > 0:
            axes[0].plot(hist['train_loss'], color=colors[i], label=f'Expert {cluster_id}')
            axes[1].plot(hist['val_loss'], color=colors[i], label=f'Expert {cluster_id}')
            axes[2].plot(hist['val_mcc'], color=colors[i], label=f'Expert {cluster_id}')

    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    axes[2].set_title('Validation MCC')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    plt.close(fig)  # Close figure to free memory in batch mode


def plot_cluster_distribution(allele_to_cluster: Dict, train_df: pd.DataFrame, save_path: str = None):
    """Plot distribution of samples across clusters."""
    train_df = train_df.copy()
    train_df['cluster'] = train_df['allele_key'].map(allele_to_cluster)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Samples per cluster
    cluster_counts = train_df['cluster'].value_counts().sort_index()
    axes[0].bar(cluster_counts.index, cluster_counts.values)
    axes[0].set_xlabel('Cluster ID')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title('Samples per Expert Cluster')

    # Alleles per cluster
    allele_per_cluster = train_df.groupby('cluster')['allele_key'].nunique()
    axes[1].bar(allele_per_cluster.index, allele_per_cluster.values)
    axes[1].set_xlabel('Cluster ID')
    axes[1].set_ylabel('Number of Alleles')
    axes[1].set_title('Unique Alleles per Expert Cluster')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved cluster distribution plot to {save_path}")
    plt.close(fig)  # Close figure to free memory in batch mode


# %% [markdown]
# # 12. Main Execution

# %%
def main(
        data_parquet_path: str,
        mhc_info_csv_path: str,
        benchmark_parquet_path: str = None,
        num_experts: int = NUM_EXPERTS,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        subset_size: Optional[int] = None,
        resume_from: Optional[str] = None
):
    """
    Main training and evaluation pipeline.

    Args:
        data_parquet_path: Path to main dataset parquet
        mhc_info_csv_path: Path to MHC info CSV
        benchmark_parquet_path: Path to benchmark parquet (optional)
        num_experts: Number of expert models
        epochs: Training epochs
        batch_size: Batch size
        subset_size: If set, use only this many samples for quick testing
        resume_from: Path to checkpoint directory to resume training from
    """

    # Load data
    train_df, val_df, unseen_test_df, benchmark_df, mhc_info = load_and_prepare_data(
        data_parquet_path, mhc_info_csv_path, benchmark_parquet_path, subset_size
    )

    # Initialize MoE trainer with optional checkpoint resumption
    moe = MixtureOfExpertsTrainer(
        train_df=train_df,
        val_df=val_df,
        mhc_info=mhc_info,
        num_experts=num_experts,
        batch_size=batch_size,
        epochs=epochs,
        patience=PATIENCE,
        resume_from=resume_from
    )

    # Plot cluster distribution (only if not resuming or plots don't exist)
    cluster_plot_path = os.path.join(OUTPUT_DIR, "plots", "cluster_distribution.png")
    if not os.path.exists(cluster_plot_path):
        print("\nCluster Distribution:")
        plot_cluster_distribution(
            moe.allele_to_cluster, train_df,
            save_path=cluster_plot_path
        )

    # Train
    history = moe.train()

    # Plot training history
    plot_training_history(
        history,
        save_path=os.path.join(OUTPUT_DIR, "plots", "training_history.png")
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    val_metrics = moe.evaluate(val_df, "Validation")
    unseen_metrics = moe.evaluate(unseen_test_df, "Unseen Alleles")

    if len(benchmark_df) > 0:
        bench_metrics = moe.evaluate(benchmark_df, "Benchmark")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Number of Experts: {num_experts}")
    print(f"Validation MCC: {val_metrics.get('mcc', 0):.4f}")
    print(f"Unseen Alleles MCC: {unseen_metrics.get('mcc', 0):.4f}")
    if len(benchmark_df) > 0:
        print(f"Benchmark MCC: {bench_metrics.get('mcc', 0):.4f}")

    # Save results
    results = {
        'num_experts': num_experts,
        'validation': val_metrics,
        'unseen_alleles': unseen_metrics,
        'benchmark': bench_metrics if len(benchmark_df) > 0 else {},
        'training_history': {str(k): v for k, v in history.items()},
        'config': {
            'max_pep_len': MAX_PEP_LEN,
            'max_mhc_len': MAX_MHC_LEN,
            'embed_dim': EMBED_DIM,
            'latent_dim': LATENT_DIM,
            'num_heads': NUM_HEADS,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': LEARNING_RATE,
            'dropout_rate': DROPOUT_RATE,
            'l2_reg': L2_REG,
            'patience': PATIENCE,
            'noise_std': NOISE_STD,
            'random_seed': RANDOM_SEED,
        }
    }

    results_path = os.path.join(OUTPUT_DIR, 'moe_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    
    # Save allele to cluster mapping
    cluster_mapping_path = os.path.join(OUTPUT_DIR, 'allele_to_cluster.json')
    with open(cluster_mapping_path, 'w') as f:
        json.dump(moe.allele_to_cluster, f, indent=2)
    print(f"Allele-cluster mapping saved to {cluster_mapping_path}")

    return moe, results


# %% [markdown]
# # 13. Run Training
#
# Modify the paths below to match your data files:
# To resume training from a checkpoint, set RESUME_FROM_CHECKPOINT at the top of the script.

# %%
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='MoE Peptide-MHC Binding Prediction')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint directory to resume training from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Total number of epochs to train (overrides EPOCHS constant)')
    args = parser.parse_args()
    
    # Override settings from command line
    if args.resume:
        RESUME_FROM_CHECKPOINT = args.resume
    if args.epochs:
        EPOCHS = args.epochs
    
    # Update OUTPUT_DIR if resuming
    if args.resume and os.path.exists(args.resume):
        OUTPUT_DIR = args.resume
        print(f"Resuming from checkpoint: {OUTPUT_DIR}")
    
    # === CONFIGURE YOUR PATHS HERE ===
    DATA_PARQUET = "/scratch-scc/users/u15472/PMBind/src/PY_runs/py6_moe/PMDb/PMDb_2025_11_18_class1.parquet"
    MHC_INFO_CSV = "/scratch-scc/users/u15472/PMBind/data/alleles/mhc_pseudo_class1.csv"
    BENCHMARK_PARQUET = "/scratch-scc/users/u15472/PMBind/data/bench_data/Julianes_bench_final_UNPUBLISHED.parquet" 

    # Check if files exist
    for path, name in [(DATA_PARQUET, "Data"), (MHC_INFO_CSV, "MHC Info")]:
        if not os.path.exists(path):
            print(f"Warning: {name} file not found at {path}")
            print("Please update the file paths before running.")

    # Run training (with optional checkpoint resumption)
    moe_model, results = main(
        data_parquet_path=DATA_PARQUET,
        mhc_info_csv_path=MHC_INFO_CSV,
        benchmark_parquet_path=BENCHMARK_PARQUET,
        num_experts=NUM_EXPERTS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        subset_size=SUBSET_SIZE,  # Set to e.g. 10000 for quick pipeline test
        resume_from=RESUME_FROM_CHECKPOINT  # Set at top of script to resume
    )


# %% [markdown]
# # 14. Additional Analysis Functions

# %%
def analyze_expert_specialization(moe: MixtureOfExpertsTrainer, mhc_info: pd.DataFrame):
    """Analyze what each expert has learned."""
    print("\n" + "=" * 60)
    print("EXPERT SPECIALIZATION ANALYSIS")
    print("=" * 60)

    for cluster_id in moe.experts.keys():
        # Get alleles in this cluster
        cluster_alleles = [k for k, v in moe.allele_to_cluster.items() if v == cluster_id]

        # Get MHC info for these alleles
        cluster_mhc = mhc_info[mhc_info['allele_key'].isin(cluster_alleles)]

        print(f"\nExpert {cluster_id}:")
        print(f"  Number of alleles: {len(cluster_alleles)}")

        if len(cluster_mhc) > 0:
            # Sample alleles
            sample_alleles = cluster_mhc['allele_key'].head(5).tolist()
            print(f"  Sample alleles: {', '.join(sample_alleles)}")


def predict_single(moe: MixtureOfExpertsTrainer, peptide: str, allele: str, mhc_sequence: str):
    """Make prediction for a single peptide-MHC pair."""
    # Create mini dataframe
    df = pd.DataFrame({
        'peptide': [peptide],
        'allele': [allele],
        'allele_key': [clean_allele_key(allele)],
        'mhc_sequence': [mhc_sequence],
        'assigned_label': [0]  # Dummy
    })

    prob = moe.predict(df)[0]
    return prob


def batch_predict(moe: MixtureOfExpertsTrainer, peptides: List[str],
                  alleles: List[str], mhc_sequences: List[str]) -> np.ndarray:
    """Make predictions for multiple peptide-MHC pairs."""
    df = pd.DataFrame({
        'peptide': peptides,
        'allele': alleles,
        'allele_key': [clean_allele_key(a) for a in alleles],
        'mhc_sequence': mhc_sequences,
        'assigned_label': [0] * len(peptides)
    })

    return moe.predict(df)
