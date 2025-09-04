#!/usr/bin/env python
import os
import pathlib
import gc
import json
import pickle
from typing import Set, Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


from utils import create_k_fold_leave_one_out_stratified_cv

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
MHC_CLASS = 1
RANDOM_SEED = 42
K_FOLDS = 10
N_RAREST_ALLELES = 100  # Number of rarest alleles for test set
MIN_ALLELES_FOR_CV = 15  # Minimum alleles needed for CV after test set extraction

# Input/Output paths
ROOT_DIR = pathlib.Path("../data/binding_affinity_data").resolve()
BINDING_PQ_PATH = ROOT_DIR / f"concatenated_class{MHC_CLASS}_all.parquet"
BENCHMARKS_PATH = pathlib.Path(f"../data/cross_validation_dataset/mhc{MHC_CLASS}/benchmarks")
OUT_DIR = pathlib.Path(f"../data/cross_validation_dataset/mhc{MHC_CLASS}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cache file for benchmark processing
BENCHMARK_CACHE_PKL = OUT_DIR / "benchmark_pairs_cache.pkl"  # legacy cache (if present)
BENCHMARK_CACHE_PQ = OUT_DIR / "benchmark_pairs_cache.parquet"  # faster cache format

# CV parameters
TRAIN_SIZE = 0.8
AUGMENTATION = "down_sampling"  # "down_sampling", "GNUSS", or None

# Performance/memory tweaks
pd.options.mode.copy_on_write = True
KEY_TRANS = str.maketrans({'*': '', ':': '', ' ': '', '/': '_'})  # for vectorized cleaning


def clean_key_vectorized(s: pd.Series) -> pd.Series:
    """Vectorized allele cleaning."""
    return s.astype(str).str.translate(KEY_TRANS).str.upper()


# def extract_test_set_from_rarest_alleles(df: pd.DataFrame, n_rarest: int = 20,
#                                          id_col: str = "mhc_embedding_key") -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """Extract test set from the n rarest alleles in the dataset.
#
#     Automatically adjusts n_rarest if it would leave too few alleles for CV.
#
#     Returns: (remaining_df, test_df)
#     """
#     print("\n=== Extracting Test Set from Rarest Alleles ===")
#     # value_counts on category dtype is faster and memory friendly
#     allele_counts = df[id_col].value_counts()
#     total_alleles = len(allele_counts)
#     print(f"Total unique alleles in dataset: {total_alleles}")
#
#     # Ensure enough alleles remain for CV
#     min_alleles_needed = K_FOLDS + MIN_ALLELES_FOR_CV
#     max_test_alleles = max(0, total_alleles - min_alleles_needed)
#
#     if n_rarest > max_test_alleles:
#         print(f"WARNING: Requested {n_rarest} test alleles, but only {total_alleles} total alleles available.")
#         print(f"Need at least {min_alleles_needed} alleles for {K_FOLDS}-fold CV.")
#         n_rarest = max_test_alleles
#         print(f"Adjusted to {n_rarest} test alleles.")
#
#     if n_rarest <= 0:
#         raise ValueError(
#             f"Dataset has too few alleles ({total_alleles}) for {K_FOLDS}-fold CV with test set extraction"
#         )
#
#     rarest_alleles = allele_counts.tail(n_rarest).index.tolist()
#
#     # Split dataset
#     test_mask = df[id_col].isin(rarest_alleles)
#     test_df = df.loc[test_mask].copy()
#     remaining_df = df.loc[~test_mask].copy().reset_index(drop=True)
#
#     print(f"\nTest set extracted:")
#     print(f"  Test set size: {len(test_df):,} samples from {len(rarest_alleles)} alleles")
#     print(f"  Remaining data: {len(remaining_df):,} samples from {remaining_df[id_col].nunique()} alleles")
#     print(f"  Test set class distribution: {dict(test_df['assigned_label'].value_counts())}")
#
#     return remaining_df, test_df


def extract_test_set_from_rarest_alleles(df: pd.DataFrame, n_rarest: int = 20,
                                         id_col: str = "mhc_embedding_key") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract test set from the n rarest alleles in the dataset with balanced labels.

    Automatically adjusts n_rarest if it would leave too few alleles for CV.
    Ensures balanced class distribution in the test set.

    Returns: (remaining_df, test_df)
    """
    print("\n=== Extracting Test Set from Rarest Alleles (Balanced) ===")
    # value_counts on category dtype is faster and memory friendly
    allele_counts = df[id_col].value_counts()
    total_alleles = len(allele_counts)
    print(f"Total unique alleles in dataset: {total_alleles}")

    # Ensure enough alleles remain for CV
    min_alleles_needed = K_FOLDS + MIN_ALLELES_FOR_CV
    max_test_alleles = max(0, total_alleles - min_alleles_needed)

    if n_rarest > max_test_alleles:
        print(f"WARNING: Requested {n_rarest} test alleles, but only {total_alleles} total alleles available.")
        print(f"Need at least {min_alleles_needed} alleles for {K_FOLDS}-fold CV.")
        n_rarest = max_test_alleles
        print(f"Adjusted to {n_rarest} test alleles.")

    if n_rarest <= 0:
        raise ValueError(
            f"Dataset has too few alleles ({total_alleles}) for {K_FOLDS}-fold CV with test set extraction"
        )

    # Get allele-wise class distribution for balanced selection
    allele_stats = df.groupby(id_col)['assigned_label'].agg(['count', 'mean']).reset_index()
    allele_stats.columns = [id_col, 'sample_count', 'positive_ratio']

    # Sort by sample count (ascending = rarest first)
    allele_stats = allele_stats.sort_values('sample_count')

    # Select alleles for balanced test set
    selected_alleles = []
    positive_samples = 0
    negative_samples = 0

    for _, row in allele_stats.iterrows():
        if len(selected_alleles) >= n_rarest:
            break

        allele_name = row[id_col]
        allele_data = df[df[id_col] == allele_name]
        allele_pos = (allele_data['assigned_label'] == 1).sum()
        allele_neg = (allele_data['assigned_label'] == 0).sum()

        # Simple heuristic: prefer alleles that help balance the test set
        current_pos_ratio = positive_samples / max(1, positive_samples + negative_samples)
        target_ratio = 0.5

        # If we need more positives and this allele has positives, or vice versa
        if (current_pos_ratio < target_ratio and allele_pos > 0) or \
                (current_pos_ratio > target_ratio and allele_neg > 0) or \
                len(selected_alleles) < n_rarest // 2:  # Always take first half regardless
            selected_alleles.append(allele_name)
            positive_samples += allele_pos
            negative_samples += allele_neg

    # If we still need more alleles, add remaining rarest ones
    remaining_rarest = [a for a in allele_stats[id_col].tolist() if a not in selected_alleles]
    while len(selected_alleles) < n_rarest and remaining_rarest:
        selected_alleles.append(remaining_rarest.pop(0))
        allele_data = df[df[id_col] == remaining_rarest[-1] if remaining_rarest else selected_alleles[-1]]
        positive_samples += (allele_data['assigned_label'] == 1).sum()
        negative_samples += (allele_data['assigned_label'] == 0).sum()

    # Split dataset
    test_mask = df[id_col].isin(selected_alleles)
    test_df = df.loc[test_mask].copy()
    remaining_df = df.loc[~test_mask].copy().reset_index(drop=True)

    print(f"\nBalanced test set extracted:")
    print(f"  Test set size: {len(test_df):,} samples from {len(selected_alleles)} alleles")
    print(f"  Remaining data: {len(remaining_df):,} samples from {remaining_df[id_col].nunique()} alleles")
    test_class_dist = dict(test_df['assigned_label'].value_counts())
    print(f"  Test set class distribution: {test_class_dist}")

    # Calculate balance ratio
    if len(test_class_dist) == 2:
        pos_count = test_class_dist.get(1, 0)
        neg_count = test_class_dist.get(0, 0)
        balance_ratio = min(pos_count, neg_count) / max(pos_count, neg_count) if max(pos_count, neg_count) > 0 else 0
        print(f"  Balance ratio (min/max): {balance_ratio:.3f}")

    return remaining_df, test_df


def _read_benchmark_file_to_pairs_df(fpath: pathlib.Path) -> Optional[pd.DataFrame]:
    """Read a single benchmark file and return unique (mhc_embedding_key, long_mer)-pairs as a DataFrame."""
    try:
        if fpath.suffix == ".parquet":
            bench_df = pd.read_parquet(fpath)
        elif fpath.suffix == ".csv":
            bench_df = pd.read_csv(fpath)
        else:  # .tsv
            bench_df = pd.read_csv(fpath, sep='\t')
    except Exception:
        return None

    # resolve column names
    cols_lower = {c.lower(): c for c in bench_df.columns}
    allele_col = next((bench_df.columns[i] for i, c in enumerate(bench_df.columns) if "allele" in c.lower()), None)
    peptide_col = None
    for key in ["peptide", "long_mer", "sequence"]:
        cand = cols_lower.get(key)
        if cand is not None:
            peptide_col = cand
            break

    if allele_col is None or peptide_col is None:
        return None

    # Clean/vectorize and drop duplicates
    pairs = pd.DataFrame({
        "mhc_embedding_key": clean_key_vectorized(bench_df[allele_col]),
        "long_mer": bench_df[peptide_col].astype(str)
    })
    pairs = pairs[(pairs["mhc_embedding_key"] != "") & (pairs["long_mer"] != "")]
    if pairs.empty:
        return None

    pairs = pairs.drop_duplicates(ignore_index=True)
    return pairs[["mhc_embedding_key", "long_mer"]]


def find_benchmark_pairs_df(benchmarks_path: pathlib.Path, use_cache: bool = True) -> pd.DataFrame:
    """Scan benchmark directory and return a DataFrame with unique (mhc_embedding_key, long_mer) pairs.

    Uses Parquet cache for fast reloads. Falls back to legacy pickle if present.
    """
    # Load Parquet cache if present
    if use_cache and BENCHMARK_CACHE_PQ.exists():
        print(f"Loading benchmark pairs from Parquet cache: {BENCHMARK_CACHE_PQ}")
        df_pairs = pd.read_parquet(BENCHMARK_CACHE_PQ)
        print(f"Loaded {len(df_pairs):,} cached benchmark pairs")
        return df_pairs

    # Legacy pickle compatibility
    if use_cache and BENCHMARK_CACHE_PKL.exists():
        print(f"Loading benchmark pairs from legacy cache: {BENCHMARK_CACHE_PKL}")
        try:
            with open(BENCHMARK_CACHE_PKL, 'rb') as f:
                cached_data = pickle.load(f)
            if isinstance(cached_data, dict) and "benchmark_pairs" in cached_data:
                pairs = list(cached_data["benchmark_pairs"])
                if pairs:
                    df_pairs = pd.DataFrame(pairs, columns=["mhc_embedding_key", "long_mer"]).drop_duplicates()
                    # Save as Parquet for future fast loads
                    df_pairs.to_parquet(BENCHMARK_CACHE_PQ, index=False)
                    print(f"Migrated legacy cache to Parquet: {BENCHMARK_CACHE_PQ}")
                    return df_pairs
        except Exception as e:
            print(f"Warning: Failed to load legacy cache ({e}), processing benchmarks...")

    print(f"Processing benchmark files from: {benchmarks_path}")
    if not benchmarks_path.exists() or not benchmarks_path.is_dir():
        print(f"Warning: Benchmark directory not found: {benchmarks_path}")
        return pd.DataFrame(columns=["mhc_embedding_key", "long_mer"])

    files: List[pathlib.Path] = (
            list(benchmarks_path.rglob("*.parquet"))
            + list(benchmarks_path.rglob("*.csv"))
            + list(benchmarks_path.rglob("*.tsv"))
    )
    print(f"Found {len(files)} benchmark files")

    all_chunks = []
    # Threaded reading (I/O bound)
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        futures = {ex.submit(_read_benchmark_file_to_pairs_df, f): f for f in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Reading benchmarks"):
            dfp = fut.result()
            if dfp is not None and not dfp.empty:
                all_chunks.append(dfp)

    if not all_chunks:
        print("No usable benchmark pairs found.")
        return pd.DataFrame(columns=["mhc_embedding_key", "long_mer"])

    bench_pairs = pd.concat(all_chunks, ignore_index=True).drop_duplicates()
    print(f"Total unique benchmark (allele, peptide) pairs: {len(bench_pairs):,}")

    # Cache to Parquet for fast reload next time
    if use_cache:
        try:
            bench_pairs.to_parquet(BENCHMARK_CACHE_PQ, index=False)
            print(f"Saved benchmark cache to: {BENCHMARK_CACHE_PQ}")
        except Exception as e:
            print(f"Warning: Failed to save Parquet cache: {e}")

    return bench_pairs


def main():
    """Main execution function."""
    print("=== Simplified K-Fold Leave-One-Allele-Out Cross-Validation with Test Set ===")
    print(f"K folds: {K_FOLDS}")
    print(f"Train size: {TRAIN_SIZE}")
    print(f"Augmentation: {AUGMENTATION}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Requested test set from {N_RAREST_ALLELES} rarest alleles")

    # 1. Load main dataset (only needed columns)
    print(f"\nLoading data from: {BINDING_PQ_PATH}")
    if not BINDING_PQ_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {BINDING_PQ_PATH}")

    needed_cols = ["allele", "assigned_label", "long_mer"]
    df = pd.read_parquet(BINDING_PQ_PATH, columns=needed_cols)
    print(f"Loaded {len(df):,} samples")

    # Vectorized cleaning of allele names, use category dtype to speed value_counts, groupby, etc.
    print("\nCleaning allele names...")
    df["mhc_embedding_key"] = clean_key_vectorized(df["allele"]).astype("category")
    print("Sample cleaned alleles:", df["mhc_embedding_key"].unique()[:5])

    # Tighten dtypes if possible
    if pd.api.types.is_numeric_dtype(df["assigned_label"]):
        # If labels are 0/1 or small integer range, compress
        try:
            df["assigned_label"] = pd.to_numeric(df["assigned_label"], downcast="integer")
        except Exception:
            pass
    else:
        # keep as category if strings
        df["assigned_label"] = df["assigned_label"].astype("category")

    # Checks
    required_cols = ["mhc_embedding_key", "assigned_label", "long_mer"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    uniq_pairs = df.drop_duplicates(["mhc_embedding_key", "long_mer"]).shape[0]
    print("Dataset overview:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Unique alleles: {df['mhc_embedding_key'].nunique()}")
    print(f"  Unique (allele, long_mer) pairs: {uniq_pairs}")
    print(f"  Class distribution: {dict(df['assigned_label'].value_counts())}")

    # 2. Find benchmark pairs (with caching, DataFrame-based)
    print("\n=== Checking for benchmark (allele, peptide) pairs ===")
    bench_pairs_df = find_benchmark_pairs_df(BENCHMARKS_PATH, use_cache=True)

    # 2a. Remove benchmark pairs via anti-join (fast, avoids python sets)
    pre_filter_count = len(df)
    if not bench_pairs_df.empty:
        # Ensure same dtypes for join columns
        bench_pairs_df["mhc_embedding_key"] = bench_pairs_df["mhc_embedding_key"].astype(df["mhc_embedding_key"].dtype)
        bench_pairs_df["long_mer"] = bench_pairs_df["long_mer"].astype(df["long_mer"].dtype)

        # Anti-join to remove overlapping pairs
        df = df.merge(bench_pairs_df.drop_duplicates(), how="left", on=["mhc_embedding_key", "long_mer"], indicator=True)
        removed = (df["_merge"] == "both").sum()
        df = df.loc[df["_merge"] == "left_only", df.columns.difference(["_merge"])].reset_index(drop=True)
        print(f"Removed {removed:,} samples with benchmark (allele, peptide) pairs")
    else:
        print("No overlapping (allele, peptide) pairs found with benchmarks")

    print("Dataset after benchmark pair removal:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Unique alleles: {df['mhc_embedding_key'].nunique()}")
    print(f"  Unique (allele, long_mer) pairs: {df.drop_duplicates(['mhc_embedding_key', 'long_mer']).shape[0]}")

    # 3. Extract test set from rarest alleles BEFORE cross-validation
    df_remaining, df_test = extract_test_set_from_rarest_alleles(
        df, n_rarest=N_RAREST_ALLELES, id_col='mhc_embedding_key'
    )

    # Save test set
    test_set_path = OUT_DIR / "test_set_rarest_alleles.parquet"
    df_test.to_parquet(test_set_path, index=False)
    print(f"✓ Test set saved to: {test_set_path}")

    # NEW: Separate into positive and negative pools for sampling
    print("\n=== Creating Positive and Negative Pools for Sampling ===")
    positives_pool = df_remaining[df_remaining["assigned_label"] == 1].copy()
    negatives_pool = df_remaining[df_remaining["assigned_label"] == 0].copy()

    positives_pool_path = OUT_DIR / "positives_pool.parquet"
    negatives_pool_path = OUT_DIR / "negatives_pool.parquet"

    positives_pool.to_parquet(positives_pool_path, index=False)
    negatives_pool.to_parquet(negatives_pool_path, index=False)

    print(f"✓ Positives pool saved to: {positives_pool_path} ({len(positives_pool):,} samples)")
    print(f"✓ Negatives pool saved to: {negatives_pool_path} ({len(negatives_pool):,} samples)")

    # Show final dataset info for CV
    print("\nFinal dataset for CV (after removing test set):")
    print(f"  Total samples: {len(df_remaining):,}")
    print(f"  Unique alleles: {df_remaining['mhc_embedding_key'].nunique()}")
    print(f"  Class distribution: {dict(df_remaining['assigned_label'].value_counts())}")

    # Check if we have enough alleles for CV
    if df_remaining['mhc_embedding_key'].nunique() < K_FOLDS:
        raise ValueError(f"Not enough alleles for {K_FOLDS}-fold CV. "
                         f"Available: {df_remaining['mhc_embedding_key'].nunique()}, Need: {K_FOLDS}")

    # 4. Generate cross-validation folds using remaining data
    print(f"\n=== Generating {K_FOLDS}-Fold Cross-Validation ===")
    folds = create_k_fold_leave_one_out_stratified_cv(
        df=df_remaining,
        k=K_FOLDS,
        target_col="assigned_label",
        id_col="mhc_embedding_key",
        train_size=TRAIN_SIZE,
        random_state=RANDOM_SEED,
        augmentation=AUGMENTATION,
        n_val_ids=20  # number of alleles to leave out for validation in each fold
    )

    # 5. Save folds with progress bar
    print("\n=== Saving Cross-Validation Folds ===")
    cv_dir = OUT_DIR / "cv_folds"
    cv_dir.mkdir(exist_ok=True)

    for i, (df_train, df_val, left_out_allele) in enumerate(tqdm(folds, total=K_FOLDS, desc="Saving folds"), 1):
        train_path = cv_dir / f"fold_{i:02d}_train.parquet"
        val_path = cv_dir / f"fold_{i:02d}_val.parquet"

        df_train.to_parquet(train_path, index=False)
        df_val.to_parquet(val_path, index=False)

        actual_ratio = len(df_train) / max(1, (len(df_train) + len(df_val)))
        print(f"  ✓ Saved fold {i:02d} (left-out: {left_out_allele}): "
              f"Train={len(df_train):,}, Val={len(df_val):,}, Ratio={actual_ratio:.3f}")

        # Memory cleanup
        del df_train, df_val
        gc.collect()

    # 6. Save summary information
    actual_test_alleles = int(df_test['mhc_embedding_key'].nunique())
    benchmark_removed = int(pre_filter_count - len(df))
    summary_info = {
        'total_original_samples': int(len(df) + len(df_test) + benchmark_removed),
        'benchmark_pairs_removed': benchmark_removed,
        'test_set_samples': int(len(df_test)),
        'test_set_alleles_requested': int(N_RAREST_ALLELES),
        'test_set_alleles_actual': actual_test_alleles,
        'cv_samples': int(len(df_remaining)),
        'cv_alleles': int(df_remaining['mhc_embedding_key'].nunique()),
        'n_folds': int(K_FOLDS),
        'test_set_path': str(test_set_path),
        'cv_folds_path': str(cv_dir),
        'positives_pool_path': str(positives_pool_path),
        'negatives_pool_path': str(negatives_pool_path),
        'benchmark_cache_path': str(BENCHMARK_CACHE_PQ if BENCHMARK_CACHE_PQ.exists() else BENCHMARK_CACHE_PKL),
    }

    summary_path = OUT_DIR / "data_split_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=== Data Split Summary ===\n")
        for key, value in summary_info.items():
            f.write(f"{key}: {value}\n")

    print(f"\n✓ Successfully created {K_FOLDS} cross-validation folds")
    print(f"✓ Test set: {len(df_test):,} samples from {actual_test_alleles} rarest alleles")
    print(f"✓ CV data: {len(df_remaining):,} samples from {df_remaining['mhc_embedding_key'].nunique()} alleles")
    print(f"✓ Files saved in: {cv_dir}")
    print(f"✓ Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()