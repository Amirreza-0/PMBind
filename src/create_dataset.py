#!/usr/bin/env python
"""
Simplified cross-validation script using create_k_fold_leave_one_out_stratified_cv.
Checks for benchmark samples, creates test set from rarest alleles, and creates train/val files for each fold.
"""

import os
import pathlib
import gc
import json
import pickle
from typing import Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.utils import resample
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
BENCHMARKS_PATH = pathlib.Path(f"/home/amirreza/Desktop/PMBind/data/cross_validation_dataset/mhc{MHC_CLASS}")
OUT_DIR = pathlib.Path(f"/home/amirreza/Desktop/PMBind/data/cross_validation_dataset/mhc{MHC_CLASS}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cache file for benchmark processing
BENCHMARK_CACHE_PATH = OUT_DIR / "benchmark_pairs_cache.pkl"

# CV parameters
TRAIN_SIZE = 0.8
SUBSET_PROP = 0.1  # Use 10% of data for testing
AUGMENTATION = None  # "down_sampling", "GNUSS", or None


def clean_key(allele_key: str) -> str:
    """Clean and normalize allele names."""
    if not isinstance(allele_key, str):
        return ""
    mapping = str.maketrans({'*': '', ':': '', ' ': '', '/': '_'})
    return allele_key.translate(mapping).upper()


def extract_test_set_from_rarest_alleles(df: pd.DataFrame, n_rarest: int = 20,
                                         id_col: str = "cleaned_allele") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract test set from the n rarest alleles in the dataset.
    Automatically adjusts n_rarest if it would leave too few alleles for CV.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (remaining_df, test_df)
    """
    print(f"\n=== Extracting Test Set from Rarest Alleles ===")

    # Count samples per allele
    allele_counts = df[id_col].value_counts()
    total_alleles = len(allele_counts)
    print(f"Total unique alleles in dataset: {total_alleles}")

    # Check if we have enough alleles for the requested split
    min_alleles_needed = K_FOLDS + MIN_ALLELES_FOR_CV
    max_test_alleles = max(0, total_alleles - min_alleles_needed)

    if n_rarest > max_test_alleles:
        print(f"WARNING: Requested {n_rarest} test alleles, but only {total_alleles} total alleles available.")
        print(f"Need at least {min_alleles_needed} alleles for {K_FOLDS}-fold CV.")
        n_rarest = max_test_alleles
        print(f"Adjusted to {n_rarest} test alleles.")

    if n_rarest <= 0:
        print(
            f"ERROR: Not enough alleles for test set extraction. Total: {total_alleles}, Need for CV: {min_alleles_needed}")
        raise ValueError(
            f"Dataset has too few alleles ({total_alleles}) for {K_FOLDS}-fold CV with test set extraction")

    # Get the n rarest alleles
    rarest_alleles = allele_counts.tail(n_rarest).index.tolist()

    print(f"Selected {n_rarest} rarest alleles and their sample counts:")
    for i, allele in enumerate(rarest_alleles, 1):
        count = allele_counts[allele]
        print(f"  {i:2d}. {allele}: {count} samples")

    # Split dataset
    test_mask = df[id_col].isin(rarest_alleles)
    test_df = df[test_mask].copy()
    remaining_df = df[~test_mask].copy().reset_index(drop=True)

    print(f"\nTest set extracted:")
    print(f"  Test set size: {len(test_df):,} samples from {len(rarest_alleles)} alleles")
    print(f"  Remaining data: {len(remaining_df):,} samples from {remaining_df[id_col].nunique()} alleles")
    print(f"  Test set class distribution: {dict(test_df['assigned_label'].value_counts())}")

    return remaining_df, test_df


def find_benchmark_pairs(benchmarks_path: pathlib.Path, use_cache: bool = True) -> Set[Tuple[str, str]]:
    """
    Scan benchmark directory and return set of (cleaned_allele, long_mer) pairs.
    Uses caching to avoid re-processing on subsequent runs.
    """
    # Check if cache exists and use_cache is True
    if use_cache and BENCHMARK_CACHE_PATH.exists():
        print(f"Loading benchmark pairs from cache: {BENCHMARK_CACHE_PATH}")
        try:
            with open(BENCHMARK_CACHE_PATH, 'rb') as f:
                cached_data = pickle.load(f)
            benchmark_pairs = cached_data['benchmark_pairs']
            print(f"Loaded {len(benchmark_pairs)} cached benchmark pairs")
            return benchmark_pairs
        except Exception as e:
            print(f"Warning: Failed to load cache ({e}), processing benchmarks...")

    print(f"Processing benchmark files from: {benchmarks_path}")

    if not benchmarks_path.exists() or not benchmarks_path.is_dir():
        print(f"Warning: Benchmark directory not found: {benchmarks_path}")
        return set()

    benchmark_files = (
            list(benchmarks_path.rglob("*.parquet")) +
            list(benchmarks_path.rglob("*.csv")) +
            list(benchmarks_path.rglob("*.tsv"))
    )

    print(f"Found {len(benchmark_files)} benchmark files")
    benchmark_pairs = set()
    processing_stats = {}

    for f in benchmark_files:
        try:
            if f.suffix == ".parquet":
                bench_df = pd.read_parquet(f)
            elif f.suffix == ".csv":
                bench_df = pd.read_csv(f)
            else:  # .tsv
                bench_df = pd.read_csv(f, sep='\t')

            # Look for both allele and peptide/long_mer columns
            allele_col = None
            peptide_col = None

            for col in bench_df.columns:
                if 'allele' in col.lower():
                    allele_col = col
                elif any(x in col.lower() for x in ['peptide', 'long_mer', 'sequence']):
                    peptide_col = col

            if allele_col and peptide_col:
                # Create pairs of (cleaned_allele, long_mer)
                pairs = set()
                for _, row in bench_df.iterrows():
                    cleaned_allele = clean_key(str(row[allele_col]))
                    long_mer = str(row[peptide_col])
                    if cleaned_allele and long_mer:
                        pairs.add((cleaned_allele, long_mer))

                benchmark_pairs.update(pairs)
                processing_stats[f.name] = len(pairs)
                print(f"  {f.name}: {len(pairs)} unique (allele, peptide) pairs")
            else:
                missing_cols = []
                if not allele_col:
                    missing_cols.append("allele")
                if not peptide_col:
                    missing_cols.append("peptide/long_mer")
                print(f"  Warning: {f.name} missing columns: {missing_cols}")
                processing_stats[f.name] = f"Missing columns: {missing_cols}"

        except Exception as e:
            print(f"  Warning: Could not process {f.name}: {e}")
            processing_stats[f.name] = f"Error: {e}"

    print(f"Total unique benchmark (allele, peptide) pairs: {len(benchmark_pairs)}")

    # Save to cache
    if use_cache:
        cache_data = {
            'benchmark_pairs': benchmark_pairs,
            'processing_stats': processing_stats,
            'timestamp': pd.Timestamp.now().isoformat(),
            'n_files_processed': len(benchmark_files),
            'benchmarks_path': str(benchmarks_path)
        }
        try:
            with open(BENCHMARK_CACHE_PATH, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved benchmark cache to: {BENCHMARK_CACHE_PATH}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    return benchmark_pairs


def main():
    """Main execution function."""
    print(f"=== Simplified K-Fold Leave-One-Allele-Out Cross-Validation with Test Set ===")
    print(f"K folds: {K_FOLDS}")
    print(f"Train size: {TRAIN_SIZE}")
    print(f"Subset proportion: {SUBSET_PROP}")
    print(f"Augmentation: {AUGMENTATION}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Requested test set from {N_RAREST_ALLELES} rarest alleles")

    # 1. Load main dataset
    print(f"\nLoading data from: {BINDING_PQ_PATH}")
    if not BINDING_PQ_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {BINDING_PQ_PATH}")

    df = pd.read_parquet(BINDING_PQ_PATH)
    print(f"Loaded {len(df):,} samples")

    # Clean allele names
    print("\nCleaning allele names...")
    df['cleaned_allele'] = df['allele'].apply(clean_key)

    # Check required columns
    required_cols = ['cleaned_allele', 'assigned_label', 'long_mer']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"Dataset overview:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Unique alleles: {df['cleaned_allele'].nunique()}")
    print(f"  Unique (allele, long_mer) pairs: {len(df.groupby(['cleaned_allele', 'long_mer']))}")
    print(f"  Class distribution: {dict(df['assigned_label'].value_counts())}")

    # 2. Find benchmark pairs (with caching)
    print(f"\n=== Checking for benchmark (allele, peptide) pairs ===")
    benchmark_pairs = find_benchmark_pairs(BENCHMARKS_PATH, use_cache=True)

    # Check overlap with main dataset
    main_pairs = set(zip(df['cleaned_allele'], df['long_mer']))
    overlapping_pairs = benchmark_pairs.intersection(main_pairs)

    if overlapping_pairs:
        print(f"Found {len(overlapping_pairs)} overlapping (allele, peptide) pairs with benchmarks")

        # Save benchmark overlap stats
        overlap_stats_path = OUT_DIR / "benchmark_overlap_stats.txt"
        with open(overlap_stats_path, 'w') as f:
            f.write("=== Benchmark Overlap Stats ===\n")
            f.write(f"Total benchmark pairs: {len(benchmark_pairs)}\n")
            f.write(f"Overlapping pairs: {len(overlapping_pairs)}\n\n")

            # Count samples for each overlapping pair
            sample_counts = {}
            for allele, peptide in overlapping_pairs:
                count = ((df['cleaned_allele'] == allele) & (df['long_mer'] == peptide)).sum()
                sample_counts[(allele, peptide)] = count

            # Sort by count for better readability
            sorted_pairs = sorted(sample_counts.items(), key=lambda x: x[1], reverse=True)

            for (allele, peptide), count in sorted_pairs:
                print(f"  ({allele}, {peptide[:20]}{'...' if len(peptide) > 20 else ''}): {count} samples")
                f.write(f"({allele}, {peptide}): {count} samples\n")

        # Remove benchmark pairs from main dataset
        pre_filter_count = len(df)
        df['pair_key'] = list(zip(df['cleaned_allele'], df['long_mer']))
        df = df[~df['pair_key'].isin(overlapping_pairs)].drop('pair_key', axis=1).reset_index(drop=True)
        print(f"Removed {pre_filter_count - len(df):,} samples with benchmark (allele, peptide) pairs")
    else:
        print("No overlapping (allele, peptide) pairs found with benchmarks")

    print(f"Dataset after benchmark pair removal:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Unique alleles: {df['cleaned_allele'].nunique()}")
    print(f"  Unique (allele, long_mer) pairs: {len(df.groupby(['cleaned_allele', 'long_mer']))}")

    # 3. Extract test set from rarest alleles BEFORE cross-validation
    df_remaining, df_test = extract_test_set_from_rarest_alleles(
        df, n_rarest=N_RAREST_ALLELES, id_col='cleaned_allele'
    )

    # Save test set
    test_set_path = OUT_DIR / "test_set_rarest_alleles.parquet"
    df_test.to_parquet(test_set_path, index=False)
    print(f"✓ Test set saved to: {test_set_path}")

    # Show final dataset info for CV
    print(f"\nFinal dataset for CV (after removing test set):")
    print(f"  Total samples: {len(df_remaining):,}")
    print(f"  Unique alleles: {df_remaining['cleaned_allele'].nunique()}")
    print(f"  Class distribution: {dict(df_remaining['assigned_label'].value_counts())}")

    # Check if we have enough alleles for CV
    if df_remaining['cleaned_allele'].nunique() < K_FOLDS:
        raise ValueError(f"Not enough alleles for {K_FOLDS}-fold CV. "
                         f"Available: {df_remaining['cleaned_allele'].nunique()}, Need: {K_FOLDS}")

    # 4. Generate cross-validation folds using remaining data
    print(f"\n=== Generating {K_FOLDS}-Fold Cross-Validation ===")

    folds = create_k_fold_leave_one_out_stratified_cv(
        df=df_remaining,  # Use remaining data (without test set)
        k=K_FOLDS,
        target_col="assigned_label",
        id_col="cleaned_allele",
        subset_prop=SUBSET_PROP,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_SEED,
        augmentation=AUGMENTATION
    )

    # 5. Save folds
    print(f"\n=== Saving Cross-Validation Folds ===")
    cv_dir = OUT_DIR / "cv_folds"
    cv_dir.mkdir(exist_ok=True)

    for i, (df_train, df_val, left_out_allele) in enumerate(folds, 1):
        # Save train and validation files with simple naming
        train_path = cv_dir / f"fold_{i:02d}_train.parquet"
        val_path = cv_dir / f"fold_{i:02d}_val.parquet"

        df_train.to_parquet(train_path, index=False)
        df_val.to_parquet(val_path, index=False)

        actual_ratio = len(df_train) / (len(df_train) + len(df_val))
        print(
            f"  ✓ Saved fold {i:02d} (left-out: {left_out_allele}): Train={len(df_train):,}, Val={len(df_val):,}, Ratio={actual_ratio:.3f}")

        # Memory cleanup
        del df_train, df_val
        gc.collect()

    # 6. Save summary information
    actual_test_alleles = len(df_test['cleaned_allele'].unique())
    summary_info = {
        'total_original_samples': len(df) + len(df_test) + (pre_filter_count - len(df) - len(df_test) if overlapping_pairs else 0),
        'benchmark_pairs_removed': (pre_filter_count - len(df) - len(df_test)) if overlapping_pairs else 0,
        'test_set_samples': len(df_test),
        'test_set_alleles_requested': N_RAREST_ALLELES,
        'test_set_alleles_actual': actual_test_alleles,
        'cv_samples': len(df_remaining),
        'cv_alleles': df_remaining['cleaned_allele'].nunique(),
        'n_folds': len(folds),
        'test_set_path': str(test_set_path),
        'cv_folds_path': str(cv_dir),
        'benchmark_cache_path': str(BENCHMARK_CACHE_PATH)
    }

    summary_path = OUT_DIR / "data_split_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=== Data Split Summary ===\n")
        for key, value in summary_info.items():
            f.write(f"{key}: {value}\n")
        f.write("\n=== Fold Details ===\n")
        for i, (_, _, left_out_allele) in enumerate(folds, 1):
            f.write(f"fold_{i:02d}: left-out allele = {left_out_allele}\n")

    print(f"\n✓ Successfully created {len(folds)} cross-validation folds")
    print(f"✓ Test set: {len(df_test):,} samples from {actual_test_alleles} rarest alleles")
    print(f"✓ CV data: {len(df_remaining):,} samples from {df_remaining['cleaned_allele'].nunique()} alleles")
    print(f"✓ Files saved in single folder: {cv_dir}")
    print(f"✓ Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()