import os
import pathlib
from typing import Set, Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def train_val_split(
        df: pd.DataFrame,
        k: int = 5,
        target_col: str = "assigned_label",
        id_col: str = "mhc_embedding_key",
        train_size: float = 0.8,
        random_state: int = 42,
        n_val_ids: int = 1,
):
    # --------
    rng = np.random.RandomState(random_state)
    # --- pick the k IDs that will be held out completely -------------------
    unique_ids = df[id_col].unique()
    if len(unique_ids) < k:
        raise ValueError(f"Not enough unique IDs ({len(unique_ids)}) for {k}-fold CV.")
    left_out_ids = rng.choice(unique_ids, size=k, replace=False)

    folds = []
    for fold_idx, left_out_id in enumerate(left_out_ids, 1):
        fold_seed = random_state + fold_idx
        rng = np.random.RandomState(fold_seed)

        # select rows not containing the left-out ID
        df_fold = df[df[id_col] != left_out_id].copy()

        # DEBUG: Check if left_out_id was actually in the data
        if left_out_id not in df[id_col].values:
            raise ValueError(f"Left-out ID {left_out_id} not found in data.")

        # DEBUG: Print data types and unique counts
        print(f"\n[DEBUG fold {fold_idx}]")
        print(f"  ID column dtype: {df_fold[id_col].dtype}")
        print(f"  Number of unique IDs in fold: {df_fold[id_col].nunique()}")
        print(f"  Total rows in fold: {len(df_fold)}")

        # 2) split train and val, val is n_val_ids of rarest IDs in the fold
        # IMPORTANT: Filter out IDs with 0 counts (happens with categorical columns)
        id_counts = df_fold[id_col].value_counts()
        id_counts = id_counts[id_counts > 0]  # Remove IDs with 0 occurrences

        # DEBUG: Print value counts
        print(f"  ID value counts (first 5):")
        print(f"    {id_counts.head().to_dict()}")
        print(f"  ID value counts (last 5):")
        print(f"    {id_counts.tail().to_dict()}")

        # Now select the rarest IDs that actually exist in the data
        if len(id_counts) < n_val_ids:
            print(f"  WARNING: Only {len(id_counts)} unique IDs with data, but requested {n_val_ids} val IDs")
            n_val_ids = min(n_val_ids, len(id_counts))

        rare_alleles = id_counts.nsmallest(n_val_ids * k).index.tolist()
        rarest_ids = rng.choice(list(rare_alleles), size=n_val_ids, replace=False)
        val_ids = set(rarest_ids)

        # DEBUG: Print selected validation IDs
        print(f"  Selected val_ids: {val_ids}")
        print(f"  Type of first val_id: {type(list(val_ids)[0]) if val_ids else 'None'}")

        # Select val rows from the original fold data
        fold_val = df_fold[df_fold[id_col].isin(val_ids)].copy()

        # DEBUG: Check the actual filtering
        print(f"  Rows matching val_ids: {len(fold_val)}")
        if len(fold_val) == 0 and val_ids:
            # Try manual check
            manual_check = df_fold[id_col].apply(lambda x: x in val_ids).sum()
            print(f"  Manual check - rows matching val_ids: {manual_check}")

            # Check if the IDs actually exist
            for vid in val_ids:
                exists = (df_fold[id_col] == vid).sum()
                print(f"    ID {vid} (type: {type(vid)}) exists in fold: {exists} times")

        # Select remaining rows for training from the original fold data
        fold_remaining = df_fold[~df_fold[id_col].isin(val_ids)].copy()

        if fold_remaining.empty:
            raise ValueError(f"No remaining data for training after leaving out IDs: {val_ids}")

        # verify if left_out id and val_ids are not present in remaining data
        if left_out_id in fold_remaining[id_col].values:
            raise ValueError(f"Left-out ID {left_out_id} found in remaining data.")
        if any(vid in fold_remaining[id_col].values for vid in val_ids):
            raise ValueError(f"Validation IDs {val_ids} found in remaining data.")

        print(
            f"[fold {fold_idx}/{k}] left-out={left_out_id} | "
            f"val-only={val_ids} | fold-ids={set(df_fold[id_col])} | fold-remaining-ids={set(fold_remaining[id_col])} | "
            f"train={len(fold_remaining)}, val={len(fold_val)} | "
            f"len-fold-ids={df_fold[id_col].nunique()} | len-fold-remaining-ids={fold_remaining[id_col].nunique()} | len-val-only-ids={len(val_ids)}"
        )

        # IMPORTANT: Actually append the fold data to the folds list!
        folds.append((fold_remaining, fold_val, left_out_id, val_ids))

    return folds


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
MHC_CLASS = 1
RANDOM_SEED = 999
K_FOLDS = 10
N_RAREST_ALLELES = 200  # Number of rarest alleles for test set
MIN_ALLELES_FOR_CV = 15  # Minimum alleles needed for CV after test set extraction
N_VAL_FOLD_SAMPLES = 15  # Number of alleles to leave out for validation in each fold

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
AUGMENTATION = None  # "down_sampling", "GNUSS", or None

# Performance/memory tweaks
pd.options.mode.copy_on_write = True
KEY_TRANS = str.maketrans({'*': '', ':': '', ' ': '', '/': '_'})  # for vectorized cleaning


def clean_key_vectorized(s: pd.Series) -> pd.Series:
    """Vectorized allele cleaning."""
    return s.astype(str).str.translate(KEY_TRANS).str.upper()


def extract_test_set_from_rarest_alleles(df: pd.DataFrame, n_rarest: int = 20,
                                         id_col: str = "mhc_embedding_key") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract test set from the n rarest alleles in the dataset.

    Returns: (remaining_df, test_df)
    """
    print("\n=== Extracting Test Set from Rarest Alleles (Balanced) ===")

    # get rarest alleles based on counts
    allele_counts = df[id_col].value_counts()
    total_alleles = len(allele_counts)
    print(f"Total unique alleles in dataset: {total_alleles}")
    rarest_alleles = allele_counts.tail(n_rarest).index.tolist()

    # Split dataset into rarest alleles and the rest
    rarest_mask = df[id_col].isin(rarest_alleles)
    rarest_df = df.loc[rarest_mask].copy()
    remaining_df = df.loc[~rarest_mask].copy().reset_index(drop=True)
    print(f"Rarest alleles data: {len(rarest_df):,} samples from {rarest_df[id_col].nunique()} alleles")

    # verify we have enough alleles for CV after test set extraction
    if remaining_df[id_col].nunique() < MIN_ALLELES_FOR_CV:
        raise ValueError(f"Not enough alleles ({remaining_df[id_col].nunique()}) left for CV after extracting "
                         f"test set from {n_rarest} rarest alleles. Minimum required: {MIN_ALLELES_FOR_CV}")

    # verify if rarest alleles are not in remaining data
    if any(allele in remaining_df[id_col].values for allele in rarest_alleles):
        raise ValueError("Rarest alleles found in remaining data after extraction.")

    return remaining_df, rarest_df


def _read_benchmark_file_to_pairs_df(fpath: pathlib.Path) -> Optional[pd.DataFrame]:
    """Read a single benchmark file and return unique (mhc_embedding_key, long_mer)-pairs as a DataFrame."""

    if fpath.suffix == ".parquet":
        bench_df = pd.read_parquet(fpath)
    elif fpath.suffix == ".csv":
        bench_df = pd.read_csv(fpath)
    elif fpath.suffix == ".tsv":  # .tsv
        bench_df = pd.read_csv(fpath, sep='\t')
    else:
        raise ValueError(f"Unsupported file type: {fpath.suffix}")

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
        raise ValueError(f"Could not find allele column in benchmark file: {fpath}")

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


def find_benchmark_pairs_df(benchmarks_path: pathlib.Path) -> pd.DataFrame:
    """Scan benchmark directory and return a DataFrame with unique (mhc_embedding_key, long_mer) pairs.

    Uses Parquet cache for fast reloads. Falls back to legacy pickle if present.
    """
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
    # Save
    if BENCHMARK_CACHE_PQ.exists():
        print(f"Using existing Parquet cache: {BENCHMARK_CACHE_PQ}")
    else:
        bench_pairs.to_parquet(BENCHMARK_CACHE_PQ, index=False)
        print(f"✓ Saved benchmark pairs cache to: {BENCHMARK_CACHE_PQ}")

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

    # clean labels - convert to int if needed and remove NaNs
    df = df.dropna(subset=["assigned_label"])
    df["assigned_label"] = df["assigned_label"].astype(int)
    print(f"Labels after cleaning: {df['assigned_label'].value_counts().to_dict()}")
    if df["assigned_label"].nunique() < 2:
        raise ValueError("Dataset must contain at least two classes after label cleaning.")
    if df["assigned_label"].nunique() > 2:
        raise ValueError("Dataset must be binary classification (0/1 labels).")

    # Vectorized cleaning of allele names, use category dtype to speed value_counts, groupby, etc.
    print("\nCleaning allele names...")
    df["mhc_embedding_key"] = clean_key_vectorized(df["allele"]).astype("category")
    print("Sample cleaned alleles:", df["mhc_embedding_key"].unique()[:5])

    uniq_pairs = df.drop_duplicates(["mhc_embedding_key", "long_mer"]).shape[0]
    print("Dataset overview:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Unique alleles: {df['mhc_embedding_key'].nunique()}")
    print(f"  Unique (allele, long_mer) pairs: {uniq_pairs}")
    print(f"  Class distribution: {dict(df['assigned_label'].value_counts())}")

    # 2. Find benchmark pairs (with caching, DataFrame-based)
    print("\n=== Checking for benchmark (allele, peptide) pairs ===")
    bench_pairs_df = find_benchmark_pairs_df(BENCHMARKS_PATH)

    # 2a. Remove benchmark pairs via anti-join (fast, avoids python sets)
    pre_filter_count = len(df)
    if not bench_pairs_df.empty:
        # Ensure same dtypes for join columns
        bench_pairs_df["mhc_embedding_key"] = bench_pairs_df["mhc_embedding_key"].astype(df["mhc_embedding_key"].dtype)
        bench_pairs_df["long_mer"] = bench_pairs_df["long_mer"].astype(df["long_mer"].dtype)

        # Anti-join to remove overlapping pairs
        # df = df.merge(bench_pairs_df.drop_duplicates(), how="left", on=["mhc_embedding_key", "long_mer"], indicator=True)
        # removed = (df["_merge"] == "both").sum()
        # df = df.loc[df["_merge"] == "left_only", df.columns.difference(["_merge"])].reset_index(drop=True)
        df = (
            df.merge(
                bench_pairs_df.drop_duplicates(),
                on=['mhc_embedding_key', 'long_mer'],
                how='left',
                indicator=True
            )
            .query("_merge == 'left_only'")
            .drop(columns='_merge')
            .reset_index(drop=True)
        )
        # verify no overlap remains
        overlap_check = pd.merge(df, bench_pairs_df, on=['mhc_embedding_key', 'long_mer'], how='inner')
        if not overlap_check.empty:
            raise ValueError("Overlap with benchmark pairs still exists after filtering.")

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
    folds = train_val_split(
        df=df_remaining,
        k=K_FOLDS,
        target_col="assigned_label",
        id_col="mhc_embedding_key",
        train_size=TRAIN_SIZE,
        random_state=RANDOM_SEED,
        n_val_ids=N_VAL_FOLD_SAMPLES  # number of alleles to leave out for validation in each fold
    )

    cv_dir = OUT_DIR / "cv_folds"
    cv_dir.mkdir(exist_ok=True)

    for i, (df_train, df_val, left_out_allele, val_ids) in enumerate(folds):
        train_path = cv_dir / f"fold_{i+1:02d}_train.parquet"
        val_path = cv_dir / f"fold_{i+1:02d}_val.parquet"

        df_train.to_parquet(train_path, index=False)
        df_val.to_parquet(val_path, index=False)

        actual_ratio = len(df_train) / max(1, (len(df_train) + len(df_val)))
        print(f"  ✓ Saved fold {i+1:02d} (left-out: {left_out_allele}, val-ids {val_ids}): "
              f"Train={len(df_train):,}, Val={len(df_val):,}, Ratio={actual_ratio:.3f}")

        # Memory cleanup
        del df_train, df_val

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