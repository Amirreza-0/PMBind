import os
import pathlib
import sys
from typing import Set, Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
MHC_CLASS = 1
RANDOM_SEED = 999
K_FOLDS = 5
MIN_ALLELES_FOR_CV = 15  # Minimum alleles needed for CV after test set extraction
N_TVAL_FOLD_SAMPLES = 20  # Number of alleles to leave out for Ensemble Validation
N_VAL_FOLD_SAMPLES = 20  # Number of alleles to leave out for validation in each fold
TAKE_SUBSET = True  # Whether to save folds as subsets of df/k
LEAVE_ALLELE_GROUP_OUT = True  # Whether to leave one allele group out completely (True) or just one allele (False)

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

# manual test alleles
test_alleles = ["BoLA-3:00101", "DLA-8850801",
                "Eqca-1600101", "Gogo-B0101",
                "H-2-Kk", "HLA-A*02:50",
                "HLA-B*45:06",
                "HLA-C*12:12", "HLA-E01:03",
                "Mamu-A7*00103", "Mamu-B*06502",
                "Patr-B17:01", "SLA-107:01",
                "HLA-C*18:01"]

tval_alleles = ["SLA-107:02",
                "Patr-B24:01",
                "Mamu-B*08701",
                "Mamu-A1*02601",
                "HLA-C*15:04",
                "HLA-B*15:42",
                "HLA-A*03:19",
                "H-2-Dd",
                "BoLA-6:04101"]

major_allele_groups = ['MAMU', 'PATR', 'SLA', 'BOLA', 'DLA', 'H-2', 'HLA-A', 'HLA-B', 'HLA-C', 'HLA-E', 'HLA-DRB',
                       'HLA-DQA', 'HLA-DQB', 'HLA-DPA', 'HLA-DPB', 'EQCA', 'GOGO']


def train_val_split(
        df: pd.DataFrame,
        k: int = 5,
        target_col: str = "assigned_label",
        id_col: str = "mhc_embedding_key",
        train_size: float = 0.8,
        random_state: int = 42,
        n_val_ids: int = 1,
        take_subset: bool = False,
        LEAVE_ALLELE_GROUP_OUT: bool = False,
        major_allele_groups: list = None
):
    """
    Perform k-fold cross-validation split with option to leave out entire allele groups.

    When LEAVE_ALLELE_GROUP_OUT is True:
    - Selects k major allele groups to leave out (one per fold)
    - Each fold leaves out ALL alleles from one major group
    - This results in more left_out_ids compared to individual allele selection
    """
    # --------
    rng = np.random.RandomState(random_state)

    if LEAVE_ALLELE_GROUP_OUT:
        if major_allele_groups is None:
            raise ValueError("major_allele_groups must be provided when LEAVE_ALLELE_GROUP_OUT is True")

        # Group alleles by major groups (e.g., HLA-A, HLA-B, MAMU, etc.)
        def get_major_group(allele: str) -> str:
            allele = str(allele).upper()
            for group in major_allele_groups:
                if allele.startswith(group):
                    return group
            return "OTHER"

        df['major_group'] = df[id_col].apply(get_major_group)
        unique_groups = df['major_group'].unique()

        if len(unique_groups) < k:
            raise ValueError(f"Not enough unique allele groups ({len(unique_groups)}) for {k}-fold CV.")

        # Select k groups for k-fold CV (one group left out per fold)
        left_out_groups = rng.choice(unique_groups, size=k, replace=False)

        # Create a mapping of groups to their IDs
        group_to_ids = {}
        for group in left_out_groups:
            ids_in_group = df[df['major_group'] == group][id_col].unique().tolist()
            group_to_ids[group] = ids_in_group

        # Calculate all IDs that will be left out across all folds
        all_left_out_ids = []
        for group in left_out_groups:
            all_left_out_ids.extend(group_to_ids[group])
        all_left_out_ids = np.array(all_left_out_ids)

        print(f"Selected allele groups for left-out: {left_out_groups}")
        print(f"Total number of left-out IDs: {len(all_left_out_ids)}")
        for group in left_out_groups:
            print(f"  Group {group}: {len(group_to_ids[group])} IDs")

        # Create the pool of data not in any left-out group
        left_out_df = df[df[id_col].isin(all_left_out_ids)].copy()
        df_rest_pool = df[~df[id_col].isin(all_left_out_ids)].copy()

        # Clean up the temporary column
        df = df.drop(columns=['major_group'])
        left_out_df = left_out_df.drop(columns=['major_group'])
        df_rest_pool = df_rest_pool.drop(columns=['major_group'])

    else:
        # Simple random selection of k unique IDs
        unique_ids = df[id_col].unique()
        if len(unique_ids) < k:
            raise ValueError(f"Not enough unique IDs ({len(unique_ids)}) for {k}-fold CV.")

        left_out_ids = rng.choice(unique_ids, size=k, replace=False)
        left_out_df = df[df[id_col].isin(left_out_ids)].copy()
        df_rest_pool = df[~df[id_col].isin(left_out_ids)].copy()

        # Create a dummy mapping for consistency
        group_to_ids = {id: [id] for id in left_out_ids}
        left_out_groups = left_out_ids  # For iteration purposes

    fold_size = int(len(df_rest_pool) / k)
    folds = []

    for fold_idx, left_out_group in enumerate(left_out_groups, 1):
        fold_seed = random_state + fold_idx
        rng = np.random.RandomState(fold_seed)

        # Get the IDs for this fold's left-out group
        if LEAVE_ALLELE_GROUP_OUT:
            fold_left_out_ids = group_to_ids[left_out_group]
        else:
            fold_left_out_ids = [left_out_group]  # Single ID case

        # Create extended dataset: rest_pool + all left_out data except current fold's left-out IDs
        df_extended = pd.concat([
            df_rest_pool,
            left_out_df[~left_out_df[id_col].isin(fold_left_out_ids)]
        ], ignore_index=True)

        if take_subset:
            if fold_size < len(df_extended):
                # Sample from df_extended
                df_fold = df_extended.sample(n=fold_size, random_state=fold_seed).copy()
                # Identify which rows from the sample came from the rest_pool
                indices_to_remove = df_rest_pool.index.intersection(df_fold.index)
                # Remove these rows from df_rest_pool for the next fold
                if not indices_to_remove.empty:
                    df_rest_pool = df_rest_pool.drop(indices_to_remove)
            else:
                df_fold = df_extended.copy()
                # If we use all data, clear the rest pool
                df_rest_pool = df_rest_pool.iloc[0:0].copy()
        else:
            df_fold = df_extended.copy()

        # Verify that left-out IDs are not in the fold
        for lid in fold_left_out_ids:
            if lid in df_fold[id_col].values:
                raise ValueError(f"Left-out ID {lid} found in fold {fold_idx} data.")

        # DEBUG: Print fold statistics
        print(f"\n[DEBUG fold {fold_idx}]")
        if LEAVE_ALLELE_GROUP_OUT:
            print(f"  Left-out group: {left_out_group} ({len(fold_left_out_ids)} IDs)")
        else:
            print(f"  Left-out ID: {left_out_group}")
        print(f"  Number of unique IDs in fold: {df_fold[id_col].nunique()}")
        print(f"  Total rows in fold: {len(df_fold)}")

        # Select validation IDs from the fold
        id_counts = df_fold[id_col].value_counts()
        id_counts = id_counts[id_counts > 0]  # Remove IDs with 0 occurrences

        # Select validation data: aim for approximately (1 - train_size) of fold size
        fold_val = pd.DataFrame(columns=df_fold.columns)
        val_target_size = int((1 - train_size) * len(df_fold))

        # Collect validation IDs until we reach the target size
        val_ids = set()
        while len(fold_val) < val_target_size and len(val_ids) < len(id_counts):
            # Select a random ID that hasn't been selected yet
            available_ids = [id for id in id_counts.index if id not in val_ids]
            if not available_ids:
                break
            val_id = rng.choice(available_ids)
            val_ids.add(val_id)
            fold_val = pd.concat([fold_val, df_fold[df_fold[id_col] == val_id]], ignore_index=True)

        print(f"  Selected {len(val_ids)} unique IDs for validation ({len(fold_val)} rows)")

        # Add left-out IDs' data to validation set
        left_out_data = left_out_df[left_out_df[id_col].isin(fold_left_out_ids)]
        fold_val = pd.concat([fold_val, left_out_data], ignore_index=True)

        print(f"  Added {len(left_out_data)} rows from left-out IDs to validation")
        print(f"  Final validation size: {len(fold_val)} rows")

        # Select remaining rows for training
        fold_train = df_fold[~df_fold[id_col].isin(val_ids)].copy()

        if fold_train.empty:
            raise ValueError(f"No remaining data for training after leaving out IDs")

        # Final verification
        for lid in fold_left_out_ids:
            if lid in fold_train[id_col].values:
                raise ValueError(f"Left-out ID {lid} found in training data.")

        if any(vid in fold_train[id_col].values for vid in val_ids):
            overlapping = [vid for vid in val_ids if vid in fold_train[id_col].values]
            raise ValueError(f"Validation IDs {overlapping} found in training data.")

        print(f"[fold {fold_idx}/{k}] "
              f"left-out={'group:' + str(left_out_group) if LEAVE_ALLELE_GROUP_OUT else left_out_group} "
              f"({len(fold_left_out_ids)} IDs) | "
              f"val-only={len(val_ids)} IDs | "
              f"train={len(fold_train)}, val={len(fold_val)}")

        # Store the fold
        folds.append((fold_train, fold_val, fold_left_out_ids, val_ids))

    return folds


def clean_key_vectorized(s: pd.Series) -> pd.Series:
    """Vectorized allele cleaning."""
    return s.astype(str).str.translate(KEY_TRANS).str.upper()


def extract_test_tval_set(df: pd.DataFrame,
                          test_alleles: List[str],
                          tval_alleles: List[str],
                          id_col: str = "mhc_embedding_key") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract test set from the dataset.

    Returns: (remaining_df, test_df)
    """
    print("\n=== Extracting Test Set ===")

    # Split dataset into test alleles and the rest
    test_mask = df[id_col].isin(test_alleles)
    test_df = df.loc[test_mask].copy()
    remaining_df = df.loc[~test_mask].copy().reset_index(drop=True)
    print(f"Test alleles data: {len(test_df):,} samples from {test_df[id_col].nunique()} alleles")

    tval_mask = remaining_df[id_col].isin(tval_alleles)
    tval_df = remaining_df.loc[tval_mask].copy()
    remaining_df2 = remaining_df.loc[~tval_mask].copy().reset_index(drop=True)

    # verify we have enough alleles for CV after test set extraction
    if remaining_df2[id_col].nunique() < MIN_ALLELES_FOR_CV:
        raise ValueError(f"Not enough alleles ({remaining_df2[id_col].nunique()}) left for CV after extracting "
                         f"test set from manual test alleles. Minimum required: {MIN_ALLELES_FOR_CV}")

    # verify if test alleles are not in remaining data
    if any(allele in remaining_df2[id_col].values for allele in test_alleles):
        raise ValueError("Test alleles found in remaining data after extraction.")

    # verify if tval alleles are not in remaining data
    if any(allele in remaining_df2[id_col].values for allele in tval_alleles):
        raise ValueError("TVal alleles found in remaining data after extraction.")

    return remaining_df2, test_df, tval_df


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


# Main execution
def main():
    """Main execution function."""
    # Redirect stdout to a log file
    log_file_path = OUT_DIR / "dataset_creation_log.txt"

    # Create a tee-like functionality to write to both file and console
    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    # Open log file and redirect stdout
    with open(log_file_path, 'w') as log_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, log_file)

        try:
            print("=== Simplified K-Fold Leave-One-Allele-Out Cross-Validation with Test Set ===")
            print(f"K folds: {K_FOLDS}")
            print(f"Train size: {TRAIN_SIZE}")
            print(f"Augmentation: {AUGMENTATION}")
            print(f"Random seed: {RANDOM_SEED}")
            print(f"Output directory: {OUT_DIR}")
            print(f"Log file: {log_file_path}")

            _execute_main_logic()

        finally:
            sys.stdout = original_stdout
            print(f"✓ All output has been saved to: {log_file_path}")


def _execute_main_logic():
    """Execute the main logic (separated for cleaner stdout handling)."""
    print("=== Simplified K-Fold Leave-One-Allele-Out Cross-Validation with Test Set ===")
    print(f"K folds: {K_FOLDS}")
    print(f"Train size: {TRAIN_SIZE}")
    print(f"Augmentation: {AUGMENTATION}")
    print(f"Random seed: {RANDOM_SEED}")

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
    df_remaining, df_test, df_tval = extract_test_tval_set(
        df, test_alleles=test_alleles, tval_alleles=tval_alleles, id_col='mhc_embedding_key'
    )
    # Save test set
    test_set_path = OUT_DIR / "test_set_rarest_alleles.parquet"
    df_test.to_parquet(test_set_path, index=False)
    print(f"✓ Test set saved to: {test_set_path}")

    # Save tval set
    tval_set_path = OUT_DIR / "tval_set_rarest_alleles.parquet"
    df_tval.to_parquet(tval_set_path, index=False)
    print(f"✓ TVal set saved to: {tval_set_path}")

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
        n_val_ids=N_VAL_FOLD_SAMPLES,  # number of alleles to leave out for validation in each fold
        take_subset=TAKE_SUBSET
    )

    cv_dir = OUT_DIR / "cv_folds"
    cv_dir.mkdir(exist_ok=True)

    for i, (df_train, df_val, left_out_allele, val_ids) in enumerate(folds):
        train_path = cv_dir / f"fold_{i + 1:02d}_train.parquet"
        val_path = cv_dir / f"fold_{i + 1:02d}_val.parquet"

        df_train.to_parquet(train_path, index=False)
        df_val.to_parquet(val_path, index=False)

        actual_ratio = len(df_train) / max(1, (len(df_train) + len(df_val)))
        print(f"  ✓ Saved fold {i + 1:02d} (left-out: {left_out_allele}, val-ids {val_ids}): "
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
        'test_set_alleles_manual': len(test_alleles),
        'test_set_alleles_actual': actual_test_alleles,
        'tval_set_samples': int(len(df_tval)),
        'tval_set_alleles_manual': len(tval_alleles),
        'tval_set_alleles_actual': int(df_tval['mhc_embedding_key'].nunique()),
        'cv_samples': int(len(df_remaining)),
        'cv_alleles': int(df_remaining['mhc_embedding_key'].nunique()),
        'n_folds': int(K_FOLDS),
        'test_set_path': str(test_set_path),
        'tval_set_path': str(tval_set_path),
        'cv_folds_path': str(cv_dir),
        'benchmark_cache_path': str(BENCHMARK_CACHE_PQ if BENCHMARK_CACHE_PQ.exists() else BENCHMARK_CACHE_PKL),
    }

    summary_path = OUT_DIR / "data_split_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=== Data Split Summary ===\n")
        for key, value in summary_info.items():
            f.write(f"{key}: {value}\n")

    print(f"\n✓ Successfully created {K_FOLDS} cross-validation folds")
    print(f"✓ Test set: {len(df_test):,} samples from {actual_test_alleles} manual alleles")
    print(f"✓ TVal set: {len(df_tval):,} samples from {df_tval['mhc_embedding_key'].nunique()} manual alleles")
    print(f"✓ CV data: {len(df_remaining):,} samples from {df_remaining['mhc_embedding_key'].nunique()} alleles")
    print(f"✓ Files saved in: {cv_dir}")
    print(f"✓ Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()