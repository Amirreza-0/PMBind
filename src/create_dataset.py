#!/usr/bin/env python
"""
Create training / validation splits for the binding-affinity dataset
and produce two fixed hold-out sets:

    • left_out_10k.csv         – 10 000 samples representative of the
                                 global class distribution
    • rare_alleles_1k.csv      – 1 000 samples originating from
                                 the rarest HLA alleles

All remaining data are split into K folds with
“leave-one-cluster-out, stratified by the label”.
"""

from __future__ import annotations
import os
import pathlib
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm                                     # progress bars

# Your own helper (must be import-able)
from src.utils import create_k_fold_leave_one_cluster_out_stratified_cv


# ---------------------------------------------------------------------
# 1. CONFIGURATION – adjust if your paths change
# ---------------------------------------------------------------------
embedding_dataset = "PMGen_whole_seq"
mhc_class = 1

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]

BINDING_PQ_PATH    = ROOT_DIR / f"data/binding_affinity_data/binding_dataset_with_synthetic_negatives_class{mhc_class}.parquet"
NPZ_PATH           = ROOT_DIR / f"data/ESM/esmc_600m/{embedding_dataset}/mhc{mhc_class}_encodings.npz"
BENCHMARKS_PATH    = ROOT_DIR / f"data/Custom_dataset/benchmarks/mhc_{mhc_class}"

# Output
OUT_DIR            = ROOT_DIR / f"data/binding_affinity_data/splits/mhc_{mhc_class}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K                  = 10      # No. folds for cross-validation
RANDOM_SEED        = 42
N_LEFT_OUT         = 10_000
N_RARE_ALLELES     = 1_000
# ---------------------------------------------------------------------


# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────
def stratified_sample(
    df: pd.DataFrame,
    target_col: str,
    n_samples: int,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return two dataframes: (sampled, remainder)
    """
    # Work on a copy
    df_ = df.reset_index(drop=True)
    # The split generator needs test_size as *fraction*, so we compute it
    frac = n_samples / len(df_)
    if frac >= 1.0:
        raise ValueError("Requested more samples than rows available.")
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=frac,
        random_state=random_state
    )
    idx_sample, idx_rest = next(splitter.split(df_, df_[target_col]))
    return df_.iloc[idx_sample].reset_index(drop=True), \
           df_.iloc[idx_rest].reset_index(drop=True)


def sample_rare_allele_subset(
    df: pd.DataFrame,
    allele_col: str,
    n_rows_target: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collect rows that belong to the rarest alleles until
    ≥ n_rows_target have been accumulated.
    Returns (rare_subset, remainder)
    """
    allele_counts = df[allele_col].value_counts(ascending=True)
    rare_alleles  = []
    running_total = 0

    for allele, cnt in allele_counts.items():
        rare_alleles.append(allele)
        running_total += cnt
        if running_total >= n_rows_target:
            break

    mask_rare = df[allele_col].isin(rare_alleles)
    rare_df   = df[mask_rare].reset_index(drop=True)
    rest_df   = df[~mask_rare].reset_index(drop=True)
    return rare_df, rest_df


# ──────────────────────────────────────────────────────────────────────
# Main routine
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # 1) ----------------------------------------------------------------
    print("→ Loading binding-affinity dataset")
    df = pd.read_parquet(BINDING_PQ_PATH)
    print(f"{len(df):,} rows (mhc_class == {mhc_class})")

    # -------------------------------------------------------------------
    # 2) Create two fixed hold-out sets
    # -------------------------------------------------------------------
    print("→ Creating left_out_10k stratified hold-out")
    left_out_10k, df = stratified_sample(
        df,
        target_col="assigned_label",
        n_samples=N_LEFT_OUT,
        random_state=RANDOM_SEED,
    )
    left_out_10k.to_csv(OUT_DIR / "left_out_10k.csv", index=False)
    print(f"Saved left_out_10k.csv with {len(left_out_10k):,} rows")

    print("→ Extracting rare_alleles_1k hold-out")
    rare_alleles_1k, df = sample_rare_allele_subset(
        df,
        allele_col="allele",
        n_rows_target=N_RARE_ALLELES
    )
    rare_alleles_1k.to_csv(OUT_DIR / "rare_alleles_1k.csv", index=False)
    print(f"Saved rare_alleles_1k.csv with {len(rare_alleles_1k):,} rows")

    print(f"Remaining for CV: {len(df):,} rows")

    # -------------------------------------------------------------------
    # 3) Build K “leave-one-cluster-out” folds (stratified by label)
    # -------------------------------------------------------------------
    print("→ Creating cross-validation folds")
    folds = create_k_fold_leave_one_cluster_out_stratified_cv(
        df,
        k=K,
        cluster_col="cluster",
        target_col="assigned_label",
        id_col="allele",
        random_state=RANDOM_SEED
    )

    folds_dir = OUT_DIR / "cv_folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    held_out_clusters_path = folds_dir / "held_out_clusters.txt"
    if held_out_clusters_path.exists():
        held_out_clusters_path.unlink()

    for i, (df_train, df_val, held_out_clusters) in enumerate(folds, start=1):
        train_path = folds_dir / f"fold_{i:02d}_train.parquet"
        val_path   = folds_dir / f"fold_{i:02d}_val.parquet"
        df_train.to_parquet(train_path, index=False, engine="pyarrow", compression="zstd")
        df_val.to_parquet(val_path,   index=False, engine="pyarrow", compression="zstd")
        print(f"Saved fold {i:02d} – train: {len(df_train):,}  val: {len(df_val):,}")
        with open(held_out_clusters_path, "a") as f:
            f.write(f"Fold {i:02d}: {', '.join(map(str, held_out_clusters))}\n")

    # -------------------------------------------------------------------
    # 4) Prepare benchmark data (optional – only if files exist)
    # -------------------------------------------------------------------
    print("→ Processing benchmark datasets")
    if BENCHMARKS_PATH.exists():
        for folder in BENCHMARKS_PATH.iterdir():
            if not folder.is_dir():
                continue
            for f in folder.glob("*.[ct]sv"):
                print(f"   – Loading {f.name}")
                if f.suffix == ".csv":
                    bench = pd.read_csv(f)
                else:
                    bench = pd.read_csv(f, sep="\t")
                # Filter to the correct mhc_class
                if "mhc_class" in bench.columns:
                    bench = bench[bench["mhc_class"] == mhc_class]
                out_name = OUT_DIR / "benchmarks" / folder.name
                out_name.mkdir(parents=True, exist_ok=True)
                bench.to_parquet(
                    out_name / f"{f.stem}.parquet",
                    index=False,
                    engine="pyarrow",
                    compression="zstd"
                )
                print(f"      ↳ saved {f.stem}.parquet")

    # -------------------------------------------------------------------
    print("✓ All done")


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
# import os
# import pathlib
# import re
# from tqdm import tqdm                          # progress bars
# from typing import Dict
# from src.utils import create_k_fold_leave_one_cluster_out_stratified_cv
#
# # ---------------------------------------------------------------------
# # 1. CONFIGURATION – adjust if your paths change
# # ---------------------------------------------------------------------
# embedding_dataset = "PMGen_whole_seq" # "PMGen_sequences" # "NetMHCpan_dataset"
# mhc_class = 1
# BINDING_CSV_PATH   = pathlib.Path(f"../data/binding_affinity_data/binding_dataset_with_synthetic_negatives_class{mhc_class}.parquet") # Training dataset
# ANALYSIS_CSV_PATH = pathlib.Path(f"../data/binding_affinity_data/allele_stats_class{mhc_class}_with_seq.csv")  # Analysis dataset
# NPZ_PATH   = pathlib.Path(
#     f"../data/ESM/esmc_600m/{embedding_dataset}/mhc{mhc_class}_encodings.npz"
# )
# BENCHMARKS_PATH = pathlib.Path(
#     f"../data/Custom_dataset/benchmarks/mhc_{mhc_class}"
# )
#
# K = 10  # Number of folds for cross-validation
#
#
# # TODO create left_out_10k.csv that contains 10,000 rows that represent the whole dataset and remove these rows from the original dataset
# # TODO create a rare_alleles_1k.csv that contains 1,000 rows with rare alleles and remove these rows from the original dataset
# # TODO create K folds with leave one cluster out cross-validation
# # TODO process and create benchmark datasets for MHC class I and II
#
# # eg. usage:
# df_train_bal, df_val_bal, left_out_cluster = create_k_fold_leave_one_cluster_out_stratified_cv(k=K, cluster_col="cluster", target_col="assigned_label", id_col="allele")


# def create_k_fold_leave_one_out_stratified_cv(
#     df: pd.DataFrame,
#     k: int = 5,
#     target_col: str = "label",
#     id_col: str = "allele",
#     subset_prop: float = 1.0,
#     train_size: float = 0.8,
#     random_state: int = 42,
#     augmentation: str = None  # "down_sampling" or "GNUSS"
# ):
#     """
#     Build *k* folds such that
#
#     1. **One whole ID (group) is left out of both train & val** (`left_out_id`).
#     2. **Validation contains exactly one additional ID** (`val_only_id`)
#        that never appears in train.
#     3. Remaining rows are split *stratified* on `target_col`
#        (`train_size` fraction for training).
#     4. Train & val are **down-sampled** to perfectly balanced label counts.
#
#     Returns
#     -------
#     list[tuple[pd.DataFrame, pd.DataFrame, Hashable]]
#         Each tuple = (train_df, val_df, left_out_id).
#     """
#     rng = np.random.RandomState(random_state)
#     if subset_prop < 1.0:
#         if subset_prop <= 0.0 or subset_prop > 1.0:
#             raise ValueError(f"subset_prop must be in (0, 1], got {subset_prop}")
#         # Take a random subset of the DataFrame
#         print(f"Taking {subset_prop * 100:.2f}% of the data for k-fold CV")
#         df = df.sample(frac=subset_prop, random_state=random_state).reset_index(drop=True)
#
#     # --- pick the k IDs that will be held out completely -------------------
#     unique_ids = df[id_col].unique()
#     if k > len(unique_ids):
#         raise ValueError(f"k={k} > unique {id_col} count ({len(unique_ids)})")
#     left_out_ids = rng.choice(unique_ids, size=k, replace=False)
#
#     folds = []
#     for fold_idx, left_out_id in enumerate(left_out_ids, 1):
#         fold_seed = random_state + fold_idx
#         mask_left_out = df[id_col] == left_out_id
#         working_df = df.loc[~mask_left_out].copy()
#
#         # ---------------------------------------------------------------
#         # 1) choose ONE id that will appear *only* in validation
#         #    (GroupShuffleSplit with test_size=1 group)
#         # ---------------------------------------------------------------
#         gss = GroupShuffleSplit(
#             n_splits=1, test_size=1, random_state=fold_seed
#         )
#         (train_groups_idx, val_only_groups_idx), = gss.split(
#             X=np.zeros(len(working_df)), y=None, groups=working_df[id_col]
#         )
#         val_only_group_id = working_df.iloc[val_only_groups_idx][id_col].unique()[0]
#
#         mask_val_only = working_df[id_col] == val_only_group_id
#         df_val_only = working_df[mask_val_only]
#         df_eligible = working_df[~mask_val_only]
#
#         # ---------------------------------------------------------------
#         # 2) stratified split of *eligible* rows
#         # ---------------------------------------------------------------
#         sss = StratifiedShuffleSplit(
#             n_splits=1, train_size=train_size, random_state=fold_seed
#         )
#         train_idx, extra_val_idx = next(
#             sss.split(df_eligible, df_eligible[target_col])
#         )
#         df_train = df_eligible.iloc[train_idx]
#         df_val   = pd.concat(
#             [df_val_only, df_eligible.iloc[extra_val_idx]], ignore_index=True
#         )
#
#         print(f"Fold size: train={len(df_train)}, val={len(df_val)} | ")
#
#         # ---------------------------------------------------------------
#         # 3) balance train and val via down-sampling
#         # ---------------------------------------------------------------
#         def _balance_down_sampling(frame: pd.DataFrame) -> pd.DataFrame:
#             min_count = frame[target_col].value_counts().min()
#             print(f"Balancing {len(frame)} rows to {min_count} per class")
#             balanced_parts = [
#                 resample(
#                     frame[frame[target_col] == lbl],
#                     replace=False,
#                     n_samples=min_count,
#                     random_state=fold_seed,
#                 )
#                 for lbl in frame[target_col].unique()
#             ]
#             return pd.concat(balanced_parts, ignore_index=True)
#
#         def _balance_GNUSS(frame: pd.DataFrame) -> pd.DataFrame:
#             """
#             Balance the DataFrame by upsampling the minority class with Gaussian noise.
#             """
#             # Determine label counts and the maximum class size
#             counts = frame[target_col].value_counts()
#             max_count = counts.max()
#
#             # Identify numeric columns for noise injection
#             numeric_cols = frame.select_dtypes(include="number").columns
#
#             balanced_parts = []
#             for label, count in counts.items():
#                 df_label = frame[frame[target_col] == label]
#                 balanced_parts.append(df_label)
#                 if count < max_count:
#                     # Upsample with replacement
#                     n_needed = max_count - count
#                     sampled = df_label.sample(n=n_needed, replace=True, random_state=fold_seed)
#                     # Add Gaussian noise to numeric features
#                     noise = pd.DataFrame(
#                         rng.normal(loc=0, scale=1e-6, size=(n_needed, len(numeric_cols))),
#                         columns=numeric_cols,
#                         index=sampled.index
#                     )
#                     sampled[numeric_cols] = sampled[numeric_cols] + noise
#                     balanced_parts.append(sampled)
#
#             # Combine and return
#             return pd.concat(balanced_parts).reset_index(drop=True)
#
#         if augmentation == "GNUSS":
#             df_train_bal = _balance_GNUSS(df_train)
#             df_val_bal   = _balance_GNUSS(df_val)
#         elif augmentation == "down_sampling":  # default to down-sampling
#             df_train_bal = _balance_down_sampling(df_train)
#             df_val_bal   = _balance_down_sampling(df_val)
#         elif not augmentation:
#             df_train_bal = df_train.copy()
#             df_val_bal = df_val.copy()
#             print("No augmentation applied, using original train and val sets.")
#         else:
#             raise ValueError(f"Unknown augmentation method: {augmentation}")
#
#         # Shuffle both datasets to avoid any ordering bias
#         df_train_bal = df_train_bal.sample(frac=1.0, random_state=fold_seed).reset_index(drop=True)
#         df_val_bal = df_val_bal.sample(frac=1.0, random_state=fold_seed).reset_index(drop=True)
#         folds.append((df_train_bal, df_val_bal, left_out_id))
#
#         print(
#             f"[fold {fold_idx}/{k}] left-out={left_out_id} | "
#             f"val-only={val_only_group_id} | "
#             f"train={len(df_train_bal)}, val={len(df_val_bal)}"
#         )
#
#     return folds


