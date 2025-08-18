#!/usr/bin/env python
"""
Create training / validation splits for the binding-affinity dataset
and produce two fixed hold-out sets:

    • left_out_10k.csv         – 10 000 samples representative of the
                                 global class distribution
    • rare_cluster_1k.csv      – 1 000 samples originating from
                                 the rarest HLA clusters

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

BINDING_PQ_PATH    = ROOT_DIR / f"data/binding_affinity_data/binding_dataset_with_swapped_negatives_class{mhc_class}.parquet"
NPZ_PATH           = ROOT_DIR / f"data/ESM/esmc_600m/{embedding_dataset}/mhc{mhc_class}_encodings.npz"
BENCHMARKS_PATH    = ROOT_DIR / f"data/mhc_{mhc_class}/benchmarks/" # must be coppied.

# Output
OUT_DIR            = ROOT_DIR / f"data/binding_affinity_data/cross_validation_dataset/mhc_{mhc_class}"
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
