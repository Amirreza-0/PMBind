#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from glob import glob
from pathlib import Path
import pandas as pd
from typing import Sequence, Dict

# ------------- configuration -------------------------------------------------
ROOT_DIR = Path("../../data/binding_affinity_data")  # parent folder that contains 1/, 2/, ...
CLASSES = (1, 2)  # run both class I and class II by default
SEQ_DIR = Path("../../data/ESM/esmc_600m/PMGen_whole_seq")  # where mhc1_sequences.csv, mhc2_sequences.csv live
# SEQ_DIR = Path("../../data/alleles")  # this is the path to get alignment sequences

# helper – build MHC-embedding key (exactly the same rule you used before)
def build_key(allele: str, seq_map: Dict = None) -> str:
    """Remove ':', '*', white-space; replace '/' with '_' and upper-case."""
    key = allele.replace(":", "").replace("*", "").replace(" ", "").replace("mice-", "").replace("-", "")
    key = key.replace("/", "_").upper()
    # get the seq_map keys that start with key
    if seq_map is not None:
        # if key is in seq_map, return it
        if key in seq_map:
            return key
        # else, find the first key that starts with key
        for k in seq_map.keys():
            if k.startswith(key):
                return k
    return key

# 1) concatenate all TSV + Parquet files from one directory
def concatenate_files(one_class_dir: Path) -> pd.DataFrame:
    tsvs = glob(str(one_class_dir / "*.tsv"))
    parquets = glob(str(one_class_dir / "*.parquet"))

    dfs = []

    # TSV
    for f in tsvs:
        dfs.append(pd.read_csv(f, sep="\t"))

    # Parquet
    for f in parquets:
        df = pd.read_parquet(f)
        if "assigned_label" in df.columns:  # keep only negatives
            df = df[df["assigned_label"] == 0]
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No TSV or Parquet found in {one_class_dir}")

    df = pd.concat(dfs, ignore_index=True)

    # allele key
    df["mhc_embedding_key"] = df["allele"].astype(str).apply(build_key)

    # choose a canonical allele spelling (the longest string per key)
    df["allele"] = (
        df.groupby("mhc_embedding_key")["allele"]
        .transform(lambda x: x.loc[x.str.len().idxmax()])
    )

    # fill NaN sources and deduplicate
    df["source"] = df["source"].fillna("netmhcpan_all_pred_negs")
    df = df.drop_duplicates(subset=["mhc_embedding_key", "long_mer"])

    return df


# 2) statistics per allele
def allele_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    positives / negatives, length range & most common source – per allele.
    """

    def most_common(col: pd.Series) -> str:
        return col.mode().iloc[0] if not col.mode().empty else "N/A"

    def most_common_positive(sub: pd.DataFrame) -> str:
        pos = sub[sub["assigned_label"] == 1]["source"]
        return pos.mode().iloc[0] if not pos.mode().empty else "N/A"

    out = (
        df.groupby("allele")
        .apply(lambda g: pd.Series({
            "positives": (g["assigned_label"] == 1).sum(),
            "negatives": (g["assigned_label"] == 0).sum(),
            "min_length": g["long_mer"].str.len().min(),
            "max_length": g["long_mer"].str.len().max(),
            "most_common_source": most_common(g["source"]),
            "common_source_positive": most_common_positive(g),
        }))
        .reset_index()
    )
    return out


# 3) attach α/β MHC sequences (optional)
def attach_mhc_sequences(stats_df: pd.DataFrame,
                         seq_map: pd.DataFrame) -> pd.DataFrame:
    """
    seq_map must contain columns ['key', 'mhc_sequence'].
    """
    key2seq: Dict[str, str] = dict(zip(seq_map["key"], seq_map["mhc_sequence"]))

    # harmonize key2seq
    key2seq = {build_key(k, key2seq): v for k, v in key2seq.items()}

    def fetch(allele: str) -> str | None:
        # strip, upper etc.
        # cleaned = allele.replace(":", "").replace("*", "").replace(" ", "")
        # multi-chain?  e.g. DRB1_0401/DRA_0101
        if "/" in allele or "_" in allele:
            cleaned = allele.replace("/", "_")
            parts = cleaned.split("_")
            if len(parts) != 2:
                return None
            seqs = [key2seq.get(build_key(p, key2seq), None) for p in parts]
            if None in seqs:
                return None
            return "/".join(seqs)
        else:
            return key2seq.get(build_key(allele, key2seq), None)

    stats_df["sequence"] = stats_df["allele"].map(fetch)
    return stats_df


# main loop
def main(mhc_classes: Sequence[int] = CLASSES) -> None:
    for cls in mhc_classes:
        # print(f"\n=== MHC class {cls} ===")
        # one_dir = ROOT_DIR / f"mhc{cls}"  # e.g. ../../data/binding_affinity_data/mhc1/
        # if not one_dir.is_dir():
        #     print(f"  WARNING: directory {one_dir} does not exist – skipped.")
        #     continue
        #
        # # 1) concatenate
        # concat_df = concatenate_files(one_dir)
        # concat_out = ROOT_DIR / f"concatenated_class{cls}.parquet"
        # concat_df.to_parquet(concat_out, index=False)
        # print(f"  • concatenated data -> {concat_out}")
        #
        # # 2) statistics
        # stats = allele_statistics(concat_df)
        # stats_out = ROOT_DIR / f"allele_stats_class{cls}.csv"
        # stats.to_csv(stats_out, index=False)
        # print(f"  • allele statistics  -> {stats_out}")

        ## Debug -
        # load stats
        stats_csv = ROOT_DIR / f"allele_stats_class{cls}.csv"
        if not stats_csv.exists():
            print(f"  WARNING: stats file {stats_csv} does not exist – skipped.")
            continue
        stats = pd.read_csv(stats_csv)

        # 3) add sequences (if mapping file exists)
        seq_csv = SEQ_DIR / f"mhc{cls}_encodings.csv"
        # seq_csv = SEQ_DIR / f"aligned_PMGen_class_{cls}.csv"
        if seq_csv.exists():
            seq_map = pd.read_csv(seq_csv)
            stats = attach_mhc_sequences(stats, seq_map)
            stats_out2 = ROOT_DIR / f"allele_stats_class{cls}_with_seq.csv"
            stats.to_csv(stats_out2, index=False)
            missing = stats["sequence"].isna().sum()
            print(f"  • sequences attached ({missing} missing) -> {stats_out2}")
        else:
            print(f"  • no sequence mapping file found for class {cls}")


if __name__ == "__main__":
    main()