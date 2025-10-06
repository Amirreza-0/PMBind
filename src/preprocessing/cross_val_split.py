# load all the data from "concatenated_class1_all.parquet"
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pathlib
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy.stats import entropy
from collections import Counter

# Performance/memory tweaks
pd.options.mode.copy_on_write = True
KEY_TRANS = str.maketrans({'*': '', ':': '', ' ': '', '/': '_'})  # for vectorized cleaning

MHC_CLASS = 2

# TODO run this in a for loop and generate sets with different seeds, in each set leave out one random allele.
def median_down_sampling(df_path, seed):
    df = pd.read_parquet(df_path)
    # take out the rows that has no positive or negative assigned_label
    df = df[df['assigned_label'].isin([0, 1])]
    # drop duplicates based on sequence and allele
    df = df.drop_duplicates(subset=['long_mer', 'allele'])
    df_copy = df.copy()
    # TODO drop pep2vec (we suspect pep2vec has a lot of noise)
    print(
        f"Initial counts: Positives={len(df[df['assigned_label'] == 1])}, Negatives={len(df[df['assigned_label'] == 0])}")
    # remove samples that have source of pep2vec
    # df = df[df['source'] != 'pep2vec']
    # print(f"Counts after removing pep2vec: Positives={len(df[df['assigned_label'] == 1])}, Negatives={len(df[df['assigned_label'] == 0])}")
    # Group by alleles and filter groups that have both positive and negative labels
    df = df.groupby('allele').filter(lambda g: g['assigned_label'].nunique() == 2)
    print(
        f"Counts after filtering for valid alleles: Positives={len(df[df['assigned_label'] == 1])}, Negatives={len(df[df['assigned_label'] == 0])}")
    # resample from negatives until a balanced set is achieved
    pos_count = len(df[df['assigned_label'] == 1])
    neg_count = len(df[df['assigned_label'] == 0])
    collected_negs_df = []
    # This loop is much faster because it drops all targeted rows in a single operation per iteration.
    while pos_count > len(collected_negs_df):
        # Get the current negative samples
        neg_df = df[df['assigned_label'] == 0]
        # Group the current negatives to find the median size
        neg_groups = neg_df.groupby('allele')
        median_count = int(neg_groups.size().median())
        print(f"current median: {median_count}")
        # If the median is 0 or no negatives can be dropped, break to avoid an infinite loop
        if median_count == 0:
            break
        indices_to_take = []
        # Iterate through negative groups to find rows to drop
        for name, group in tqdm(neg_groups, desc="Resampling negatives", total=len(neg_groups)):
            if len(group)//2 > median_count:
                # Sample the rows from the group that we want to drop
                drop_sample = group.sample(n=int(median_count), random_state=seed)
                indices_to_take.extend(drop_sample.index)
            else:
                print(f"drop all for allele {name} with count {len(group)}")
                indices_to_take.extend(group.index)

        if not indices_to_take:
            # Break if no rows were selected for dropping in this pass
            break
        # Drop all collected indices at once - this is far more efficient
        collected_negs_df = df.loc[indices_to_take]
        df = df.drop(indices_to_take)
        print(f"Collected {len(indices_to_take)} negatives.")
        # Update the negative count for the next iteration
        neg_count = len(df[df['assigned_label'] == 0])
        print(f"Updated counts: Positives={pos_count}, Negatives={neg_count}")
    print(f"Final counts: Positives={len(df[df['assigned_label'] == 1])}, df_remained_negatives={len(df[df['assigned_label'] == 0])}")
    print("collected negatives:", len(collected_negs_df))
    pos_df = df[df['assigned_label'] == 1]
    df2 = pd.concat([pos_df, collected_negs_df], ignore_index=True)
    # print the unique number of alleles
    print(f"Unique alleles: {df2['allele'].nunique()}, Total samples: {len(df2)}")
    # save to parquet
    df2.to_parquet(f"cleaned_balanced_class{MHC_CLASS}_seed_{seed}.parquet", index=False)
    # save stats allele,positives,negatives,min_length,max_length,most_common_source,common_source_positive,sequence
    stats = []
    grouped = df2.groupby('allele')
    for name, group in grouped:
        positives = len(group[group['assigned_label'] == 1])
        negatives = len(group[group['assigned_label'] == 0])
        min_length = group['long_mer'].str.len().min()
        max_length = group['long_mer'].str.len().max()
        most_common_source = group['source'].mode()[0] if not group['source'].mode().empty else None
        common_source_positive = group[group['assigned_label'] == 1]['source'].mode()[0] if not group[group['assigned_label'] == 1]['source'].mode().empty else None
        stats.append([name, positives, negatives, min_length, max_length, most_common_source, common_source_positive])
    stats_df = pd.DataFrame(stats, columns=['allele', 'positives', 'negatives', 'min_length', 'max_length', 'most_common_source', 'common_source_positive'])
    stats_df.to_csv(f"allele_stats_seed_{seed}.csv", index=False)

    dropped_df = _anti_join(df_copy, df2)

    # get value counts of unique alleles in dropped_df
    dropped_allele_counts = dropped_df['allele'].value_counts()
    print("Dropped allele counts:")
    print(dropped_allele_counts)
    # take the 200 rarest alleles from dropped_df (or fewer if not enough)
    n_rare = min(200, len(dropped_allele_counts))
    rare_alleles = dropped_allele_counts.nsmallest(n_rare).index.tolist()
    # select all samples belonging to those rare alleles
    rare_samples = dropped_df[dropped_df['allele'].isin(rare_alleles)].reset_index(drop=True)
    print(f"Selected {len(rare_samples)} samples from {len(rare_alleles)} rare alleles.")
    return df2, rare_samples


def clean_key(allele_key: str) -> str:
    """
    Clean allele keys by removing special characters and converting to uppercase.
    This is useful for matching keys in embedding dictionaries.
    """
    if allele_key is None:
        return "None"
    mapping = str.maketrans({'*': '', ':': '', ' ': '', '/': '_'})
    return allele_key.translate(mapping).upper()

def clean_key_vectorized(s: pd.Series) -> pd.Series:
    """Vectorized allele cleaning."""
    return s.astype(str).str.translate(KEY_TRANS).str.upper()

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

    return bench_pairs

def remove_benchmark_samples(fold, benchmarks_dir="data/benchmarks"):
    benchmark_path = pathlib.Path(benchmarks_dir)
    benchmark_df_cache_path = benchmark_path / "bench_cache.parquet"
    print(benchmark_path)
    benchmark_df_cache = pd.DataFrame()
    files = list(benchmark_path.rglob("*.parquet")) + list(benchmark_path.rglob("*.csv")) + list(benchmark_path.rglob("*.tsv"))
    print(f"Found {len(files)} benchmark files")
    for file in files:
        print(file)
        bench_df = pd.read_parquet(benchmark_df_cache_path) if benchmark_df_cache_path.exists() else None
        if bench_df is not None:
            benchmark_df_cache = pd.concat([benchmark_df_cache, bench_df], ignore_index=True).drop_duplicates()

    # Build benchmark pairs including a 'source' column derived from the filename stem
    # files = list(benchmark_path.rglob("*.parquet")) + list(benchmark_path.rglob("*.csv")) + list(benchmark_path.rglob("*.tsv"))
    # print(f"Found {len(files)} benchmark files")
    # all_chunks = []
    # for f in files:
    #     try:
    #         dfp = _read_benchmark_file_to_pairs_df(f)
    #         if dfp is not None and not dfp.empty:
    #             dfp["source"] = f.stem
    #             all_chunks.append(dfp)
    #     except Exception as e:
    #         print(f"Warning: failed to read benchmark file {f}: {e}")

    # if all_chunks:
    #     benchmark_df_cache = pd.concat(all_chunks, ignore_index=True).drop_duplicates()
    # else:
    #     benchmark_df_cache = pd.DataFrame(columns=["allele", "long_mer", "source"])

    # if not benchmark_df_cache.empty:
    #     # Normalize expected columns
    #     if 'allele' not in benchmark_df_cache.columns and 'mhc_embedding_key' in benchmark_df_cache.columns:
    #         benchmark_df_cache = benchmark_df_cache.rename(columns={'mhc_embedding_key': 'allele'})
    #     if 'allele' not in benchmark_df_cache.columns:
    #         print("Warning: Missing 'allele' column in benchmark data; duplicate/conflict checks will be skipped.")
    #     # Ensure 'assigned_label' exists for later logic (use NaN placeholder if absent)
    #     if 'assigned_label' not in benchmark_df_cache.columns:
    #         benchmark_df_cache['assigned_label'] = np.nan
    #     benchmark_df_cache.to_parquet(benchmark_df_cache_path, index=False)
    #     print(f"Saved benchmark pairs cache to: {benchmark_df_cache_path}")
    # else:
    #     print("Warning: No benchmark pairs found to cache.")

    # verify if there are any rows that has different labels but same (long_mer, allele) pair
    dup_mask = benchmark_df_cache.duplicated(subset=['allele', 'long_mer'], keep=False)
    dup_rows = benchmark_df_cache[dup_mask]
    if dup_rows.empty:
        print("No duplicate (allele,long_mer) pairs in benchmark dataset.")
    else:
        # Check for conflicting labels among duplicated (allele,long_mer) pairs
        label_counts = dup_rows.groupby(['allele', 'long_mer'])['assigned_label'].nunique()
        conflicting_pairs = label_counts[label_counts > 1]

        if not conflicting_pairs.empty:
            print(f"Found {len(conflicting_pairs)} conflicting (allele,long_mer) pairs with inconsistent labels:")
            for (allele, long_mer) in conflicting_pairs.index:
                conflicting_rows = benchmark_df_cache[
                    (benchmark_df_cache['allele'] == allele) & (benchmark_df_cache['long_mer'] == long_mer)
                    ]
                cols_to_show = ['allele', 'long_mer', 'assigned_label']
                if 'source' in conflicting_rows.columns:
                    cols_to_show.append('source')
                print(f"Allele: {allele}, Peptide: {long_mer}")
                print(conflicting_rows[cols_to_show])
                print("-" * 50)

            # Drop all conflicting (allele,long_mer) pairs
            benchmark_df_cache = benchmark_df_cache[
                ~benchmark_df_cache.set_index(['allele', 'long_mer']).index.isin(conflicting_pairs.index)
            ]
            print(f"After removing conflicting pairs, benchmark dataset size: {len(benchmark_df_cache)}")
        else:
            print("Duplicate (allele,long_mer) pairs detected, but labels are consistent (no conflicts).")

    # if benchmark_df_cache[dup_check].groupby(['allele_x', 'long_mer_x'])['assigned_label_x'].nunique().max() > 1:
    #     conflicting_pairs = benchmark_df_cache[dup_check].groupby(['allele', 'long_mer'])['assigned_label'].nunique()
    #     conflicting_pairs = conflicting_pairs[conflicting_pairs > 1]
    #     print(f"Found {len(conflicting_pairs)} conflicting pairs:")
    #     for (allele, long_mer), _ in conflicting_pairs.items():
    #         conflicting_rows = benchmark_df_cache[
    #             (benchmark_df_cache['allele'] == allele) & (benchmark_df_cache['long_mer'] == long_mer)]
    #         print(f"Allele: {allele}, Peptide: {long_mer}")
    #         # include source in the printed output if available
    #         cols_to_show = ['allele', 'long_mer', 'assigned_label']
    #         if 'source' in conflicting_rows.columns:
    #             cols_to_show.append('source')
    #         print(conflicting_rows[cols_to_show])
    #         print("-" * 50)
    #     # raise ValueError("Benchmark dataset has conflicting labels for the same (allele, long_mer) pairs.") # TODO enable this later
    #     # drop conflicting pairs
    #     benchmark_df_cache = benchmark_df_cache[
    #         ~benchmark_df_cache.set_index(['allele', 'long_mer']).index.isin(conflicting_pairs.index)]
    #     print(f"After removing conflicting pairs, benchmark dataset size: {len(benchmark_df_cache)}")
    # else:
    #     raise "Warning: Benchmark dataset is empty. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

    dup_pmbind_check = fold.duplicated(subset=['allele', 'long_mer'], keep=False)
    if fold[dup_pmbind_check].groupby(['allele', 'long_mer'])['assigned_label'].nunique().max() > 1:
        raise ValueError("PMBind balanced dataset has conflicting labels for the same (allele, long_mer) pairs.")

    # Ensure same dtypes for join columns (robust handling if benchmark_df_cache lacks expected cols)
    if 'allele' not in benchmark_df_cache.columns:
        if 'mhc_embedding_key' in benchmark_df_cache.columns:
            benchmark_df_cache = benchmark_df_cache.rename(columns={'mhc_embedding_key': 'allele'})
        else:
            benchmark_df_cache['allele'] = pd.Series(dtype=fold['allele'].dtype)

    if 'long_mer' not in benchmark_df_cache.columns:
        benchmark_df_cache['long_mer'] = pd.Series(dtype=fold['long_mer'].dtype)

    benchmark_df_cache['reduced_allele'] = benchmark_df_cache['allele'].apply(clean_key)
    fold['reduced_allele'] = fold['allele'].apply(clean_key)
    benchmark_df_cache['long_mer'] = benchmark_df_cache['long_mer'].astype(fold['long_mer'].dtype)

    # Anti-join to remove overlapping pairs
    # df = df.merge(bench_pairs_df.drop_duplicates(), how="left", on=["mhc_embedding_key", "long_mer"], indicator=True)
    # removed = (df["_merge"] == "both").sum()
    # df = df.loc[df["_merge"] == "left_only", df.columns.difference(["_merge"])].reset_index(drop=True)
    pmbind_filtered = (
        fold.merge(
            benchmark_df_cache.drop_duplicates(),
            on=['reduced_allele', 'long_mer'],
            how='left',
            indicator=True
        )
        .query("_merge == 'left_only'")
        .drop(columns='_merge')
        .reset_index(drop=True)
    )
    # verify no overlap remains
    overlap_check = pd.merge(pmbind_filtered, benchmark_df_cache, on=['reduced_allele', 'long_mer'], how='inner')
    if not overlap_check.empty:
        raise ValueError("Overlap with benchmark pairs still exists after filtering.")

    print(f"Removed {len(fold) - len(pmbind_filtered)} overlapping pairs with benchmark datasets.")

    # clean column names of filtered pmbind - current : Calculating overlap analysis for pmbind_filtered... columns: ['long_mer', 'allele_x', 'assigned_label_x', 'source_x', 'mhc_class_x', 'mhc_embedding_key_x', 'reduced_allele', 'Date', 'IEDB reference', 'allele_y', 'Peptide length', 'Measurement type', 'Measurement value', 'assigned_label_y', 'NetMHCpan 2.8', 'NetMHCpan 3.0', 'NetMHCpan 4.0', 'SMM', 'ANN 3.4', 'ANN 4.0', 'ARB', 'SMMPMBEC', 'IEDB Consensus', 'NetMHCcons', 'PickPocket', 'mhc_embedding_key_y', 'source_y', 'mhc_class_y']
    pmbind_filtered = pmbind_filtered.rename(columns={
        'allele_x': 'allele',
        'assigned_label_x': 'assigned_label',
        'source_x': 'source',
        'mhc_class_x': 'mhc_class',
        'mhc_embedding_key_x': 'mhc_embedding_key'
    })
    pmbind_filtered = pmbind_filtered[
        ['long_mer', 'allele', 'assigned_label', 'source', 'mhc_class', 'mhc_embedding_key']]
    print(f"Filtered PMBind dataset size: {len(pmbind_filtered)}")

    return pmbind_filtered, benchmark_df_cache

# get stats of each dataset: size, positive class ratio, negative class ratio, number of unique alleles, number of unique peptides,
# per unique allele: (average peptide length, max peptide length, min peptide length, peptide entropy (Shannon entropy), allele entropy (Shannon entropy))
# save to a csv file
def calculate_basic_dataset_stats(df, dataset_name):
    """Calculate basic statistics for a dataset"""
    stats = {
        'dataset': dataset_name,
        'size': len(df),
        'positive_ratio': (df['assigned_label'] == 1).mean() if 'assigned_label' in df.columns else None,
        'negative_ratio': (df['assigned_label'] == 0).mean() if 'assigned_label' in df.columns else None,
        'n_unique_alleles': df['allele'].nunique() if 'allele' in df.columns else None,
        'n_unique_peptides': df['long_mer'].nunique(),
    }
    return stats

def statistics(fold, benchmark_df, stats_dir):

    print("\n=== BASIC DATASET STATISTICS ===")
    datasets_for_basic_stats = [
        (fold, 'pmbind_filtered'),
        (benchmark_df, 'benchmark_combined'),
    ]

    basic_stats = []
    for data, name in datasets_for_basic_stats:
        print(f"Calculating basic stats for {name}...")
        stats = calculate_basic_dataset_stats(data, name)
        basic_stats.append(stats)

    basic_stats_df = pd.DataFrame(basic_stats)
    # ensure output directory exists
    out_dir = pathlib.Path(os.path.join(stats_dir, name))
    out_dir.mkdir(parents=True, exist_ok=True)
    basic_stats_df.to_csv(out_dir / 'basic_dataset_statistics.csv', index=False)
    print(basic_stats_df.to_string())
    print(basic_stats_df.to_string())

def dataset_analysis(fold, benchmark_df_cache, stats_dir):
    # 1. OVERLAP ANALYSIS - How many pairs from each dataset are in benchmark
    print("\n=== OVERLAP ANALYSIS WITH BENCHMARK ===")
    # Define datasets to check for overlap with benchmark
    datasets_to_check = [
        (fold, 'pmbind_filtered'),
    ]
    overlap_results = []
    for data, name in datasets_to_check:
        # Check overlap using mhc_embedding_key and long_mer
        print(f"Calculating overlap analysis for {name}... columns: {data.columns.tolist()}")
        data = data.copy()
        data["reduced_allele"] = data["allele"].apply(clean_key)
        data["long_mer"] = data["long_mer"].astype(str)
        benchmark_df_cache["long_mer"] = benchmark_df_cache["long_mer"].astype(str)
        benchmark_df_cache["reduced_allele"] = benchmark_df_cache["allele"].apply(clean_key)
        overlap = pd.merge(data, benchmark_df_cache, on=['reduced_allele', 'long_mer'], how='inner')

        overlap_result = {
            'dataset': name,
            'total_pairs': len(data),
            'overlapping_pairs': len(overlap),
            'overlap_percentage': (len(overlap) / len(data)) * 100 if len(data) > 0 else 0
        }
        overlap_results.append(overlap_result)

        print(
            f"{name}: {len(overlap):,} / {len(data):,} pairs overlap with benchmark ({overlap_result['overlap_percentage']:.2f}%)")

    overlap_df = pd.DataFrame(overlap_results)
    out_dir = pathlib.Path(os.path.join(stats_dir, name))
    out_dir.mkdir(parents=True, exist_ok=True)
    overlap_df.to_csv(out_dir / 'benchmark_overlap_analysis.csv', index=False)

# ALLELE-LEVEL ANALYSIS - Positive/Negative counts per allele per dataset
def allele_level_analysis(fold, benchmark_df_cache, stats_dir):
    datasets_to_check = [
        (fold, 'pmbind_filtered'),
    ]
    print("\n=== ALLELE-LEVEL POSITIVE/NEGATIVE ANALYSIS ===")

    # Get all unique alleles across all datasets (only from datasets with 'allele' column)
    all_alleles = set()
    datasets_with_labels = []

    for data, name in datasets_to_check:
        if 'assigned_label' in data.columns and 'allele' in data.columns:
            datasets_with_labels.append((data, name))
            # Apply clean_key to standardize allele names
            cleaned_alleles = data['allele'].apply(clean_key).unique()
            all_alleles.update(cleaned_alleles)

    # Also add benchmark dataset if it has allele column
    if 'assigned_label' in benchmark_df_cache.columns and 'allele' in benchmark_df_cache.columns:
        datasets_with_labels.append((benchmark_df_cache, 'benchmark_combined'))
        cleaned_alleles = benchmark_df_cache['allele'].apply(clean_key).unique()
        all_alleles.update(cleaned_alleles)

    all_alleles = sorted(list(all_alleles))
    print(f"Found {len(all_alleles)} unique cleaned alleles across all datasets")

    if len(datasets_with_labels) == 0:
        print("Warning: No datasets found with both 'allele' and 'assigned_label' columns!")
    else:
        # Create comprehensive allele analysis table
        allele_analysis = pd.DataFrame({'reduced_allele': all_alleles})

        for data, name in datasets_with_labels:
            print(f"Processing allele statistics for {name}...")

            # Create a copy with cleaned allele names
            data_copy = data.copy()
            data_copy['reduced_allele'] = data_copy['allele'].apply(clean_key)

            # Count positives and negatives per cleaned allele
            allele_counts = data_copy.groupby(['reduced_allele', 'assigned_label']).size().unstack(fill_value=0)

            # Ensure we have both 0 and 1 columns
            if 0 not in allele_counts.columns:
                allele_counts[0] = 0
            if 1 not in allele_counts.columns:
                allele_counts[1] = 0

            allele_counts = allele_counts.reset_index()
            allele_counts.columns = ['reduced_allele', f'{name}_neg', f'{name}_pos']

            # Merge with main allele analysis table
            allele_analysis = allele_analysis.merge(
                allele_counts,
                on='reduced_allele',
                how='left'
            )

            # Fill NaN with 0 (alleles not present in this dataset)
            allele_analysis[f'{name}_neg'] = allele_analysis[f'{name}_neg'].fillna(0).astype(int)
            allele_analysis[f'{name}_pos'] = allele_analysis[f'{name}_pos'].fillna(0).astype(int)

        # Add total columns for each dataset
        for data, name in datasets_with_labels:
            allele_analysis[f'{name}_total'] = allele_analysis[f'{name}_neg'] + allele_analysis[f'{name}_pos']

        # Save the comprehensive allele analysis
        out_dir = pathlib.Path(os.path.join(stats_dir, name))
        out_dir.mkdir(parents=True, exist_ok=True)
        allele_analysis.to_csv(out_dir / 'allele_positive_negative_analysis.csv', index=False)

        print(f"\nAllele analysis saved to 'stats/allele_positive_negative_analysis.csv'")
        print(f"Table contains {len(allele_analysis)} cleaned alleles and {len(datasets_with_labels)} datasets")

        # Display summary of allele analysis
        print("\n=== ALLELE ANALYSIS SUMMARY ===")
        total_columns = [col for col in allele_analysis.columns if col.endswith('_total')]
        for col in total_columns:
            dataset_name = col.replace('_total', '')
            total_alleles_with_data = (allele_analysis[col] > 0).sum()
            print(f"{dataset_name}: {total_alleles_with_data} alleles have data")


# CREATE SUMMARY REPORT
def create_report(basic_stats_df, overlap_df, allele_analysis, datasets_with_labels, stats_dir):
    print("\nCreating comprehensive report...")
    out_dir = pathlib.Path(os.path.join(stats_dir, 'report'))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'dataset_analysis_report.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE DATASET ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. BASIC DATASET STATISTICS\n")
        f.write("-" * 40 + "\n")
        for idx, row in basic_stats_df.iterrows():
            f.write(f"\n{row['dataset'].upper()}:\n")
            f.write(f"  Size: {row['size']:,}\n")
            if pd.notna(row['positive_ratio']):
                f.write(f"  Positive ratio: {row['positive_ratio']:.2%}\n")
                f.write(f"  Negative ratio: {row['negative_ratio']:.2%}\n")
            f.write(f"  Unique alleles: {int(row['n_unique_alleles']):,}\n")
            f.write(f"  Unique peptides: {int(row['n_unique_peptides']):,}\n")

        f.write("\n\n2. BENCHMARK OVERLAP ANALYSIS\n")
        f.write("-" * 40 + "\n")
        for idx, row in overlap_df.iterrows():
            f.write(f"{row['dataset']}: {row['overlapping_pairs']:,} / {row['total_pairs']:,} pairs ")
            f.write(f"({row['overlap_percentage']:.2f}% overlap with benchmark)\n")

        all_alleles = allele_analysis[['dataset', 'overlapping_pairs', 'total_pairs']]

        f.write(f"\n\n3. ALLELE ANALYSIS SUMMARY\n")
        f.write("-" * 40 + "\n")
        if 'allele_analysis' in locals() and len(datasets_with_labels) > 0:
            f.write(f"Total unique cleaned alleles across all datasets: {len(all_alleles)}\n")
            total_columns = [col for col in allele_analysis.columns if col.endswith('_total')]
            for col in total_columns:
                dataset_name = col.replace('_total', '')
                total_alleles_with_data = (allele_analysis[col] > 0).sum()
                f.write(f"{dataset_name}: {total_alleles_with_data} alleles have data\n")
        else:
            f.write("No datasets found with both 'allele' and 'assigned_label' columns for allele analysis.\n")

    print(f"\nFiles generated:")
    print("  - basic_dataset_statistics.csv")
    print("  - benchmark_overlap_analysis.csv")
    print("  - allele_positive_negative_analysis.csv")
    print("  - dataset_analysis_report.txt")

    print(f"\nAnalysis complete!")

# Replace index-based drops
def _anti_join(base, remove_df):
    keys = ['allele', 'long_mer']
    rm = remove_df[keys].drop_duplicates()
    return (base.merge(rm, on=keys, how='left', indicator=True)
            .query('_merge=="left_only"')
            .drop(columns='_merge'))

def take_test_set_samples(fold):
    # test: all samples of 4 unique reduced_allele: SLA-30401, BOLA-300101, H-2-KK, HLA-A3004, HLA-B2701, HLA-C1701
    if MHC_CLASS == 2:
        test_alleles = ['H-2-IAD-A_H-2-IAD-B', 'HLA-DQA10501_HLA-DQB10402', 'HLA-DRA_HLA-DRB30301', 'HLA-DPA10301_HLA-DPB10402']
    else:
        test_alleles = ['SLA-30401', 'BOLA-300101', 'H-2-KK', 'HLA-A3004', 'HLA-B2701', 'HLA-C1701', 'MAMU-B06601']
    test_set = fold[fold['allele'].apply(clean_key).isin(test_alleles)].reset_index(
        drop=True)
    print(f"Test set created with {len(test_set)} samples from alleles: {test_alleles}")
    pmbind_filtered_remaining = _anti_join(fold, test_set)
    overlap_check_test = pd.merge(pmbind_filtered_remaining, test_set, on=['allele', 'long_mer'], how='inner')
    if not overlap_check_test.empty:
        raise ValueError("Overlap with test pairs still exists after filtering.")
    return test_set, pmbind_filtered_remaining

def leave_one_allele_out_split(pmbind_filtered, seed):
    # create test set with all samples of 4 unique reduced_allele: SLA-30401, BOLA-300101, H-2-KK, HLA-A3004, HLA-B2701, HLA-C1701
    # create validation set with some samples of these alleles: HLA-A0201, HLA-A0301, HLA-B2705, HLA-B0702, HLA-C0401, HLA-C0602
    # make sure no overlap between train/val/test and pmbind_filtered
    # use clean_key function to clean the allele names before matching
    # create a fixed validation set from these alleles (take some samples not entirely): HLA-A0201, HLA-A0301, HLA-B2705, HLA-B0702, HLA-C0401, HLA-C0602
    val_alleles = ['HLA-A0201', 'HLA-A0301', 'HLA-B2705', 'HLA-B0702', 'HLA-C0401', 'HLA-C0602', 'MAMU-B05201',
                   'PATR-B1301', 'SLA-20401', 'BOLA-601301']
    all_val_candidates = pmbind_filtered[
        pmbind_filtered['allele'].apply(clean_key).isin(val_alleles)].reset_index(drop=True)
    val_set = all_val_candidates.groupby('allele').apply(lambda x: x.sample(min(len(x), 100), random_state=seed)).reset_index(
        drop=True)

    pmbind_filtered_remaining = _anti_join(pmbind_filtered, val_set)

    # verify no overlap remains
    overlap_check_val = pd.merge(pmbind_filtered_remaining, val_set, on=['allele', 'long_mer'], how='inner')
    if not overlap_check_val.empty:
        raise ValueError("Overlap with validation pairs still exists after filtering.")


    # Normalize labels
    pmbind_filtered_remaining['assigned_label'] = (
        pmbind_filtered_remaining['assigned_label']
        .map({1: 1, '1': 1, 0: 0, '0': 0, -1: 0})
        .astype(int)
    )

    # leave one random allele left out, add the left out allele to validation set
    Left_out_id = np.random.choice(pmbind_filtered_remaining['allele'].unique(), size=1, replace=False)[0]
    left_out_samples = pmbind_filtered_remaining[pmbind_filtered_remaining['allele'] == Left_out_id]
    val_set = pd.concat([val_set, left_out_samples], ignore_index=True)
    pmbind_filtered_remaining = pmbind_filtered_remaining[pmbind_filtered_remaining['allele'] != Left_out_id]
    print(f"Left out allele {Left_out_id} with {len(left_out_samples)} samples added to validation set.")

    return pmbind_filtered_remaining, val_set

# # create different train files with different random seeds and sizes.
# # a balanced train set of size 10000, 20000, 50000, 100000
# seeds = [42, 54, 1, 999, 123]
# train_sizes = [10000, 50000, 100000, len(pmbind_filtered_remaining) // len(seeds)]
# new_df_path = "new_df2"
#
# for size in train_sizes:
#     available_data = pmbind_filtered_remaining.copy()
#
#     # Separate positive and negative samples and shuffle them
#     pos_samples = available_data[available_data['assigned_label'] == 1].sample(frac=1, random_state=42)
#     neg_samples = available_data[available_data['assigned_label'] == 0].sample(frac=1, random_state=42)
#
#     # determine the minimum number of per_class_sample available across alleles
#     alleles = pos_samples['allele'].unique()
#     min_per_class_sample = available_data['allele'].value_counts().min()
#
#     possible_splits = min(len(seeds), min_per_class_sample)
#
#     for i in range(possible_splits):
#         seed = seeds[i]
#         print("current min_per_class_sample:", min_per_class_sample)
#         if min_per_class_sample == 0:
#             print("Stopping: At least one allele has zero samples for a class.")
#             break
#         elif min_per_class_sample == 1:
#             print(
#                 "found only one positive sample for at least one allele, saving the whole remaining data as one split")
#             # TODO save the whole remaining data as one split
#             train_set = available_data
#             filename = f"{new_df_path}/train_size{size}_seed{seed}_remaining.parquet"
#             train_set.to_parquet(filename, index=False)
#             break
#         else:
#             min_per_class_sample = available_data['allele'].value_counts().min()
#
#         # Build allele-level balanced split ensuring each included allele has both classes
#         eligible = available_data.groupby('allele').filter(lambda g: set(g['assigned_label'].unique()) == {0, 1})
#         alleles = eligible['allele'].unique()
#
#         # TODO leave one allele out for validation/test sets
#         print(f"Generating training set of size {size} with seed {seed}...")
#         left_out_allele = np.random.choice(alleles, size=1, replace=False)[0]
#         left_out_samples = eligible[eligible['allele'] == left_out_allele]
#         # save left out samples to a separate file for inspection
#         left_out_filename = f"{new_df_path}/left_out_allele_{left_out_allele}_size{size}_seed{seed}.parquet"
#
#         eligible = eligible[eligible['allele'] != left_out_allele]
#
#         if len(alleles) == 0:
#             print(f"Should not happen !!!! - No alleles with both classes available for size {size}. Stopping.!!!!!")
#             break
#         if len(eligible) < size:
#             # saving the whole remaining data as one split
#             print(
#                 f"Not enough eligible samples ({len(eligible)}) to create a train set of size {size}. Saving remaining {len(available_data)} samples as one split.")
#             # TODO save the whole remaining data as one split
#             train_set = available_data
#             filename = f"{new_df_path}/train_size{size}_seed{seed}_remaining.parquet"
#             train_set.to_parquet(filename, index=False)
#             break
#
#         target_per_class_size = size // 2
#
#         eligible_pos = eligible[eligible['assigned_label'] == 1]
#         eligible_neg = eligible[eligible['assigned_label'] == 0]
#
#         # TODO calculate the portion of positive and negative samples to take from each allele
#         # Calculate the proportion of samples each allele contributes to the total positive and negative counts
#         pos_proportions = eligible_pos['allele'].value_counts(normalize=True)
#         neg_proportions = eligible_neg['allele'].value_counts(normalize=True)
#
#         # Determine the number of positive and negative samples to draw from each allele
#         pos_samples_per_allele = (pos_proportions * target_per_class_size).round().astype(int)
#         neg_samples_per_allele = (neg_proportions * target_per_class_size).round().astype(int)
#
#         print(pos_samples_per_allele)
#         print(neg_samples_per_allele)
#
#         # take the maximum of 1 and the calculated number of samples to ensure at least one sample is taken from each allele
#         pos_samples_per_allele = pos_samples_per_allele.apply(lambda x: max(1, x))
#         neg_samples_per_allele = neg_samples_per_allele.apply(lambda x: max(1, x))
#
#         # Adjust for rounding errors to ensure the total count is exactly the target size
#         pos_diff = target_per_class_size - pos_samples_per_allele.sum()
#         if pos_diff != 0 and not pos_samples_per_allele.empty:
#             pos_samples_per_allele[pos_proportions.idxmax()] += pos_diff
#
#         neg_diff = target_per_class_size - neg_samples_per_allele.sum()
#         if neg_diff != 0 and not neg_samples_per_allele.empty:
#             neg_samples_per_allele[neg_proportions.idxmax()] += neg_diff
#         # TODO sample per allele -
#         #  from each allele randomly select samples
#         #  (make sure to sample both classes -
#         #  and if the number of min_per_class_sample <
#         #  possible_splits, take only one sample from the class with very few samples)
#
#         train_parts = []
#         for allele in eligible['allele'].unique():
#             pos_to_sample = pos_samples_per_allele.get(allele, 0)
#             neg_to_sample = neg_samples_per_allele.get(allele, 0)
#
#             if pos_to_sample > 0:
#                 allele_pos_data = eligible_pos[eligible_pos['allele'] == allele]
#                 train_parts.append(allele_pos_data.sample(
#                     n=pos_to_sample,
#                     random_state=seed,
#                     replace=len(allele_pos_data) < pos_to_sample
#                 ))
#
#             if neg_to_sample > 0:
#                 allele_neg_data = eligible_neg[eligible_neg['allele'] == allele]
#                 train_parts.append(allele_neg_data.sample(
#                     n=neg_to_sample,
#                     random_state=seed,
#                     replace=len(allele_neg_data) < neg_to_sample
#                 ))
#
#         if train_parts:
#             train_set = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
#             filename = f"{new_df_path}/train_size{size}_seed{seed}.parquet"
#             train_set.to_parquet(filename, index=False)
#             print(f"Generated training set: {filename} with size {len(train_set)}")
#
#         # TODO take out the selected samples from available_data
#         # remove selected samples from available_data
#         available_data = _anti_join(available_data, train_set)
#         print(f"Remaining samples for next split: {len(available_data)}")
#
#         # TODO continue until we reach the desired size or run out of eligible alleles
#         if len(available_data) < size:
#             print(
#                 f"Not enough remaining samples ({len(available_data)}) to create another train set of size {size}. Stopping.")
#             train_set = available_data
#             filename = f"{new_df_path}/train_size{size}_seed{seed}_remaining.parquet"
#             train_set.to_parquet(filename, index=False)
#             print(f"Generated training set: {filename} with size {len(train_set)}")
#             break
#
# # save val and test sets
# val_set.to_parquet(f"{new_df_path}/val_set.parquet", index=False)
# print(f"Generated validation set: val_set.parquet with size {len(val_set)}")
# test_set.to_parquet(f"{new_df_path}/test_set.parquet", index=False)
# print(f"Generated test set: test_set.parquet with size {len(test_set)}")


# create stats for new datasets
# 1. per allele: number of samples, positive ratio, min, max, avg peptide length, peptide entropy
def shannon_entropy(seq_list):
    """Calculate Shannon entropy for a list of sequences."""
    if len(seq_list) == 0:
        return 0.0
    # Count frequency of each unique sequence
    freq = Counter(seq_list)
    probabilities = [count / len(seq_list) for count in freq.values()]
    return entropy(probabilities, base=2)  # Using base 2 for entropy in bits


def allele_level_stats(df, dataset_name):
    """Calculate allele-level statistics for a dataset."""
    stats = []
    for allele, group in df.groupby('allele'):
        peptide_lengths = group['long_mer'].str.len()
        stats.append({
            'dataset': dataset_name,
            'allele': allele,
            'num_samples': len(group),
            'positive_ratio': (group['assigned_label'] == 1).mean() if 'assigned_label' in group.columns else None,
            'min_peptide_length': peptide_lengths.min(),
            'max_peptide_length': peptide_lengths.max(),
            'avg_peptide_length': peptide_lengths.mean(),
            'peptide_entropy': shannon_entropy(group['long_mer'].tolist())
        })
    return pd.DataFrame(stats)


# # Get all parquet files in new_df/ folder
# import os
# import glob
#
# new_df_files = glob.glob(f"{new_df_path}/*.parquet")
# print(f"Found {len(new_df_files)} parquet files in {new_df_path}/")
#
# allele_stats_list = []
#
# # Process each file in new_df/
# for file_path in new_df_files:
#     filename = os.path.basename(file_path).replace('.parquet', '')
#     print(f"Calculating allele-level stats for {filename}...")
#
#     try:
#         df = pd.read_parquet(file_path)
#         stats_df = allele_level_stats(df, filename)
#         allele_stats_list.append(stats_df)
#     except Exception as e:
#         print(f"Error processing {filename}: {e}")
#
# # Also include existing datasets for comparison
# datasets_for_allele_stats = [
#     (df_train, 'train'),
#     (df_val, 'val'),
#     (df_test, 'test'),
#     (pmbind_filtered, 'pmbind_filtered'),
#     (benchmark_df_cache, 'benchmark_combined')
# ]
#
# for data, name in datasets_for_allele_stats:
#     print(f"Calculating allele-level stats for {name}...")
#     try:
#         stats_df = allele_level_stats(data, name)
#         allele_stats_list.append(stats_df)
#     except Exception as e:
#         print(f"Error processing {name}: {e}")
#
# # Combine all statistics
# if allele_stats_list:
#     allele_stats_df = pd.concat(allele_stats_list, ignore_index=True)
#     allele_stats_df.to_csv(f'{new_df_path}/allele_level_statistics.csv', index=False)
#     print(
#         f"Saved allele statistics for {len(allele_stats_df['dataset'].unique())} datasets to '{new_df_path}/allele_level_statistics.csv'")
#
#     # Print summary for each dataset
#     print("\n=== DATASET SUMMARY ===")
#     for dataset in allele_stats_df['dataset'].unique():
#         subset = allele_stats_df[allele_stats_df['dataset'] == dataset]
#         total_samples = subset['num_samples'].sum()
#         num_alleles = len(subset)
#         avg_pos_ratio = subset['positive_ratio'].mean()
#         print(f"{dataset}: {total_samples:,} samples across {num_alleles} alleles (avg pos ratio: {avg_pos_ratio:.3f})")
# else:
#     print("No data processed for allele statistics.")
#
# benchmark_df_cache.to_parquet(f'{new_df_path}/benchmark_combined.parquet', index=False)
#
# print("\nData generation and analysis complete!")


def main():
    df_path = f"../../data/binding_affinity_data/concatenated_class{MHC_CLASS}_all.parquet"
    seeds = [999, 54, 1, 42, 123, 7, 21, 84, 105, 111]
    for seed in seeds:
        print("Processing seed:", seed)
        fold, rare_samples = median_down_sampling(df_path, seed)
        fold, benchmark_df_cache = remove_benchmark_samples(fold, benchmarks_dir=f"../../data/mhc{MHC_CLASS}/benchmarks")
        fold_test, fold_filtered = take_test_set_samples(fold)
        fold_train, fold_val, = leave_one_allele_out_split(fold_filtered, seed)
        statistics(fold_train, benchmark_df_cache, stats_dir=f"../../data/reforge/mhc{MHC_CLASS}/stats_{seed}")
        dataset_analysis(fold_train, benchmark_df_cache, stats_dir=f"../../data/reforge/mhc{MHC_CLASS}/stats_{seed}")
        allele_level_analysis(fold_train, benchmark_df_cache, stats_dir=f"../../data/reforge/mhc{MHC_CLASS}/stats_{seed}")

        out_dir = pathlib.Path(f"../../data/reforge/mhc{MHC_CLASS}/seed_splits/")
        out_dir.mkdir(parents=True, exist_ok=True)

        fold_train.to_parquet(out_dir / f"train_seed_{seed}.parquet", index=False)
        fold_val.to_parquet(out_dir / f"val_seed_{seed}.parquet", index=False)
        fold_test.to_parquet(out_dir / f"test_seed_{seed}.parquet", index=False)
        rare_samples.to_parquet(out_dir / f"test_set2_rare_samples_seed_{seed}.parquet", index=False)
        print(f"Saved datasets for seed {seed}.")

if __name__ == "__main__":
    main()