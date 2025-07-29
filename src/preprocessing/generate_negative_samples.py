#!/usr/bin/env python
# make_extra_negatives.py
#
# Build extra negatives for alleles whose negative:positive ratio is < 5,
# using positive samples from the most distant (furthest) sequence clusters.

import json
import os
import tempfile
from collections import defaultdict
from io import StringIO
from typing import Dict, List

import numpy as np
import pandas as pd
import Levenshtein                           # pip install python-Levenshtein
from scipy.spatial.distance import pdist, squareform

from src.utils import cluster_aa_sequences   # import the earlier function

# ----------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------
ANALYSIS_CSV   = "../../data/binding_affinity_data/allele_stats_class1_with_seq.csv"
BINDING_CSV    = "../../data/binding_affinity_data/concatenated_class1.parquet"
UPDATED_BIND   = "../../data/binding_affinity_data/binding_dataset_with_synthetic_negatives_class1.csv"
SAMPLING_DICT  = "../../data/binding_affinity_data/sampling_dict_class1.json"

TARGET_RATIO   = 5          # want negatives ≥ TARGET_RATIO × positives
LINKAGE_K      = 8          # number of clusters for the sequence tree
MAX_CLUSTERS_TO_SEARCH = None   # None = search *all* clusters


# ----------------------------------------------------------------------
# Helper: compute a full pair-wise (Levenshtein) distance matrix
# ----------------------------------------------------------------------
def make_distance_matrix(seqs: List[str]) -> np.ndarray:
    """Return an NxN matrix of normalised Levenshtein distances."""
    def d(a, b):
        return Levenshtein.distance(a, b) / max(len(a), len(b))
    vec = pdist(np.array(seqs, dtype=object)[:, None],  # (N,1) → 2-D shape
                lambda u, v: d(u[0], v[0]))
    return squareform(vec)   # (N, N)


# ----------------------------------------------------------------------
# 1) Load analysis dataset and add cluster labels
# ----------------------------------------------------------------------
print("Loading analysis file and clustering sequences …")
df_analysis, _ = cluster_aa_sequences(
    ANALYSIS_CSV,
    id_col="allele",
    seq_col="sequence",
    k=LINKAGE_K,
    linkage_method="average",
    gap_mode="ignore_gaps",
    plot=False,
)

# Build some quick look-ups
alleles           = df_analysis["allele"].tolist()
seqs              = df_analysis["sequence"].tolist()
clusters          = df_analysis["cluster"].tolist()
allele2cluster    = dict(zip(alleles, clusters))
cluster2alleles   = defaultdict(list)
for a, c in zip(alleles, clusters):
    cluster2alleles[c].append(a)

# Full pair-wise distance matrix (used later for “furthest cluster”)
dist_mat = make_distance_matrix(seqs)
allele_index = {a: i for i, a in enumerate(alleles)}   # map allele → row in dist_mat


# ----------------------------------------------------------------------
# 2) Decide which alleles need extra negatives
# ----------------------------------------------------------------------
need_negatives = {}
for _, row in df_analysis.iterrows():
    allele = row["allele"]
    pos    = row["positives"]
    neg    = row["negatives"]
    target_neg = TARGET_RATIO * pos
    if neg < target_neg:
        need_negatives[allele] = int(target_neg - neg)

print(f"Alleles needing extra negatives: {len(need_negatives)}")
if not need_negatives:
    print("Nothing to do — all alleles already have ≥ 5× negatives.")
    exit(0)

# ----------------------------------------------------------------------
# 3) Build a sampling dictionary: allele → list of *distant* alleles
# ----------------------------------------------------------------------
print("Computing furthest clusters / alleles …")
sampling_dict: Dict[str, List[str]] = {}

# Pre-compute average distance *between* clusters
cluster_ids = sorted(cluster2alleles.keys())
n_clusters  = len(cluster_ids)
cluster_dist = np.zeros((n_clusters, n_clusters))

for i, ci in enumerate(cluster_ids):
    for j, cj in enumerate(cluster_ids):
        if j <= i:
            continue
        # All pair-wise distances between members of ci and cj
        idx_i = [allele_index[a] for a in cluster2alleles[ci]]
        idx_j = [allele_index[a] for a in cluster2alleles[cj]]
        submat = dist_mat[np.ix_(idx_i, idx_j)]
        cluster_dist[i, j] = cluster_dist[j, i] = submat.mean()

# For each allele that needs negatives, find clusters with largest distance
for allele, required in need_negatives.items():
    c0   = allele2cluster[allele]
    i0   = cluster_ids.index(c0)
    # Sort clusters by distance descending
    order = np.argsort(cluster_dist[i0])[::-1]
    distant_clusters = [cluster_ids[k] for k in order
                        if cluster_ids[k] != c0]      # exclude own cluster
    if MAX_CLUSTERS_TO_SEARCH:
        distant_clusters = distant_clusters[:MAX_CLUSTERS_TO_SEARCH]

    # Collect alleles that inhabit those clusters
    sampling_alleles = []
    for c in distant_clusters:
        sampling_alleles.extend(cluster2alleles[c])
    sampling_dict[allele] = sampling_alleles

# Save a JSON copy just for bookkeeping/debugging
with open(SAMPLING_DICT, "w") as fp:
    json.dump(sampling_dict, fp, indent=2)
print(f"Sampling dictionary written to {SAMPLING_DICT}")


# ----------------------------------------------------------------------
# 4) Load binding dataset and build extra negatives
# ----------------------------------------------------------------------
print("Loading binding dataset …")
df_bind = pd.read_parquet(BINDING_CSV)

# We treat rows where assigned_label == 1 (positive) as potential
# “negatives” for SOME OTHER allele.
POSITIVE_MASK = df_bind["assigned_label"] == 1

extra_rows = []
for allele, needed in need_negatives.items():
    donors   = sampling_dict[allele]
    pool     = df_bind[POSITIVE_MASK & df_bind["allele"].isin(donors)]
    if pool.empty:
        print(f"WARNING: no donor rows found for {allele}")
        continue
    take = pool.sample(n=min(needed, len(pool)), replace=False).copy()
    # Re-label:
    take["allele"]          = allele
    take["assigned_label"]  = 0          # turn into negative
    extra_rows.append(take)
    print(f"{allele}: added {len(take)} extra negatives")

if not extra_rows:
    print("No extra negatives were added (check warnings above).")
    exit(0)

df_extra = pd.concat(extra_rows, ignore_index=True)

# ----------------------------------------------------------------------
# 5) Concatenate and save the updated binding dataset
# ----------------------------------------------------------------------
df_updated = pd.concat([df_bind, df_extra], ignore_index=True)
df_updated.to_csv(UPDATED_BIND, index=False)
print(f"Updated binding file written to {UPDATED_BIND}")

# TODO create a phylogeny of all alleles in df1 and df2
# TODO generate negatives for alleles without negative sample pairs by taking samples from furthest trees

# We have a dataset of all positive samples.
# we want to generate negative samples for the alleles that have lower than 5 times the positive values.
#
# data frames are csv files with allele and sequence columns.
#
# write the complete python code

# TODO create a phylogeny of all alleles in df1 and df2 based on their sequences
# TODO generate negatives for alleles without negative sample pairs by taking samples from furthest trees

