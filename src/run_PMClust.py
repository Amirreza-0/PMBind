#!/usr/bin/env python
"""
Minimal runtime for the minimal BiCross reconstruction model
------------------------------------------------------------
• No folds – the whole parquet file is used.
• Streams data: each parquet batch is converted on-the-fly.
• Trains purely for reconstruction (peptide + MHC).
• After training, obtains encoder outputs and clusters them.

Author: 2024  (minimal example)
"""

from __future__ import annotations
import os, argparse, json, pathlib, gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans  # using K-means again
from sklearn.manifold import TSNE           # quick 2-D projection
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import seaborn as sns

# ----------------------------------------------------------------------
# YOUR OWN  HELPERS  ----------------------------------------------------
# Replace these three util imports with the real ones from your code base
# ----------------------------------------------------------------------
from src.utils import (
    seq_to_onehot,
    MaskedEmbedding,
    PositionalEncoding,
    AttentionLayer,
    MASK_TOKEN, PAD_TOKEN,
    NORM_TOKEN, PAD_VALUE,
    get_embed_key
)
from models import bicross_recon_mini

# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 0. GLOBAL embedding archive (opened once)
# ----------------------------------------------------------------------
EMB_DB = None  # will be initialised in main()

# -----------------------------------------------------------------------------
# tiny helpers
# -----------------------------------------------------------------------------
def get_embedding_from_npz(key: str) -> np.ndarray:
    """
    Very cheap accessor: arrays are *memory-mapped*; only the slice we copy
    into the minibatch will actually be paged into RAM.
    """
    if key not in EMB_DB:
        raise KeyError(f"Embedding key `{key}` not found in .npz")
    return EMB_DB[key]

def random_mask(length: int, mask_fraction: float = 0.3) -> np.ndarray:
    """Return an array of size `length` filled with
       {MASK_TOKEN | NORM_TOKEN | PAD_TOKEN}.
       The caller still has to set PAD positions afterwards.
       we mask 0.3 of the positions by default. a bit more than 0.15 because our sequences are short.
       """
    out = np.full((length,), NORM_TOKEN, dtype=np.float32)
    n_mask = int(mask_fraction * length)
    if n_mask:
        # Ensure we don't try to sample from an empty range if length is 0
        if length > 0:
            idx = np.random.choice(length, n_mask, replace=False)
            out[idx] = MASK_TOKEN
    return out

# -----------------------------------------------------------------------------
# 1. A memory-light streaming Parquet → tf.data pipeline
# -----------------------------------------------------------------------------
def rows_to_tensors(rows: pd.DataFrame,
                    max_pep_len: int,
                    max_mhc_len: int,
                    seq_map: dict[str, str],
                    mask_frac: float = 0.15):
    """
    Vectorise a pandas DataFrame → (inputs, targets) for the reconstruction
    model. Targets are full one-hot tensors.
    """
    n = len(rows)
    inputs = {
        "pep_onehot": np.zeros((n, max_pep_len, 21), np.float32),
        "pep_mask":   np.zeros((n, max_pep_len), np.float32),
        "mhc_latent": np.zeros((n, max_mhc_len, 1152), np.float32),
        "mhc_mask":   np.zeros((n, max_mhc_len), np.float32),
    }
    targets = {
        "pep_reconstruction": np.zeros((n, max_pep_len, 21), np.float32),
        "mhc_reconstruction": np.zeros((n, max_mhc_len, 21), np.float32),
    }

    # Clean the keys of the provided seq_map once for efficiency
    seq_map_cleaned = {k.strip(): v.strip() for k, v in seq_map.items()}

    for i, (_, r) in enumerate(rows.iterrows()):
        # ---------- 1) peptide OHE + mask + target --------------------
        pep_seq = r["long_mer"].upper()
        pep_OHE = seq_to_onehot(pep_seq, max_seq_len=max_pep_len)
        inputs["pep_onehot"][i] = pep_OHE
        targets["pep_reconstruction"][i] = pep_OHE
        pep_mask = random_mask(max_pep_len, mask_frac)
        pep_mask[np.all(pep_OHE == PAD_VALUE, axis=-1)] = PAD_TOKEN
        inputs["pep_mask"][i] = pep_mask

        # ---------- 2) MHC embedding ---------------------------------
        raw_key = r["mhc_embedding_key"]
        # Use the provided utility to get the canonical key
        emb_key_norm = get_embed_key(raw_key, seq_map_cleaned)
        emb = get_embedding_from_npz(emb_key_norm)
        l_mhc = emb.shape[0]
        inputs["mhc_latent"][i, :l_mhc, :] = emb
        mhc_mask = random_mask(max_mhc_len, mask_frac)
        mhc_mask[l_mhc:] = PAD_TOKEN
        inputs["mhc_mask"][i] = mhc_mask

        # ---------- 3) MHC target OHE ---------------------------------
        # Use the normalized key and the cleaned map to get the sequence
        mhc_seq = seq_map_cleaned.get(emb_key_norm, "").upper()
        if not mhc_seq:
            raise KeyError(f"No MHC sequence for key '{emb_key_norm}' (raw: '{raw_key}') in seq_map")
        mhc_OHE = seq_to_onehot(mhc_seq, max_seq_len=max_mhc_len)
        targets["mhc_reconstruction"][i] = mhc_OHE

    return inputs, targets

def build_tf_dataset(parquet_path: str,
                     max_pep_len: int,
                     max_mhc_len: int,
                     seq_map: dict[str, str],
                     batch_size: int = 128):
    """
    Stream the parquet file and create a tf.data.Dataset that yields
    (inputs, targets) tuples ready for model.fit().
    """
    qfile = pq.ParquetFile(parquet_path)
    def gen():
        for batch in qfile.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()
            inputs, tg = rows_to_tensors(df, max_pep_len, max_mhc_len, seq_map=seq_map)
            yield inputs, tg

    input_signature = {
        "pep_onehot": tf.TensorSpec((None, max_pep_len, 21), tf.float32),
        "pep_mask": tf.TensorSpec((None, max_pep_len), tf.float32),
        "mhc_latent": tf.TensorSpec((None, max_mhc_len, 1152), tf.float32),
        "mhc_mask": tf.TensorSpec((None, max_mhc_len), tf.float32),
    }
    target_signature = {
        "pep_reconstruction": tf.TensorSpec((None, max_pep_len, 21), tf.float32),
        "mhc_reconstruction": tf.TensorSpec((None, max_mhc_len, 21), tf.float32),
    }
    ds = tf.data.Dataset.from_generator(gen, output_signature=(input_signature, target_signature))
    return ds.prefetch(tf.data.AUTOTUNE)

# -----------------------------------------------------------------------------
# 2.  MAIN
# -----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Path to parquet file(s) (comma-separated OK)")
    ap.add_argument("--emb_npz", required=True, help="Path to .npz file with MHC embeddings")
    ap.add_argument("--seq_map", required=True, help="CSV with MHC sequence mapping (key → mhc_sequence)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--clusters", type=int, default=20, help="K for KMeans")
    ap.add_argument("--outdir", default="run_minimal")
    ap.add_argument("--valid_parquet", default=None,
                    help="Optional validation parquet for model.fit() and cluster verification")
    ap.add_argument("--visualise", action="store_true",
                    help="Produce a 2-D t-SNE plot of the clustered latent space")
    ap.add_argument("--dendrogram", action="store_true",
                    help="Draw a hierarchical-clustering dendrogram with allele labels")
    ap.add_argument("--draw_latents", action="store_true",
                    help="Draw the latent space embeddings heatmap (requires --visualise)")
    args = ap.parse_args(argv)

    parquet_files = args.parquet.split(",")
    pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)

    global EMB_DB
    print(f"Opening embedding archive {args.emb_npz}")
    EMB_DB = np.load(args.emb_npz, mmap_mode='r')
    print(f"✓ archive contains {len(EMB_DB.files)} embeddings")

    # -----------------------------------------------------------------
    #   META-DATA (max lengths)
    # -----------------------------------------------------------------
    max_pep_len = 0
    for f in parquet_files:
        pf = pq.ParquetFile(f)
        # Only read the necessary column to find max length
        lengths = pf.read(columns=["long_mer"]).column(0).to_pylist()
        current_max = max(len(s) for s in lengths if s is not None)
        max_pep_len = max(max_pep_len, current_max)

    any_key = EMB_DB.files[0]
    max_mhc_len = EMB_DB[any_key].shape[0]
    print(f"max_peptide_len={max_pep_len}   max_mhc_len={max_mhc_len}")

    # -----------------------------------------------------------------
    #   DATASET
    # -----------------------------------------------------------------
    seq_map = pd.read_csv(args.seq_map, index_col="key")["mhc_sequence"].to_dict()
    import functools

    ds_train = functools.reduce(
        lambda ds1, ds2: ds1.concatenate(ds2),
        [build_tf_dataset(f, max_pep_len, max_mhc_len, seq_map, args.batch) for f in parquet_files]
    )

    # --- Collect allele labels in the same order ------------------
    allele_labels = []
    for f in parquet_files:  # iterated in the same order
        allele_labels.extend(
            pq.read_table(f, columns=["allele"])
            .to_pandas()["allele"]
            .astype(str)  # ensure str for plotting
            .tolist()
        )

    ds_valid = None  # optional validation
    if args.valid_parquet:
        ds_valid = build_tf_dataset(args.valid_parquet,
                                    max_pep_len, max_mhc_len,
                                    seq_map, args.batch)

    # --------------------------------------------------------------
    #   MODEL
    # --------------------------------------------------------------
    encoder, enc_dec = bicross_recon_mini(max_pep_len, max_mhc_len, mask_token=MASK_TOKEN, pad_token=PAD_TOKEN)
    print(enc_dec.summary(line_length=120))

    # --------------------------------------------------------------
    #   TRAIN
    # --------------------------------------------------------------
    hist = enc_dec.fit(ds_train,
                       epochs=args.epochs,
                       verbose=1,
                       shuffle=True,
                       validation_data=ds_valid)

    with open(os.path.join(args.outdir, "history.json"), "w") as fp:
        json.dump(hist.history, fp, indent=2)
    gc.collect()

    # --------------------------------------------------------------
    #   EMBEDDING /  CLUSTERING
    # --------------------------------------------------------------
    print("Generating latent space embeddings for clustering...")
    inputs, _ = next(iter(ds_train))  # get the first batch
    latents_list = [
            encoder.predict(inputs)["latent_mhc_q"] for inputs, _ in ds_train
        ]
    latents_flat = np.concatenate(latents_list, axis=0)

    # Pool across the sequence dimension (mean)
    latents_flat = latents_flat.mean(axis=1)
    print('####DEBUG', latents_flat.shape)

    # Normalise then cluster
    print("Clustering latent space...")
    x = StandardScaler().fit_transform(latents_flat)

    # Perform K-means clustering using the provided number of clusters
    kmeans = MiniBatchKMeans(n_clusters=args.clusters,
                             random_state=0,
                             batch_size=256,
                             n_init='auto')
    labels = kmeans.fit_predict(x)

    # ------------------------------------------------------------------
    #  Assign a “majority” cluster to every allele and
    #  write the updated data set to disk
    # ------------------------------------------------------------------
    print("Deriving majority-vote cluster per allele …")

    # 1) Assemble a frame that holds one row   = one (sample, allele)
    label_frame = pd.DataFrame({"allele": allele_labels,
                                "cluster_sample": labels})

    # 2) Majority vote: for every allele pick the most frequent cluster id
    allele2cluster = (
        label_frame.groupby("allele")["cluster_sample"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )

    # 3) Map that majority id back to every single row
    label_frame["cluster"] = label_frame["allele"].map(allele2cluster)

    # 4) Merge with the original parquet rows so that
    #    *all* columns are preserved + our new “cluster”
    print("Merging cluster labels back into the original data …")
    all_rows = []  # will collect one DataFrame per parquet file
    for f in parquet_files:  # same order that allele_labels were gathered!
        df = pq.read_table(f).to_pandas()
        all_rows.append(df)

    df_full = pd.concat(all_rows, ignore_index=True)
    df_full["cluster"] = label_frame["cluster"]  # same ordering as above

    # 5) Persist
    out_parquet = os.path.join(args.outdir, "dataset_with_clusters.parquet")
    out_csv = os.path.join(args.outdir, "dataset_with_clusters.csv")
    df_full.to_parquet(out_parquet, index=False)
    df_full.to_csv(out_csv, index=False)
    print(f"✓ Updated data set written to:\n   {out_parquet}\n   {out_csv}")

    # 6) (optional) also save the allele → cluster map for later use
    pd.Series(allele2cluster, name="cluster").to_csv(
        os.path.join(args.outdir, "allele_cluster_map.csv"))
    print("✓ Allele → cluster map saved.")

    ### hierarchical clustering dendrogram
    if args.dendrogram:
            print("Creating hierarchical-clustering dendrogram from unique alleles …")

            # Load the updated dataset with clusters
            updated_df = pd.read_parquet(os.path.join(args.outdir, "dataset_with_clusters.parquet"))

            # Get unique alleles and their cluster assignments
            unique_alleles = updated_df.drop_duplicates(subset=["allele"])
            allele_labels_unique = unique_alleles["allele"].astype(str).to_numpy()
            cluster_labels_unique = unique_alleles["cluster"].to_numpy()

            # Get latent vectors for unique alleles
            latents = np.load(os.path.join(args.outdir, "cross_latents_pooled.npy"))
            # Map allele to its first occurrence index
            allele_to_idx = {allele: idx for idx, allele in enumerate(updated_df["allele"].astype(str))}
            unique_indices = [allele_to_idx[allele] for allele in allele_labels_unique]
            x_sample = latents[unique_indices]
            allele_sample = allele_labels_unique

            # ── linkage / dendrogram ─────────────────────────────────
            Z = linkage(pdist(x_sample, metric="euclidean"), method="ward")
            plt.figure(figsize=(30, 6))
            dendrogram(Z,
                       labels=allele_sample,
                       leaf_rotation=90,
                       leaf_font_size=6,
                       color_threshold=0.0,
                       orientation="top")
            plt.xlabel("Allele")
            plt.title("Hierarchical clustering of unique allele encoder latents (Ward linkage, updated dataset)")
            plt.tight_layout()
            dendro_path = os.path.join(args.outdir, "latent_dendrogram_updated.png")
            plt.savefig(dendro_path, dpi=300)
            plt.close()
            print(f"✓ Dendrogram saved to {dendro_path}")
    
    # verify on validation set if available
    if ds_valid:
        print("→ Verifying clustering on validation set …")
        enc_val = encoder.predict(ds_valid, verbose=1)["latent_mhc_q"]
        enc_val_flat = enc_val.mean(axis=1)
        enc_val_scaled = StandardScaler().fit_transform(enc_val_flat)
        val_labels = kmeans.predict(enc_val_scaled)
        # For a sanity check, show the distribution of predicted labels
        unique, counts = np.unique(val_labels, return_counts=True)
        print("Validation label distribution:")
        for lab, cnt in zip(unique, counts):
            print(f"  Cluster {lab}: {cnt} points")
        
        
    # Visualise the clusters if requested
    if args.visualise:
        print("Rendering 2-D t-SNE projection …")
        tsne = TSNE(n_components=2, init="random", learning_rate="auto")
        xy = tsne.fit_transform(x)
        plt.figure(figsize=(7, 6))
        cmap = plt.get_cmap("tab20")
        # Plot each cluster; highlight one allele (e\.g\. the first in allele_labels)
        highlight_allele = allele_labels[0] if allele_labels else None
        for lab in sorted(set(labels)):
            idx = labels == lab
            plt.scatter(xy[idx, 0], xy[idx, 1], s=6, c=[cmap(lab % 20)], label=f"cluster {lab}")
        # Highlight the selected allele in red
        if highlight_allele is not None:
            allele_idx = [i for i, a in enumerate(allele_labels) if a == highlight_allele]
            plt.scatter(xy[allele_idx, 0], xy[allele_idx, 1], s=30, c="red", label=f"allele {highlight_allele}")
        plt.title(f"K-Means clusters (k={args.clusters})")
        plt.legend(markerscale=2, fontsize=7, bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        viz_path = os.path.join(args.outdir, "latent_tsne.png")
        plt.savefig(viz_path, dpi=300)
        plt.close()
        print(f"✓ 2-D plot saved to {viz_path}")

    if args.draw_latents:
        print("Drawing the latent space heatmap …")
        plt.figure(figsize=(10, 8))
        sns.heatmap(latents_flat, cmap="viridis", cbar_kws={"label": "Latent value"})
        plt.title("Latent space heatmap (pooled across sequences)")
        plt.xlabel("Latent dimensions")
        plt.ylabel("Samples")
        heatmap_path = os.path.join(args.outdir, "latent_heatmap.png")
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print(f"✓ Heatmap saved to {heatmap_path}")


    np.save(os.path.join(args.outdir, "cross_latents_pooled.npy"), latents_flat)
    np.save(os.path.join(args.outdir, "cluster_labels.npy"), labels)
    print(f"✓ Saved pooled encoder latents and cluster labels to {args.outdir}")

    enc_dec.save_weights(os.path.join(args.outdir, "enc_dec.weights.h5"))

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError as e:
            print(e)

    # Use your actual paths here
    parquet_path = "../data/binding_affinity_data/positives_class1_subset.parquet"
    validation_path = "../data/binding_affinity_data/positives_class1_subset_rarest_alleles.parquet"
    embedding_dir = "/media/amirreza/lasse/trash/mhc1_encodings_updated.npz"
    seq_csv = "/media/amirreza/lasse/trash//mhc1_encodings_updated.csv"

    # Example argv list for running the script
    main(argv=[
        "--parquet", parquet_path,
        "--emb_npz", embedding_dir,
        "--seq_map", seq_csv,
        "--epochs", "1",
        "--batch", "256",
        "--clusters", "10",
        "--outdir", "run_minimal_dedo",
        "--valid_parquet", validation_path,
        "--visualise",
        "--dendrogram",
        "--draw_latents"
    ])