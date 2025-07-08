#!/usr/bin/env python
"""
=========================

MEMORY-OPTIMIZED End‑to‑end trainer for a **peptide×MHC cross‑attention reconstruction**.
Loads parquet files in true streaming fashion without loading entire datasets into memory.
Loads the corresponding latent embeddings from disk on demand using the keys in the parquet file.

Author: Amirreza (memory-optimized version, 2025)
"""

from __future__ import annotations
import os
import sys

print(sys.executable)

# =============================================================================
# CRITICAL: GPU Memory Configuration - MUST BE FIRST
# =============================================================================
import tensorflow as tf


def configure_gpu_memory():
    """Configure TensorFlow to use GPU memory efficiently"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU memory growth enabled")
        else:
            print("No GPUs found - running on CPU")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")


# Configure GPU immediately
configure_gpu_memory()

# ---------------------------------------------------------------------
# ► Use all logical CPU cores for TF ops that still run on CPU
# ---------------------------------------------------------------------
NUM_CPUS = os.cpu_count() or 1
tf.config.threading.set_intra_op_parallelism_threads(NUM_CPUS)
tf.config.threading.set_inter_op_parallelism_threads(NUM_CPUS)
print(f'✓ TF intra/inter-op threads set to {NUM_CPUS}')

# Set memory-friendly environment variables
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import math
import argparse, datetime, pathlib, json
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_score,
    recall_score, f1_score, accuracy_score, roc_auc_score
)
import seaborn as sns
import pyarrow.parquet as pq
import gc
import weakref
import pyarrow as pa, pyarrow.compute as pc

from concurrent.futures import ProcessPoolExecutor
import functools, itertools

from model_recon import bicross_recon

pa.set_cpu_count(os.cpu_count())


# =============================================================================
# Memory monitoring functions
# =============================================================================
def monitor_memory():
    """Monitor system memory usage"""
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.used / 1e9:.1f}GB / {memory.total / 1e9:.1f}GB ({memory.percent:.1f}% used)")

    try:
        from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
        nvmlInit()
        deviceCount = nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            print(
                f"GPU {i}: {info.used / 1e9:.1f}GB / {info.total / 1e9:.1f}GB ({100 * info.used / info.total:.1f}% used)")
    except:
        print("GPU memory monitoring not available")


def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    try:
        tf.keras.backend.clear_session()
    except:
        pass


# ----------------------------------------------------------------------------
# Peptide encoding utilities
# ----------------------------------------------------------------------------
AA = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard AAs, order fixed
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}
UNK_IDX = 21  # index for unknown
PAD_VALUE_PEP = 0 # set manually to avoid confusion with UNK
PAD_TOKEN_PEP = -2  # padding token for peptides
PAD_VALUE_MHC = 0
PAD_TOKEN_MHC = -2  # padding token for MHC
MASK_IDX = -1



def peptides_to_onehot(sequence: str, max_seq_len: int) -> np.ndarray:
    """Convert peptide sequence to one-hot encoding"""
    arr = np.full((max_seq_len, 21), PAD_TOKEN_PEP, dtype=np.float32) # initialize padding with -2
    for j, aa in enumerate(sequence.upper()[:max_seq_len]):
        arr[j, AA_TO_IDX.get(aa, UNK_IDX)] = 1.0
        # print number of UNKs in the sequence
    num_unks = np.sum(arr[:, UNK_IDX])
    if num_unks > 0:
        print(f"Warning: {num_unks} unknown amino acids in sequence '{sequence}'")
    return arr


def _read_embedding_file(key: str, path: str | os.PathLike) -> np.ndarray:
    """Robust loader for latent embeddings"""
    path = f"{path}/{key}.npy" if isinstance(path, str) else os.path.join(path, f"{key}.npy")
    try:
        arr = np.load(path)
        if isinstance(arr, np.ndarray) and arr.dtype == np.float32:
            return arr
        raise ValueError
    except ValueError:
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            obj = obj.item()
        if isinstance(obj, dict) and "embedding" in obj:
            return obj["embedding"].astype("float32")
        raise ValueError(f"Unrecognised embedding file {path}")


# ----------------------------------------------------------------------------
# Streaming dataset utilities
# ----------------------------------------------------------------------------
class StreamingParquetReader:
    """Memory-efficient streaming parquet reader"""

    def __init__(self, parquet_path: str, batch_size: int = 1000):
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self._file = None
        self._num_rows = None

    def __enter__(self):
        self._file = pq.ParquetFile(self.parquet_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file = None

    @property
    def num_rows(self):
        """Get total number of rows without loading data"""
        if self._num_rows is None:
            if self._file is None:
                with pq.ParquetFile(self.parquet_path) as f:
                    self._num_rows = f.metadata.num_rows
            else:
                self._num_rows = self._file.metadata.num_rows
        return self._num_rows

    def iter_batches(self):
        """Iterate over parquet file in batches"""
        if self._file is None:
            raise RuntimeError("Reader not opened. Use within 'with' statement.")

        for batch in self._file.iter_batches(batch_size=self.batch_size):
            df = batch.to_pandas()
            yield df
            del df, batch  # Explicit cleanup

    def sample_for_metadata(self, n_samples: int = 1000):
        """Sample a small portion for metadata extraction"""
        with pq.ParquetFile(self.parquet_path) as f:
            # Read first batch for metadata
            first_batch = next(f.iter_batches(batch_size=min(n_samples, self.num_rows)))
            return first_batch.to_pandas()


def get_dataset_metadata(parquet_path: str):
    """Extract dataset metadata without loading full dataset"""
    with StreamingParquetReader(parquet_path) as reader:
        sample_df = reader.sample_for_metadata(reader.num_rows)

        metadata = {
            'total_rows': reader.num_rows,
            'max_peptide_length': int(sample_df['long_mer'].str.len().max()) if 'long_mer' in sample_df.columns else 0,
            'class_distribution': sample_df[
                'assigned_label'].value_counts().to_dict() if 'assigned_label' in sample_df.columns else {},
        }

        del sample_df
        return metadata


def calculate_class_weights(parquet_path: str):
    """Calculate class weights from a sample of the dataset"""
    with StreamingParquetReader(parquet_path, batch_size=1000) as reader:
        label_counts = {0: 0, 1: 0}
        for batch_df in reader.iter_batches():
            batch_labels = batch_df['assigned_label'].values
            unique, counts = np.unique(batch_labels, return_counts=True)
            for label, count in zip(unique, counts):
                if label in [0, 1]:
                    label_counts[int(label)] += count
            del batch_df

    # Calculate balanced class weights
    total = sum(label_counts.values())
    if total == 0 or label_counts[0] == 0 or label_counts[1] == 0:
        return {0: 1.0, 1: 1.0}

    return {
        0: total / (2 * label_counts[0]),
        1: total / (2 * label_counts[1])
    }


# ---------------------------------------------------------------------
# Utility that is executed in worker processes
# (must be top-level so it can be pickled on Windows)
# ---------------------------------------------------------------------
def _row_to_tensor_pack(row_dict: dict, max_pep_seq_len: int, max_mhc_len: int, masking_portion: float = 0.15):
    """Convert a single row (already in plain-python dict form) into tensors."""
    # --- peptide one-hot ------------------------------------------------
    pep = row_dict["long_mer"].upper()
    pep_one_hot = peptides_to_onehot(pep, max_seq_len=max_pep_seq_len)

    # create an array of random masking positions of peptide. use MASK_IDX for masked positions and 1 for unmasked positions, and PAD_TOKEN_PEP for padding positions eg. [-1, -1, 0, 0, -2, -2]
    # select random positions to mask that are not padding positions
    pep_mask_arr = np.full((max_pep_seq_len,), 0, dtype=np.float32)  # start with all unmasked
    num_masked = int(max_pep_seq_len * masking_portion)
    if num_masked > 0:
        # Choose random indices to mask, avoiding padding positions
        mask_indices = np.random.choice(
            np.arange(max_pep_seq_len),
            size=num_masked,
            replace=False
        )
        for idx in mask_indices:
            if pep_one_hot[idx] != PAD_VALUE_PEP:  # only mask if not padding
                pep_mask_arr[idx] = MASK_IDX
            elif pep_one_hot[idx] == PAD_VALUE_PEP:
                pep_mask_arr[idx] = PAD_TOKEN_PEP
            else: # this should not happen, but just in case
                raise ValueError(f"Unexpected peptide value at index {idx}: {pep_one_hot[idx]}")
    else:
        raise ValueError(f"Unexpected peptide value {pep}")



    # --- load MHC embedding --------------------------------------------
    embd_key = row_dict["mhc_embedding_key"]
    mhc_emb = _read_embedding_file(embd_key, EMBEDDINGS_DIR)

    # Add random Gaussian noise to the MHC embedding
    noise_std = NOISE_STD
    mhc_embed = mhc_emb + np.random.normal(0, noise_std, mhc_emb.shape)

    # pad MHC to max_mhc_len if needed
    if mhc_emb.shape[0] < max_mhc_len:
        pad_mhc = np.full((max_mhc_len, mhc_emb.shape[1]), PAD_VALUE_MHC, dtype=np.float32)
        pad_mhc[:mhc_emb.shape[0]] = mhc_emb
        mhc_emb = pad_mhc
    elif mhc_emb.shape[0] > max_mhc_len:
        raise ValueError(
            f"MHC length {mhc_emb.shape[0]} exceeds max_mhc_len {max_mhc_len} for row: {row_dict}")

    if mhc_emb.shape[0] != max_mhc_len:               # sanity check
        raise ValueError(f"MHC length mismatch: {mhc_emb.shape[0]} vs {max_mhc_len}")
    mhc_emb = mhc_emb.astype("float32")

    # make the mask array, choose 0.15 random positions to mask. Make an array set 0 for masked positions, 1 for unmasked and pad the rest with -1
    mhc_mask_arr = np.full((max_mhc_len,), 1, dtype=np.float32)  # start with all unmasked
    num_masked_mhc = int(max_mhc_len * masking_portion)
    if num_masked_mhc > 0:
        # Choose random indices to mask, avoiding padding positions
        mask_indices_mhc = np.random.choice(
            np.arange(max_mhc_len),
            size=num_masked_mhc,
            replace=False
        )
        for idx in mask_indices_mhc:
            if mhc_emb[idx] != PAD_VALUE_MHC:  # only mask if not padding
                mhc_mask_arr[idx] = MASK_IDX
            elif mhc_emb[idx] == PAD_VALUE_MHC:
                mhc_mask_arr[idx] = PAD_TOKEN_MHC
            else:  # this should not happen, but just in case
                raise ValueError(f"Unexpected MHC value at index {idx}: {mhc_emb[idx]}")
    else:
        raise ValueError(f"Unexpected MHC value {mhc_emb}")

    return (pep_one_hot, pep_mask_arr, mhc_embed, mhc_mask_arr), row_dict["allele"]



def streaming_data_generator(
        parquet_path: str,
        max_pep_seq_len: int,
        max_mhc_len: int,
        batch_size: int = 1000):
    """
    Yields *individual* samples, but converts an entire Parquet batch
    on multiple CPU cores first.
    """
    with StreamingParquetReader(parquet_path, batch_size) as reader, \
         ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:

        # Partial function to avoid re-sending constants
        worker_fn = functools.partial(
            _row_to_tensor_pack,
            max_pep_seq_len=max_pep_seq_len,
            max_mhc_len=max_mhc_len,
        )

        for batch_df in reader.iter_batches():
            # Convert Arrow table → list[dict] once; avoids pandas overhead
            dict_rows = batch_df.to_dict(orient="list")      # columns -> python lists
            # Re-shape to list[dict(row)]
            rows_iter = ( {k: dict_rows[k][i] for k in dict_rows}  # row dict
                          for i in range(len(batch_df)) )

            # Parallel map; chunksize tuned for large batches
            results = pool.map(worker_fn, rows_iter, chunksize=64)

            # # Stream each converted sample back to the generator consumer
            # yield from results      # <-- keeps memory footprint tiny
            for result, sample_id in zip(results, dict_rows["allele"]):
                yield result + (sample_id,)

            # explicit clean-up
            del batch_df, dict_rows, rows_iter, results

# TODO
def create_streaming_dataset(parquet_path: str,
                             max_pep_seq_len: int,
                             max_mhc_len: int,
                             batch_size: int = 128,
                             buffer_size: int = 1000):
    """
    Same semantics as before, but the generator already does parallel
    preprocessing.  We now ask tf.data to interleave multiple generator
    shards in parallel as well.
    """
    output_signature = (
        (
            tf.TensorSpec(shape=(max_pep_seq_len, 21),  dtype=tf.float32),
            tf.TensorSpec(shape=(max_pep_seq_len,),     dtype=tf.float32),  # Mask for peptide
            tf.TensorSpec(shape=(max_mhc_len, 1152),    dtype=tf.float32),
            tf.TensorSpec(shape=(max_mhc_len,),         dtype=tf.float32),  # Mask for MHC
        ),
        tf.TensorSpec(shape=(), dtype=tf.string),  # MHC allele ID
    )

    # Create raw dataset with features and IDs
    raw_ds = tf.data.Dataset.from_generator(
        lambda: streaming_data_generator(
            parquet_path,
            max_pep_seq_len,
            max_mhc_len,
            buffer_size),
        output_signature=output_signature,
    )

    # Parallel interleave for speed
    raw_ds = raw_ds.interleave(
        lambda pep_one_hot, pep_mask_arr, mhc_embed, mhc_mask_arr, mhc_id: tf.data.Dataset.from_tensors((pep_one_hot, pep_mask_arr, mhc_embed, mhc_mask_arr, mhc_id)),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    # Separate dataset and IDs
    ds = raw_ds.map(lambda pep_one_hot, pep_mask_arr, mhc_embed, mhc_mask_arr, _: (pep_one_hot, pep_mask_arr, mhc_embed, mhc_mask_arr),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ids_ds = raw_ds.map(lambda _, __, ___, ____, mhc_id: mhc_id,
                     num_parallel_calls=tf.data.AUTOTUNE)

    return ds, ids_ds


# ----------------------------------------------------------------------------
# get cross latent npy
# ----------------------------------------------------------------------------
def save_cross_latent_npy(cross_latent_model, ds, run_dir: str, name: str = "cross_latents_fold_{fold_id}", mhc_ids: np.ndarray = None):
    cross_latents = cross_latent_model.predict(ds, verbose=0)
    save_path = os.path.join(run_dir, f'{name}.npz')
    # Prepare data for saving
    if mhc_ids is not None:
        if isinstance(mhc_ids, tf.data.Dataset):
            mhc_ids = np.array(list(mhc_ids.as_numpy_iterator()))
        if isinstance(cross_latents, tf.Tensor):
            cross_latents = cross_latents.numpy()
        savez_kwargs = {'cross_latents': cross_latents}
        if mhc_ids is not None:
            savez_kwargs['mhc_ids'] = mhc_ids
    else:
        np.save(save_path.replace('.npz', '.npy'), cross_latents)

# ----------------------------------------------------------------------------
# Visualization utilities (keeping the same as original)
# ----------------------------------------------------------------------------
def plot_training_curve(history: tf.keras.callbacks.History, run_dir: str, fold_id: int = None,
                        model=None, val_dataset=None):
    """Plot training curves and validation metrics"""
    # TODO


def plot_test_metrics(model, test_dataset, run_dir: str, fold_id: int = None,
                      history=None, string: str = None):
    """Plot comprehensive evaluation metrics for test dataset"""
    # TODO

def plot_reconstruction(model, test_dataset, run_dir: str, fold_id: int = None):
    """Plot reconstruction results for test dataset"""
    # TODO

def plot_attn(att_model, val_loader, run_dir: str, fold_id: int = None):
    # TODO revision
    """Generate and save attention heatmaps for 5 samples."""
    # -------------------------------------------------------------
    # ATTENTION VISUALISATION – take ONE batch from validation
    # -------------------------------------------------------------
    (pep_ex, mhc_ex), labels = next(iter(val_loader))  # first batch
    att_scores = att_model.predict([pep_ex, mhc_ex], verbose=0)
    print("attn_scores", att_scores.shape)
    # save attention scores
    if fold_id is None:
        fold_id = 0
    run_dir = os.path.join(run_dir, f"fold_{fold_id}")
    pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
    print(f"✓ Attention scores saved to {run_dir}")
    out_attn = os.path.join(run_dir, f"attn_scores_fold{fold_id}.npy")
    np.save(out_attn, att_scores)
    print(f"✓ Attention scores saved to {out_attn}")
    # -------------------------------------------------------------
    # att_scores shape : (B, heads, pep_len, mhc_len)
    att_mean = att_scores.mean(axis=1)  # (B,pep,mhc)
    print("att_mean shape:", att_mean.shape)
    # Find positive and negative samples
    labels_np = labels.numpy().flatten()
    pos_indices = np.where(labels_np == 1)[0]
    neg_indices = np.where(labels_np == 0)[0]
    # Select up to 5 samples (prioritize positive samples, then use negative if needed)
    num_pos = min(5, len(pos_indices))
    num_neg = min(5 - num_pos, len(neg_indices)) if num_pos < 5 else 0
    selected_pos = pos_indices[:num_pos]
    selected_neg = neg_indices[:num_neg] if num_neg > 0 else []
    selected_samples = list(selected_pos) + list(selected_neg)
    print(f"Plotting attention maps for {len(selected_pos)} positive and {len(selected_neg)} negative samples")
    # Generate heatmaps for each selected sample
    for i, sample_id in enumerate(selected_samples[:5]):
        sample_type = "Positive" if sample_id in pos_indices else "Negative"
        A = att_mean[sample_id]
        A = A.transpose()
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(
            A,
            cmap="viridis",
            xticklabels=[
                AA[pep_ex[sample_id][j].numpy().argmax()] if float(tf.reduce_sum(pep_ex[sample_id][j])) > 0 else ""
                for j in range(A.shape[1])
            ],
            yticklabels=[f"M{i}" for i in range(A.shape[0])],
            cbar_kws={"label": "attention"},
            linewidths=0.1,  # Add lines between cells
            linecolor='black',  # White lines for better contrast
            linestyle=':'  # Dashed lines
        )
        # Improve labels and title
        plt.xlabel("Peptide Position (Amino Acid)")
        plt.ylabel("MHC Position")
        plt.title(f"Fold {fold_id} - Attention Heatmap\nSample {sample_id} ({sample_type} Example)")
        # Add box around the entire heatmap
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(2)
        out_png = os.path.join(run_dir, f"attention_fold{fold_id}_sample{sample_id}_{sample_type.lower()}.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Attention heat-map {i+1}/5 saved to {out_png}")


# ----------------------------------------------------------------------------
# Main training function
# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True,
                   help="Path to the dataset directory")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--outdir", default=None,
                   help="Output dir (default: runs/run_YYYYmmdd-HHMMSS)")
    p.add_argument("--buffer_size", type=int, default=1000,
                   help="Buffer size for streaming data loading")
    p.add_argument("--test_batches", type=int, default=3,
                   help="Number of batches to use for test dataset evaluation")

    args = p.parse_args(argv)

    run_dir = args.outdir or f"runs/run_{datetime.datetime.now():%Y%m%d-%H%M%S}"
    pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
    print(f"★ Outputs → {run_dir}\n")

    # Set seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    print("Setting random seeds for reproducibility...")

    print("Initial memory state:")
    monitor_memory()

    # Extract metadata from datasets without loading them fully
    print("Extracting dataset metadata...")

    # Get fold information
    fold_dir = os.path.join(args.dataset_path, 'folds')
    fold_files = sorted([f for f in os.listdir(fold_dir) if f.endswith('.parquet')])
    n_folds = len(fold_files) // 2

    # Find maximum peptide length across all datasets
    max_peptide_length = 0
    max_mhc_length = 180 # TODO set from dataset metadata

    print("Scanning datasets for maximum peptide length...")
    all_parquet_files = [
        os.path.join(args.dataset_path, "test1.parquet"),
        os.path.join(args.dataset_path, "test2.parquet")
    ]

    # Add fold files
    for i in range(1, n_folds + 1):
        all_parquet_files.extend([
            os.path.join(fold_dir, f'fold_{i}_train.parquet'),
            os.path.join(fold_dir, f'fold_{i}_val.parquet')
        ])

    for pq_file in all_parquet_files:
        if os.path.exists(pq_file):
            metadata = get_dataset_metadata(pq_file)
            max_peptide_length = max(max_peptide_length, metadata['max_peptide_length'])
            print(
                f"  {os.path.basename(pq_file)}: max_len={metadata['max_peptide_length']}, rows={metadata['total_rows']}")

    print(f"✓ Maximum peptide length across all datasets: {max_peptide_length}")

    # Create fold datasets and class weights
    folds = []
    class_weights = []

    for i in range(1, n_folds + 1):
        print(f"\nProcessing fold {i}/{n_folds}")
        train_path = os.path.join(fold_dir, f'fold_{i}_train.parquet')
        val_path = os.path.join(fold_dir, f'fold_{i}_val.parquet')

        # Calculate class weights from training data
        print(f"  Calculating class weights...")
        cw = calculate_class_weights(train_path)
        print(f"  Class weights: {cw}")

        # Create streaming datasets
        train_ds, val_ids = create_streaming_dataset(train_path, max_peptide_length, max_mhc_length,
                                             buffer_size=args.buffer_size)
        train_ds = (train_ds
                    .shuffle(buffer_size=args.buffer_size, reshuffle_each_iteration=True)
                    .batch(args.batch)
                    .take(args.test_batches)
                    .prefetch(tf.data.AUTOTUNE))

        train_ids = np.asarray(val_ids)


        val_ds, val_ids =   create_streaming_dataset(val_path, max_peptide_length, max_mhc_length,
                                           buffer_size=args.buffer_size)
        val_ds = (val_ds
                  .batch(args.batch)
                  .take(args.test_batches)
                  .prefetch(tf.data.AUTOTUNE))


        folds.append((train_ds, val_ds))
        class_weights.append(cw)

        val_ids = np.asarray(val_ids)

        # Force cleanup
        cleanup_memory()

    # Create bench datasets
    print("Creating test datasets...")
    n_benchs = 10 # CHANGE
    # for folder in BENCHMARKS_DIR:
        # for file in os.listdir(folder):
            # create bench ds named file_name_bench_ds
    # TODO
    # test1_ds, test1_ids = create_streaming_dataset(os.path.join(args.dataset_path, "test1.parquet"),
    #                                      max_peptide_length, max_mhc_length, buffer_size=args.buffer_size)
    # test1_ds = (test1_ds
    #             .batch(args.batch)
    #             .prefetch(tf.data.AUTOTUNE))
    #
    # test1_ids = np.array(list(test1_ids.as_numpy_iterator()))
    #
    # test2_ds, test2_ids = create_streaming_dataset(os.path.join(args.dataset_path, "test2.parquet"),
    #                                      max_peptide_length, max_mhc_length, buffer_size=args.buffer_size)
    # test2_ds = (test2_ds
    #             .batch(args.batch)
    #             .prefetch(tf.data.AUTOTUNE))
    #
    # test2_ids = np.array(list(test2_ids.as_numpy_iterator()))


    print(f"✓ Created {n_folds} fold datasets and {n_benchs} benchmark datasets")
    print("Memory after dataset creation:")
    monitor_memory()

    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for fold_id, ((train_loader, val_loader), class_weight) in enumerate(zip(folds, class_weights), start=1):
        print(f'\n🔥 Training fold {fold_id}/{n_folds}')

        # Clean up before each fold
        cleanup_memory()

        # Build fresh model for each fold
        print("Building model...")
        model, attn_model, cross_latent_model = bicross_recon(max_peptide_length,max_mhc_length)
        model.summary()

        # Callbacks
        ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(run_dir, f'best_fold_{fold_id}.weights.h5'),
            monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        early_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

        # Verify data shapes
        for (x_pep, latents), labels in train_loader.take(1):
            print(f"✓ Input shapes: peptide={x_pep.shape}, mhc={latents.shape}, labels={labels.shape}")
            break

        print("Memory before training:")
        monitor_memory()

        # Train model
        print("🚀 Starting training...")
        hist = model.fit(
            train_loader,
            validation_data=val_loader,
            epochs=args.epochs,
            class_weight=class_weight,
            callbacks=[ckpt_cb, early_cb],
            verbose=1,
        )

        print("Memory after training:")
        monitor_memory()

        # Plot training curves
        plot_training_curve(hist, run_dir, fold_id, model, val_loader)
        plot_attn(attn_model, val_loader, run_dir, fold_id)

        # Save model and metadata
        model.save_weights(os.path.join(run_dir, f'model_fold_{fold_id}.weights.h5'))
        metadata = {
            "fold_id": fold_id,
            "epochs": args.epochs,
            "batch_size": args.batch,
            "max_peptide_length": max_peptide_length,
            "max_mhc_length": max_mhc_length,
            "class_weights": class_weight,
            "run_dir": run_dir,
            "mhc_class": MHC_CLASS
        }
        with open(os.path.join(run_dir, f'metadata_fold_{fold_id}.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

        # TODO evaluate on benchmarks
        # # Evaluate on test sets
        # print(f"\n📊 Evaluating fold {fold_id} on test sets...")
        #
        # # Test1 evaluation
        # print("Evaluating on test1 (balanced alleles)...")
        # plot_test_metrics(model, test1_ds, run_dir, fold_id, string="Test1_balanced_alleles")
        #
        # # Test2 evaluation
        # print("Evaluating on test2 (rare alleles)...")
        # plot_test_metrics(model, test2_ds, run_dir, fold_id, string="Test2_rare_alleles")
        #
        # # save cross_latents for test1 and test2
        # save_cross_latent_npy(cross_latent_model, test1_ds, run_dir, name=f"cross_latent_test1_fold_{fold_id}", mhc_ids=test1_ids, labels_only=test1_labels_copy)
        # save_cross_latent_npy(cross_latent_model, test2_ds, run_dir, name=f"cross_latent_test2_fold_{fold_id}", mhc_ids=test2_ids, labels_only=test2_labels_copy)

        print(f"✅ Fold {fold_id} completed successfully")

        # Cleanup
        del model, hist
        cleanup_memory()

    print("\n🎉 Training completed successfully!")
    print(f"📁 All results saved to: {run_dir}")


if __name__ == "__main__":
    BUFFER = 8192  # Reduced buffer size for memory efficiency
    NOISE_STD = 0.01  # Standard deviation for Gaussian noise
    MHC_CLASS = 2
    dataset_path = f"../data/Custom_dataset/PMGen_sequences/mhc_{MHC_CLASS}"
    EMBEDDINGS_DIR = f"../data/Custom_dataset/PMGen_whole_seq/mhc_{MHC_CLASS}/embeddings"
    main([
        "--dataset_path", dataset_path,
        "--epochs", "5",
        "--batch", "128",
        "--buffer_size", "8192",
        "--test_batches", "2000",
    ])