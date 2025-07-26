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
import tensorflow as tf
from tensorflow import keras

from src.utils import OHE_to_seq_single

print(sys.executable)

# =============================================================================
# CRITICAL: GPU Memory Configuration - MUST BE FIRST
# =============================================================================
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

# from models import bicross_recon
from models2 import BiCrossModel, build_bicross_net
from utils import seq_to_onehot, UNK_IDX, MASK_TOKEN, PAD_TOKEN, PAD_VALUE, MASK_VALUE, OHE_to_seq, OHE_to_seq_single

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

# TODO change, load df once and refer
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


def _get_seq_from_key(key: str, mhc_class: int, seq_map) -> str:
    """Extract a sequence from embedding key, supporting class II concatenation."""
    if mhc_class == 2 and "_" in key:
        k1, k2 = key.split("_", 1)
        seq1 = seq_map.get(k1, "")
        seq2 = seq_map.get(k2, "")
        if not seq1 or not seq2:
            raise ValueError(f"No valid sequences found for key(s): {key}")
        return f"{seq1}/{seq2}"

    seq = seq_map.get(key, "")
    if not seq:
        raise ValueError(f"No valid sequence found for key: {key}")
    return seq


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
            'max_peptide_length': int(sample_df['long_mer'].str.len().max()) if 'long_mer' in sample_df.columns else 0}
            # 'class_distribution': sample_df[
            #    'assigned_label'].value_counts().to_dict() if 'assigned_label' in sample_df.columns else {},}

        del sample_df
        return metadata


# def calculate_class_weights(parquet_path: str):
#     """Calculate class weights from a sample of the dataset"""
#     with StreamingParquetReader(parquet_path, batch_size=1000) as reader:
#         label_counts = {0: 0, 1: 0}
#         for batch_df in reader.iter_batches():
#             batch_labels = batch_df['assigned_label'].values
#             unique, counts = np.unique(batch_labels, return_counts=True)
#             for label, count in zip(unique, counts):
#                 if label in [0, 1]:
#                     label_counts[int(label)] += count
#             del batch_df
#
#     # Calculate balanced class weights
#     total = sum(label_counts.values())
#     if total == 0 or label_counts[0] == 0 or label_counts[1] == 0:
#         return {0: 1.0, 1: 1.0}
#
#     return {
#         0: total / (2 * label_counts[0]),
#         1: total / (2 * label_counts[1])
#     }


# ---------------------------------------------------------------------
# Utility that is executed in worker processes
# (must be top-level so it can be pickled on Windows)
# ---------------------------------------------------------------------
def _row_to_tensor_pack(row_dict: dict, max_pep_seq_len: int, max_mhc_len: int, masking_portion: float = 0.15, seq_map = None):
    """Convert a single row (already in plain-python dict form) into tensors."""
    # --- peptide one-hot ------------------------------------------------
    pep = row_dict["long_mer"].upper()
    pep_one_hot = seq_to_onehot(pep, max_seq_len=max_pep_seq_len)

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
        # for idx in mask_indices:
        #     if pep_one_hot[idx] != PAD_VALUE:  # only mask if not padding
        #         pep_mask_arr[idx] = MASK_TOKEN
        #     elif pep_one_hot[idx] == PAD_VALUE:
        #         pep_mask_arr[idx] = PAD_TOKEN
        #     else: # this should not happen, but just in case
        #         raise ValueError(f"Unexpected peptide value at index {idx}: {pep_one_hot[idx]}")
        for idx in mask_indices:
            if idx < len(pep):
                pep_mask_arr[idx] = MASK_TOKEN
            else:
                pep_mask_arr[idx] = PAD_TOKEN
    else:
        raise ValueError(f"Unexpected peptide value {pep}")


    # --- load MHC embedding --------------------------------------------
    embd_key = row_dict["mhc_embedding_key"]
    mhc_emb = _read_embedding_file(embd_key, EMBEDDINGS_DIR)
    mhc_seq = _get_seq_from_key(embd_key, row_dict["mhc_class"], seq_map)
    mhc_one_hot = seq_to_onehot(mhc_seq, max_seq_len=max_mhc_len)

    # Add random Gaussian noise to the MHC embedding
    mhc_emb = mhc_emb + np.random.normal(0, NOISE_STD, mhc_emb.shape).astype(np.float32)

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
            if idx < len(mhc_emb):
                mhc_mask_arr[idx] = MASK_TOKEN
            else:
                mhc_mask_arr[idx] = PAD_TOKEN
        # for idx in mask_indices_mhc:
        #     if mhc_emb[idx] != PAD_VALUE:  # only mask if not padding
        #         mhc_mask_arr[idx] = MASK_TOKEN
        #     elif mhc_emb[idx] == PAD_VALUE:
        #         mhc_mask_arr[idx] = PAD_TOKEN
        #     else:  # this should not happen, but just in case
        #         raise ValueError(f"Unexpected MHC value at index {idx}: {mhc_emb[idx]}")
    else:
        raise ValueError(f"Unexpected MHC value {mhc_emb}")

    return {
        "pep_onehot": pep_one_hot,
        "pep_mask": pep_mask_arr,
        "mhc_latent": mhc_emb,
        "mhc_mask": mhc_mask_arr,
        "mhc_onehot": mhc_one_hot,
    }, row_dict["allele"]


def streaming_data_generator(
        parquet_path: str,
        max_pep_seq_len: int,
        max_mhc_len: int,
        batch_size: int = 1000,
        seq_map = None,
        positive_only: bool = False):
    """
    Yields *individual* samples, but converts an entire Parquet batch
    on multiple CPU cores first.

    Parameters:
        parquet_path: Path to the parquet file
        max_pep_seq_len: Maximum peptide sequence length
        max_mhc_len: Maximum MHC sequence length
        batch_size: Size of batches to process
        positive_only: If True, only yield samples with positive labels (assigned_label=1)
    """
    with StreamingParquetReader(parquet_path, batch_size) as reader, \
         ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:

        # Partial function to avoid re-sending constants
        worker_fn = functools.partial(
            _row_to_tensor_pack,
            max_pep_seq_len=max_pep_seq_len,
            max_mhc_len=max_mhc_len,
            seq_map=seq_map,
        )

        for batch_df in reader.iter_batches():
            # Filter for positive samples if requested
            if positive_only and 'assigned_label' in batch_df.columns:
                batch_df = batch_df[batch_df['assigned_label'] == 1]
                if len(batch_df) == 0:
                    continue  # Skip empty batches after filtering

            # Convert Arrow table → list[dict] once; avoids pandas overhead
            dict_rows = batch_df.to_dict(orient="list")      # columns -> python lists
            # Re-shape to list[dict(row)]
            rows_iter = ( {k: dict_rows[k][i] for k in dict_rows}  # row dict
                          for i in range(len(batch_df)) )

            # Parallel map; chunksize tuned for large batches
            results = pool.map(worker_fn, rows_iter, chunksize=64)

            # Stream each converted sample back to the generator consumer
            for result in results:
                yield result

            # explicit clean-up
            del batch_df, dict_rows, rows_iter, results


def create_streaming_dataset(parquet_path: str,
                             max_pep_seq_len: int,
                             max_mhc_len: int,
                             buffer_size: int = 1000,
                             seq_map=None):
    """
    Same semantics as before, but the generator already does parallel
    preprocessing. We now ask tf.data to interleave multiple generator
    shards in parallel as well.
    """
    output_signature = (
        {
            "pep_onehot": tf.TensorSpec(shape=(max_pep_seq_len, 21), dtype=tf.float32),
            "pep_mask": tf.TensorSpec(shape=(max_pep_seq_len,), dtype=tf.float32),
            "mhc_latent": tf.TensorSpec(shape=(max_mhc_len, 1152), dtype=tf.float32),
            "mhc_mask": tf.TensorSpec(shape=(max_mhc_len,), dtype=tf.float32),
            "mhc_onehot": tf.TensorSpec(shape=(max_mhc_len, 21), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(), dtype=tf.string),  # MHC allele ID
    )

    # Create raw dataset with features and IDs
    raw_ds = tf.data.Dataset.from_generator(
        lambda: streaming_data_generator(
            parquet_path,
            max_pep_seq_len,
            max_mhc_len,
            buffer_size,
            seq_map,
            positive_only=True),  # for reconstruction task
        output_signature=output_signature,
    )

    # Parallel interleave for speed
    raw_ds = raw_ds.interleave(
        lambda *x: tf.data.Dataset.from_tensors(x if len(x) > 1 else x[0]),  # Wrap each element (or tuple) into a single-item dataset
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    # Separate dataset and IDs
    ds = raw_ds.map(lambda inputs, allele_id: inputs,
                    num_parallel_calls=tf.data.AUTOTUNE)
    ids_ds = raw_ds.map(lambda inputs, allele_id: allele_id,
                        num_parallel_calls=tf.data.AUTOTUNE)

    return ds, ids_ds


# ----------------------------------------------------------------------------
# get cross latent npy
# ----------------------------------------------------------------------------
def save_cross_latent_npy(encoder_model, ds, run_dir: str, name: str = "cross_latents_fold_{fold_id}", mhc_ids: np.ndarray = None):
    cross_latents = encoder_model.predict(ds, verbose=0)
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
def plot_training_recon_curve(history: tf.keras.callbacks.History, run_dir: str, fold_id: int = None,
                    model=None, val_dataset=None):
    """Plot training curves and validation metrics"""
    print("History keys:", history.history.keys())
    fold_str = f"_fold_{fold_id}" if fold_id is not None else ""
    plot_dir = os.path.join(run_dir, f"plots{fold_str}")
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Extract history data
    hist_dict = history.history

    # Dynamically create subplots for all keys
    num_keys = len(hist_dict)
    num_cols = 2
    num_rows = (num_keys + 1) // num_cols

    plt.figure(figsize=(14, 5 * num_rows))

    for i, key in enumerate(hist_dict.keys(), start=1):
        plt.subplot(num_rows, num_cols, i)
        plt.plot(hist_dict[key], label=f'Training {key}')
        if f'val_{key}' in hist_dict:
            plt.plot(hist_dict[f'val_{key}'], label=f'Validation {key}')
        plt.title(key.replace('_', ' ').title())
        plt.xlabel('Epoch')
        plt.ylabel(key.replace('_', ' ').title())
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    loss_plot_path = os.path.join(plot_dir, f"training_curves{fold_str}.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to {loss_plot_path}")


# TODO verify
# def plot_test_metrics(model, test_dataset, run_dir: str, fold_id: int = None,
#                       history=None, string: str = None):
#     """Plot comprehensive evaluation metrics for test dataset"""
#     fold_str = f"_fold_{fold_id}" if fold_id is not None else ""
#     test_str = f"_{string}" if string else ""
#     plot_dir = os.path.join(run_dir, f"plots{fold_str}")
#     pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
#
#     print(f"Evaluating model on test dataset...")
#
#     # Run predictions on test dataset
#     all_predictions = []
#     all_targets_pep = []
#     all_targets_mhc = []
#     all_masks_pep = []
#     all_masks_mhc = []
#
#     for i, (x_pep, pep_mask, mhc_emb, mhc_mask, mhc_target) in enumerate(test_dataset):
#         predictions = model.predict([x_pep, pep_mask, mhc_emb, mhc_mask], verbose=0)
#
#         # Extract predictions and targets
#         pep_preds = predictions['pep_reconstruction']
#         mhc_preds = predictions['mhc_reconstruction']
#
#         # Store for metrics calculation
#         all_predictions.append((pep_preds, mhc_preds))
#         all_targets_mhc.append(mhc_target)
#         all_masks_pep.append(pep_mask)
#         all_masks_mhc.append(mhc_mask)
#
#         if i >= 10:  # Limit evaluation to 10 batches for efficiency
#             break
#
#     # Create figure for evaluation metrics
#     plt.figure(figsize=(16, 14))
#
#     # 1. Peptide reconstruction accuracy for masked positions
#     plt.subplot(2, 2, 1)
#     pep_accs = []
#     for i, ((pep_pred, _), pep_target, pep_mask) in enumerate(zip(all_predictions, all_targets_pep, all_masks_pep)):
#         # Only evaluate masked positions
#         mask = tf.cast(tf.equal(pep_mask, MASK_TOKEN), tf.float32).numpy()
#
#         # For each position, get the predicted amino acid (argmax)
#         pep_pred_class = np.argmax(pep_pred, axis=-1)
#         pep_target_class = np.argmax(pep_target, axis=-1)
#
#         # Calculate accuracy only at masked positions
#         correct = (pep_pred_class == pep_target_class) * mask
#         accuracy = np.sum(correct) / (np.sum(mask) + 1e-10)
#         pep_accs.append(accuracy)
#
#     plt.bar(range(len(pep_accs)), pep_accs)
#     plt.axhline(np.mean(pep_accs), color='r', linestyle='--', label=f'Mean: {np.mean(pep_accs):.3f}')
#     plt.xlabel('Batch')
#     plt.ylabel('Accuracy')
#     plt.title('Peptide Reconstruction Accuracy (Masked Positions)')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)
#
#     # 2. MHC reconstruction accuracy for masked positions
#     plt.subplot(2, 2, 2)
#     mhc_accs = []
#     for i, ((_, mhc_pred), mhc_target, mhc_mask) in enumerate(zip(all_predictions, all_targets_mhc, all_masks_mhc)):
#         # Only evaluate masked positions
#         mask = tf.cast(tf.equal(mhc_mask, MASK_TOKEN), tf.float32).numpy()
#
#         # For each position, get the predicted amino acid (argmax)
#         mhc_pred_class = np.argmax(mhc_pred, axis=-1)
#         mhc_target_class = np.argmax(mhc_target, axis=-1)
#
#         # Calculate accuracy only at masked positions
#         correct = (mhc_pred_class == mhc_target_class) * mask
#         accuracy = np.sum(correct) / (np.sum(mask) + 1e-10)
#         mhc_accs.append(accuracy)
#
#     plt.bar(range(len(mhc_accs)), mhc_accs)
#     plt.axhline(np.mean(mhc_accs), color='r', linestyle='--', label=f'Mean: {np.mean(mhc_accs):.3f}')
#     plt.xlabel('Batch')
#     plt.ylabel('Accuracy')
#     plt.title('MHC Reconstruction Accuracy (Masked Positions)')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)
#
#     # 3. Peptide reconstruction loss
#     plt.subplot(2, 2, 3)
#     # Calculate categorical cross-entropy loss for peptide reconstruction
#     pep_losses = []
#     for ((pep_pred, _), pep_target, pep_mask) in zip(all_predictions, all_targets_pep, all_masks_pep):
#         # Only compute loss for masked positions
#         mask = tf.cast(tf.equal(pep_mask, MASK_TOKEN), tf.float32).numpy()
#         mask = mask[:, :, np.newaxis]  # Add channel dimension for broadcasting
#
#         # Calculate cross entropy loss
#         epsilon = 1e-10
#         loss = -np.sum(pep_target * np.log(pep_pred + epsilon) * mask) / (np.sum(mask) + epsilon)
#         pep_losses.append(loss)
#
#     plt.bar(range(len(pep_losses)), pep_losses)
#     plt.axhline(np.mean(pep_losses), color='r', linestyle='--', label=f'Mean: {np.mean(pep_losses):.3f}')
#     plt.xlabel('Batch')
#     plt.ylabel('Loss')
#     plt.title('Peptide Reconstruction Loss (Masked Positions)')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)
#
#     # 4. MHC reconstruction loss
#     plt.subplot(2, 2, 4)
#     # Calculate categorical cross-entropy loss for MHC reconstruction
#     mhc_losses = []
#     for ((_, mhc_pred), mhc_target, mhc_mask) in zip(all_predictions, all_targets_mhc, all_masks_mhc):
#         # Only compute loss for masked positions
#         mask = tf.cast(tf.equal(mhc_mask, MASK_TOKEN), tf.float32).numpy()
#         mask = mask[:, :, np.newaxis]  # Add channel dimension for broadcasting
#
#         # Calculate cross entropy loss
#         epsilon = 1e-10
#         loss = -np.sum(mhc_target * np.log(mhc_pred + epsilon) * mask) / (np.sum(mask) + epsilon)
#         mhc_losses.append(loss)
#
#     plt.bar(range(len(mhc_losses)), mhc_losses)
#     plt.axhline(np.mean(mhc_losses), color='r', linestyle='--', label=f'Mean: {np.mean(mhc_losses):.3f}')
#     plt.xlabel('Batch')
#     plt.ylabel('Loss')
#     plt.title('MHC Reconstruction Loss (Masked Positions)')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)
#
#     plt.tight_layout()
#     metrics_plot_path = os.path.join(plot_dir, f"test_metrics{fold_str}{test_str}.png")
#     plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
#     plt.close()
#
#     # Save metrics summary
#     metrics_summary = {
#         'peptide_reconstruction_accuracy_mean': float(np.mean(pep_accs)),
#         'peptide_reconstruction_accuracy_std': float(np.std(pep_accs)),
#         'mhc_reconstruction_accuracy_mean': float(np.mean(mhc_accs)),
#         'mhc_reconstruction_accuracy_std': float(np.std(mhc_accs)),
#         'peptide_reconstruction_loss_mean': float(np.mean(pep_losses)),
#         'mhc_reconstruction_loss_mean': float(np.mean(mhc_losses)),
#     }
#
#     with open(os.path.join(run_dir, f"metrics_summary{fold_str}{test_str}.json"), 'w') as f:
#         json.dump(metrics_summary, f, indent=4)
#
#     print(f"✓ Test metrics saved to {metrics_plot_path}")
#     return metrics_summary

def plot_ablation_study(model, dataset, run_dir: str, fold_id: int = None):
    """
    Plot ablation study results for peptide and MHC reconstruction.
    :param model: Trained model for reconstruction.
    :param dataset: dataset containing (pep_OHE, pep_mask, mhc_emb, mhc_mask)
    :param run_dir: Save directory for plots
    :param fold_id: Fold ID for saving plots in fold-specific directory
    :return: None
    """
    fold_str = f"_fold_{fold_id}" if fold_id is not None else ""
    plot_dir = os.path.join(run_dir, f"plots{fold_str}")
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Collect ablation results
    pred_results = []
    sample_count = 0
    max_samples = 5

    for (pep_OHE, pep_mask, mhc_emb, mhc_mask, mhc_ohe), allele_ids in dataset:
        if sample_count >= max_samples:
            break

        # Process each sample in the batch
        batch_size = pep_OHE.shape[0]
        for i in range(batch_size):
            if sample_count >= max_samples:
                break

            # masked_pos: model_accuracy
            ablation_results = {}
            for j in range(pep_OHE.shape[1]):
                # Mask the j-th amino acid in the peptide sequence
                masked_pep_OHE = np.copy(pep_OHE)
                masked_pep_OHE[i, j, :] = 0
                masked_pep_mask = np.copy(pep_mask)
                masked_pep_mask[i, j] = MASK_TOKEN  # Mask the position
                # Get predictions with the masked peptide
                predictions = model.predict([masked_pep_OHE, masked_pep_mask, mhc_emb, mhc_mask], verbose=0)
                pep_pred_ohe = predictions['pep_reconstruction']

                # get accuracy between pep_target[i] and pep_pred_ohe
                pep_target_class = np.argmax(pep_OHE[i], axis=-1)
                pep_pred_class = np.argmax(pep_pred_ohe[i], axis=-1)
                accuracy = np.mean(pep_target_class == pep_pred_class)
                ablation_results[j] = accuracy

            original_pep_seq = OHE_to_seq(pep_OHE[i])
            # Get MHC allele ID if it's being passed with the dataset
            mhc_id = allele_ids[i]
            if isinstance(mhc_id, bytes):
                mhc_id = mhc_id.decode('utf-8')

            # plot the ablation results
            pred_results.append({
                'pep_seq': original_pep_seq,
                'ablation_results': ablation_results,
                'masked_positions': list(ablation_results.keys()),
                'accuracies': list(ablation_results.values()),
                'mhc_allele': mhc_id,
            })
            sample_count += 1

            # plot the results for this sample
            plt.figure(figsize=(10, 6))
            positions = list(ablation_results.keys())
            accuracies = list(ablation_results.values())

            # Create bar chart with position-specific accuracies
            bars = plt.bar(positions, accuracies, color='skyblue')

            # Highlight positions with lowest accuracy (most important)
            min_acc = min(accuracies)
            for i, acc in enumerate(accuracies):
                if acc == min_acc:
                    bars[i].set_color('crimson')

            # Label x-axis with position numbers and amino acids
            plt.xticks(positions,
                       [f"{pos}:{original_pep_seq[pos]}" for pos in positions if pos < len(original_pep_seq)])

            # Add labels and title
            plt.xlabel('Peptide Position (Position:AminoAcid)')
            plt.ylabel('Reconstruction Accuracy')
            plt.title(f"Ablation Study - {original_pep_seq}\nMHC Allele: {mhc_id}")

            # Add horizontal line for average accuracy
            avg_acc = sum(accuracies) / len(accuracies)
            plt.axhline(y=avg_acc, color='gray', linestyle='--', alpha=0.7,
                        label=f'Average: {avg_acc:.3f}')

            # Set y-axis limits slightly above and below actual data range
            plt.ylim([max(0, min(accuracies) * 0.9), min(1.0, max(accuracies) * 1.1)])

            # Add legend
            plt.legend(['Average Accuracy', 'Position Accuracy'])
            plt.tight_layout()

            # Save the plot
            plt.savefig(os.path.join(plot_dir, f"ablation_sample_{sample_count}{fold_str}.png"), dpi=300)
            plt.close()



# TODO verify
def plot_reconstruction(model, dataset, SEQ_MAP, run_dir: str, fold_id: int = None):
    """
    Plot reconstruction results for peptides and MHC sequences.
    :param model: Trained model for reconstruction.
    :param dataset: dataset containing (pep_OHE, pep_mask, mhc_emb, mhc_mask) and (mhc_target, pep_target)
    :param SEQ_MAP: {"key": "sequence"} mapping for MHC sequences, key is the allele ID
    :param run_dir: Save directory for plots
    :param fold_id: Fold ID for saving plots in fold-specific directory
    :return: None
    """
    fold_str = f"_fold_{fold_id}" if fold_id is not None else ""
    plot_dir = os.path.join(run_dir, f"plots{fold_str}")
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)


    # Create a color map for amino acids (using a qualitative colormap)
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as patches

    # Collect reconstruction results
    sequences_data = []
    sample_count = 0
    max_samples = 5

    for (pep_OHE, pep_mask, mhc_emb, mhc_mask, mhc_target), allele_ids in dataset:
        if sample_count >= max_samples:
            break

        # Get predictions
        predictions = model.predict([pep_OHE, pep_mask, mhc_emb, mhc_mask], verbose=0)
        pep_pred = predictions['pep_reconstruction']
        mhc_pred = predictions['mhc_reconstruction']

        # Process each sample in the batch
        batch_size = pep_OHE.shape[0]
        for i in range(batch_size):
            if sample_count >= max_samples:
                break

            # Get original peptide and MHC sequences
            original_pep_seq = OHE_to_seq_single(pep_OHE[i])
            original_mhc_seq = OHE_to_seq_single(mhc_target[i])
            # Get predicted peptide and MHC sequences
            pep_pred_seq = OHE_to_seq_single(pep_pred[i])
            mhc_pred_seq = OHE_to_seq_single(mhc_pred[i])
            # Get MHC allele ID
            mhc_id = allele_ids[i]
            if isinstance(mhc_id, bytes):
                mhc_id = mhc_id.decode('utf-8')
            # Get MHC sequence from SEQ_MAP
            mhc_seq = SEQ_MAP.get(mhc_id, "Unknown")
            if isinstance(mhc_seq, bytes):
                mhc_seq = mhc_seq.decode('utf-8')
            # Store sequences and predictions
            sequences_data.append({
                'pep_original': original_pep_seq,
                'pep_predicted': pep_pred_seq,
                'mhc_original': original_mhc_seq,
                'mhc_predicted': mhc_pred_seq,
                'mhc_allele': mhc_id,
                'mhc_sequence': mhc_seq
            })
            sample_count += 1

            # print results for this sample
            print(sequences_data[sample_count])




# TODO verify
def plot_cross_attn(model, test_dataset, run_dir: str, fold_id: int = None):
    """Generate and save attention heatmaps for masked sequence reconstruction."""
    fold_str = f"_fold_{fold_id}" if fold_id is not None else ""
    plot_dir = os.path.join(run_dir, f"plots{fold_str}")
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Define amino acid letters for visualization
    AA = "ACDEFGHIKLMNPQRSTVWY-"  # 20 standard AAs + gap

    # Process one batch from the dataset
    for (pep_OHE, pep_mask, mhc_emb, mhc_mask, mhc_OHE) in test_dataset.take(1):
        # Get model predictions including attention scores
        predictions = model.predict(
            [pep_OHE, pep_mask, mhc_emb, mhc_mask, mhc_OHE],
            verbose=0
        )

        # Extract attention scores and average over heads if necessary
        attn_scores_ca = predictions.get("attention_scores_CA", None)
        if attn_scores_ca is not None and attn_scores_ca.ndim == 4:
            attn_scores_ca = attn_scores_ca.mean(axis=1)

        attn_scores_cnn = predictions.get("mhc_to_pep_attn_score", None)
        if attn_scores_cnn is not None and attn_scores_cnn.ndim == 4:
            attn_scores_cnn = attn_scores_cnn.mean(axis=1)

        if attn_scores_ca is None and attn_scores_cnn is None:
            print("No attention scores found in model predictions")
            return

        # Convert one-hot encodings to sequences
        pep_seqs = []
        mhc_seqs = []
        for i in range(min(5, pep_OHE.shape[0])):  # Process up to 5 samples
            pep_seq = ''.join([AA[np.argmax(pep_OHE[i, j])] for j in range(pep_OHE.shape[1])])
            mhc_seq = ''.join([AA[np.argmax(mhc_OHE[i, j])] for j in range(mhc_OHE.shape[1])])
            pep_seqs.append(pep_seq.replace('-', ''))
            mhc_seqs.append(mhc_seq.replace('-', ''))

            # Plot cross-attention maps
            plt.figure(figsize=(14, 10))

            # Plot cross-attention scores if available
            if attn_scores_ca is not None:
                plt.subplot(1, 2, 1)
                attn = attn_scores_ca[i]  # Get attention for this sample

                # Create heatmap
                sns.heatmap(attn, cmap="viridis", xticklabels=list(pep_seq),
                            yticklabels=list(mhc_seq))
                plt.title(f"Cross-Attention Map (Sample {i + 1})")
                plt.xlabel("Peptide Sequence")
                plt.ylabel("MHC Sequence")

            # Plot CNN attention if available
            if attn_scores_cnn is not None:
                plt.subplot(1, 2, 2 if attn_scores_ca is not None else 1)
                attn_cnn = attn_scores_cnn[i]

                # Create heatmap
                sns.heatmap(attn_cnn, cmap="rocket", xticklabels=list(pep_seq),
                            yticklabels=list(mhc_seq))
                plt.title(f"CNN Attention Map (Sample {i + 1})")
                plt.xlabel("Peptide Sequence")
                plt.ylabel("MHC Sequence")

            plt.tight_layout()
            attn_plot_path = os.path.join(plot_dir, f"attention_map_sample_{i + 1}{fold_str}.png")
            plt.savefig(attn_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"✓ Attention maps saved to {plot_dir}")



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
    #p.add_argument("--debug_batches", type=int, default=3,
    #               help="Number of batches to use for test dataset evaluation")

    args = p.parse_args(argv)

    run_dir = args.outdir or f"runs/run_{NOISE_STD}_{datetime.datetime.now():%Y%m%d-%H%M%S}"
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


    # load sequences and define seq_map
    base = pathlib.Path(args.dataset_path)
    if base.name == 'embeddings':
        base = base.parent
    csv_path = base / f"mhc{MHC_CLASS}_sequences.csv"
    seqs_df = pd.read_csv(csv_path, dtype={"key": str, "mhc_sequence": str})
    SEQ_MAP = dict(zip(seqs_df["key"], seqs_df["mhc_sequence"]))

    # Set metadata for MHC class - read one embedding file and set it from its shape
    emb_files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith('.npy')]
    # Remove extra '.npy' extension if present (e.g., 'file.npy.npy' -> 'file.npy')
    emb_key = emb_files[0][:-4]
    emb_file = os.path.join(EMBEDDINGS_DIR, emb_key)
    if emb_file:
        emb_shape = _read_embedding_file(emb_key, EMBEDDINGS_DIR).shape
        max_mhc_length = emb_shape[0] if len(emb_shape) > 0 else 0
        print(f"✓ MHC embedding shape: {emb_shape}, max MHC length: {max_mhc_length}")

    # max_mhc_length = mhc_seqs_df['mhc_sequence'].str.len().max() if not mhc_seqs_df.empty else 0

    # if MHC_CLASS == 2:
    #     max_mhc_length = max_mhc_length * 2  # Concatenate two sequences for class II

    # Find maximum peptide length across all datasets
    max_peptide_length = 0

    print("Scanning datasets for maximum peptide length...")
    benchmarks_dir = os.path.join(args.dataset_path, "benchmarks")
    all_parquet_files = []
    for root, _, files in os.walk(benchmarks_dir):
        for file in files:
            if file.endswith('.parquet'):
                all_parquet_files.append(os.path.join(root, file))
    print(f"Found {len(all_parquet_files)} parquet files in `{benchmarks_dir}` and its subdirectories")

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
        # print(f"  Calculating class weights...")
        # cw = calculate_class_weights(train_path)
        # print(f"  Class weights: {cw}")

        # Create streaming datasets
        train_ds, val_ids = create_streaming_dataset(train_path, max_peptide_length, max_mhc_length,
                                             buffer_size=args.buffer_size, seq_map=SEQ_MAP)
        train_ds = (train_ds
                    .shuffle(buffer_size=args.buffer_size, reshuffle_each_iteration=True)
                    .batch(args.batch)
                    #.take(args.debug_batches)
                    .prefetch(tf.data.AUTOTUNE))

        train_ids = np.asarray(val_ids)


        val_ds, val_ids =   create_streaming_dataset(val_path, max_peptide_length, max_mhc_length,
                                           buffer_size=args.buffer_size, seq_map=SEQ_MAP)
        val_ds_combined = tf.data.Dataset.zip((val_ds, val_ids))

        val_ds = (val_ds
                  .batch(args.batch)
                  #.take(args.debug_batches)
                  .prefetch(tf.data.AUTOTUNE))

        val_ds_combined = (val_ds_combined
                           .batch(args.batch)
                           #.take(args.debug_batches)
                           .prefetch(tf.data.AUTOTUNE))



        folds.append((train_ds, val_ds, val_ds_combined))
        # class_weights.append(cw)

        # Force cleanup
        cleanup_memory()

    # Create bench datasets
    print("Creating test datasets...")
    n_benchs = 10 # CHANGE
   # Set up benchmark directory paths
    BENCHMARKS_DIR = [os.path.join(args.dataset_path, "benchmarks")]
    bench_datasets = []
    bench_ids = []

    # Create benchmark datasets
    # for folder in BENCHMARKS_DIR:
    #     if os.path.exists(folder):
    #         bench_files = [f for f in os.listdir(folder) if f.endswith('.parquet')]
    #         n_benchs = len(bench_files)
    #         print(f"Found {n_benchs} benchmark files in {folder}")
    #
    #         for file in bench_files:
    #             bench_path = os.path.join(folder, file)
    #             bench_name = os.path.splitext(file)[0]
    #             print(f"Creating benchmark dataset for {bench_name}...")
    #
    #             bench_ds, bench_id = create_streaming_dataset(
    #                 bench_path,
    #                 max_peptide_length,
    #                 max_mhc_length,
    #                 buffer_size=args.buffer_size,
    #                 seq_map=SEQ_MAP
    #             )
    #
    #             bench_ds = (bench_ds
    #                         .batch(args.batch)
    #                         .prefetch(tf.data.AUTOTUNE))
    #
    #             bench_ids.append(np.array(list(bench_id.as_numpy_iterator())))
    #             bench_datasets.append((bench_ds, bench_name))
    #             print(f"✓ Created benchmark dataset: {bench_name}")



    print(f"✓ Created {n_folds} fold datasets and {n_benchs} benchmark datasets")
    print("Memory after dataset creation:")
    monitor_memory()

    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for fold_id, (train_loader, val_loader, val_comb_loader) in enumerate(folds):
        print(f'\n🔥 Training fold {fold_id}/{n_folds}')

        # Clean up before each fold
        cleanup_memory()

        # Build fresh model for each fold
        print(f"Building model with max_pep_len={max_peptide_length}, max_mhc_len={max_mhc_length}")
        model = build_bicross_net(max_peptide_length, max_mhc_length, mask_token=MASK_TOKEN,
                                  pad_token=PAD_TOKEN, pep_emb_dim=128, mhc_emb_dim=128, heads=4)

        model.compile(optimizer=keras.optimizers.Adam(1e-3))


        # Callbacks
        # ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=os.path.join(run_dir, f'best_fold_{fold_id}.weights.h5'),
        #     monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        # early_cb = tf.keras.callbacks.EarlyStopping(
        #     monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

        # Verify data shapes
        # for (x_pep, pep_mask_arr, mhc_embed, mhc_mask_arr), allele in train_loader.take(1):
        #     print(f"✓ Input shapes: peptide={x_pep.shape}, pep_mask={pep_mask_arr.shape}, mhc_embed={mhc_embed.shape}, mhc_mask={mhc_mask_arr.shape}, allele={allele.shape}")
        #     break

        print("Memory before training:")
        monitor_memory()

        if not train_loader:
            raise ValueError("The train_loader dataset is empty. Check dataset path or preprocessing logic.")
        batch = next(iter(train_loader))
        pep_OHE = batch["pep_onehot"]
        pep_mask = batch["pep_mask"]
        mhc_emb  = batch["mhc_latent"]
        mhc_mask = batch["mhc_mask"]
        mhc_OHE  = batch["mhc_onehot"]
        # Train model
        print("🚀 Starting training...")

        hist = model.fit(
            x=train_loader,
            validation_data=val_loader,
            epochs=args.epochs,
            # callbacks=[ckpt_cb, early_cb],
            verbose=1,
        )

        print("Memory after training:")
        monitor_memory()

        # Plot training curves
        print("📈 Plotting training curves...")
        plot_training_recon_curve(hist, run_dir, fold_id, model=model, val_dataset=val_loader)
        # plot_reconstruction(model=encoder_decoder_model,dataset=val_comb_loader, run_dir=run_dir, SEQ_MAP=SEQ_MAP, fold_id=fold_id)
        # plot_ablation_study(model=encoder_decoder_model,dataset=val_comb_loader, run_dir=run_dir, fold_id=fold_id)
        # plot_cross_attn(encoder_decoder_model, val_comb_loader, run_dir, fold_id)
        # plot_test_metrics(encoder_decoder_model, val_comb_loader, run_dir, fold_id)

        # Save model and metadata
        model.save_weights(os.path.join(run_dir, f'model_fold_{fold_id}.weights.h5'))
        metadata = {
            "fold_id": fold_id,
            "epochs": args.epochs,
            "batch_size": args.batch,
            "max_peptide_length": max_peptide_length,
            "max_mhc_length": max_mhc_length,
            "run_dir": run_dir,
            "mhc_class": MHC_CLASS
        }
        with open(os.path.join(run_dir, f'metadata_fold_{fold_id}.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

        # Evaluate on benchmark datasets
        # print(f"\n📊 Evaluating fold {fold_id} on benchmark datasets...")
        #
        # for bench_ds, bench_name in bench_datasets:
        #     print(f"Evaluating on benchmark: {bench_name}...")
        #
        #     # Plot metrics for this benchmark
        #     # metrics_summary = plot_test_metrics(
        #     #     model,
        #     #     bench_ds,
        #     #     run_dir,
        #     #     fold_id,
        #     #     string=f"bench_{bench_name}"
        #     # )
        #
        #     # Find the matching bench_id index
        #     bench_idx = [i for i, (ds, name) in enumerate(bench_datasets) if name == bench_name][0]
        #
        #     # Save cross latents for this benchmark dataset
        #     save_cross_latent_npy(
        #         model,
        #         bench_ds,
        #         run_dir,
        #         name=f"cross_latent_{bench_name}_fold_{fold_id}",
        #         mhc_ids=bench_ids[bench_idx] if bench_idx < len(bench_ids) else None
        #     )
        #
        #     print(f"✓ Completed evaluation on {bench_name}")

        print(f"✅ Fold {fold_id} completed successfully")

        # Cleanup
        del model, hist
        cleanup_memory()

    print("\n🎉 Training completed successfully!")
    print(f"📁 All results saved to: {run_dir}")


if __name__ == "__main__":
    BUFFER = 1024  # Reduced buffer size for memory efficiency
    NOISE_STD = 0.01  # Standard deviation for Gaussian noise
    MHC_CLASS = 1
    dataset_path = f"../data/mhc_{MHC_CLASS}"
    EMBEDDINGS_DIR = f"../data/mhc_{MHC_CLASS}/embeddings"
    main([
        "--dataset_path", dataset_path,
        "--epochs", "10",
        "--batch", "256",
        "--buffer_size", "1024",
        #"--debug_batches", "3",
    ])