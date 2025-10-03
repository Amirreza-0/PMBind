#!/usr/bin/env python
"""
Optimized training script for pmbind_multitask.

Features:
- Uses pre-generated TFRecord files (`create_tfrecords.py`) storing peptide indices, MHC indices, and embedding IDs.
- Loads a consolidated MHC embedding lookup (`mhc_embedding_lookup.npz`) once as a constant tensor to eliminate per-sample disk access.
- Reconstructs full tensor batches on-the-fly from compact identifiers to reduce storage footprint and improve I/O efficiency.
- Supports  mixed precision.
- Employs a high-throughput `tf.data` pipeline (parallel interleave, map, prefetch, large batches) for scalable GPU training.
"""
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse
from sklearn.utils.class_weight import compute_class_weight

# Local imports
from utils import (get_embed_key, clean_key, NORM_TOKEN, MASK_TOKEN, PAD_TOKEN, PAD_VALUE, MASK_VALUE,
                   masked_categorical_crossentropy, BinaryMCC,
                   seq_to_indices_blosum62, seq_to_ohe_indices, seq_to_blossom62, seq_to_onehot,
                   PAD_INDEX_62, BLOSUM62, AA, PAD_INDEX_OHE, AMINO_ACID_MAP, AA_BLOSUM,
                   AsymmetricPenaltyBinaryCrossentropy, load_metadata, load_embedding_table, normalize_embedding_tf,
                   load_embedding_db, apply_dynamic_masking, min_max_norm, log_norm_zscore, _preprocess_df_chunk)

from models import pmbind_multitask as pmbind
from visualizations import visualize_training_history

mixed_precision = True  # Enable mixed precision for significant speedup

if mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f'Mixed precision enabled: {policy}')

# Paths - load from json config
with open("training_paths.json", "r") as f:
    config = json.load(f)
tfrecord_dir = config["tfrecord_dir"]
allele_seq_path = config["allele_seq_path"]
embedding_key_path = config["embedding_key_path"]
embedding_table_path = config["embedding_table_path"]
val_parquet_path = config["val_parquet_path"]
train_parquet_path = config["train_parquet_path"]
bench1_parquet_path = config["bench1_parquet_path"]
bench2_parquet_path = config["bench2_parquet_path"]
bench3_parquet_path = config["bench3_parquet_path"]
print("Paths loaded.")

## --- Globals ---
MHC_CLASS = 1

## --- Constant lookup table for on-the-fly feature creation ---
_blosum_vectors = [np.asarray(BLOSUM62[aa], dtype=np.float32) for aa in AA_BLOSUM]
vector_len = _blosum_vectors[0].shape[0]
_blosum_vectors.append(np.full((vector_len,), PAD_VALUE, dtype=np.float32))
BLOSUM62_TABLE = tf.constant(np.stack(_blosum_vectors, axis=0), dtype=tf.float32)
metadata = load_metadata(tfrecord_dir)

# Load EMB_DB_p for DataGenerator
print("\nLoading MHC embedding lookup table for DataGenerator...")
EMB_DB_p = load_embedding_db(embedding_table_path)

lookup_path = os.path.join(tfrecord_dir, "train_mhc_embedding_lookup.npz")
MHC_EMBEDDING_TABLE_RAW = load_embedding_table(lookup_path, metadata)
print(f"Loaded raw MHC embedding table: {MHC_EMBEDDING_TABLE_RAW.shape}")

# Normalize the entire lookup table before starting the training.
print("Normalizing the MHC embedding lookup table...")
MHC_EMBEDDING_TABLE = normalize_embedding_tf(MHC_EMBEDDING_TABLE_RAW, method=None)
print(f"Normalized MHC embedding table created.")
print(f"Min value in normalized table: {tf.reduce_min(MHC_EMBEDDING_TABLE):.2f}")
print(f"Max value in normalized table: {tf.reduce_max(MHC_EMBEDDING_TABLE):.2f}")

# Extract all the important values from metadata
MAX_PEP_LEN = metadata['MAX_PEP_LEN']
MAX_MHC_LEN = metadata['MAX_MHC_LEN']
ESM_DIM = metadata['ESM_DIM']
train_samples = metadata['train_samples']

# Load sequence and embedding mappings
print("\nLoading sequence and embedding mappings...")
print(f"Allele sequence file: {allele_seq_path}")
print(f"Embedding key file: {embedding_key_path}")

# Check if files exist
if not os.path.exists(allele_seq_path):
    print(f"Error: Allele sequence file not found at {allele_seq_path}")
    raise FileNotFoundError(f"Allele sequence file not found: {allele_seq_path}")

if not os.path.exists(embedding_key_path):
    print(f"Error: Embedding key file not found at {embedding_key_path}")
    raise FileNotFoundError(f"Embedding key file not found: {embedding_key_path}")

# Load with error handling
try:
    seq_df = pd.read_csv(allele_seq_path, index_col="allele")
    seq_map = {clean_key(k): v for k, v in seq_df["mhc_sequence"].to_dict().items()}
except Exception as e:
    print(f"Error loading sequence file: {e}")
    print(f"Available columns in {allele_seq_path}:")
    temp_df = pd.read_csv(allele_seq_path)
    print(temp_df.columns.tolist())
    raise e

try:
    embed_df = pd.read_csv(embedding_key_path, index_col="key")
    embed_map = embed_df["mhc_sequence"].to_dict()
except Exception as e:
    print(f"Error loading embedding key file: {e}")
    print(f"Available columns in {embedding_key_path}:")
    temp_df = pd.read_csv(embedding_key_path)
    print(temp_df.columns.tolist())
    raise e

#### ──────────────────────────────────────────────────────────────────────

#### Utils for training
#### ----------------------------------------------------------------------
class OptimizedDataGenerator(keras.utils.Sequence):
    """Optimized data generator for training with dynamic masking support."""

    def __init__(self, df, seq_map, embed_map, max_pep_len, max_mhc_len, batch_size, apply_masking=True,
                 normalization_method=None, class_weights_dict=None):
        super().__init__()
        self.df = df
        self.seq_map = seq_map
        self.embed_map = embed_map
        self.max_pep_len = max_pep_len
        self.max_mhc_len = max_mhc_len
        self.batch_size = batch_size
        self.apply_masking = apply_masking
        self.long_mer_arr = df['long_mer'].to_numpy()
        self.emb_key_arr = df['_emb_key'].to_numpy()
        self.cleaned_key_arr = df['_cleaned_key'].to_numpy()
        self.mhc_seq_arr = df['_mhc_seq'].to_numpy()
        self.label_arr = df['assigned_label'].to_numpy()
        self.indices = np.arange(len(df))
        self.normalization_method = normalization_method
        self.class_weights_dict = class_weights_dict
        if self.class_weights_dict is None:
            self.class_weights_dict = {0: 1.0, 1: 1.0}

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        start_idx, end_idx = index * self.batch_size, min((index + 1) * self.batch_size, len(self.indices))
        return self._generate_batch(self.indices[start_idx:end_idx])

    def _get_embedding(self, emb_key, cleaned_key):
        """Get embedding and apply robust normalization."""
        if MHC_CLASS == 2:
            parts = cleaned_key.split('_')
            if len(parts) >= 2:
                k1, k2 = get_embed_key(clean_key(parts[0]), self.embed_map), get_embed_key(clean_key(parts[1]),
                                                                                           self.embed_map)
                emb = np.concatenate([EMB_DB_p[k1], EMB_DB_p[k2]], axis=0)
            else:
                emb = EMB_DB_p[emb_key]
        else:
            try:
                emb = EMB_DB_p[emb_key]
            except KeyError:
                raise KeyError(f"embedding not found for emb_key {emb_key}. for cleaned_key: {cleaned_key}")

        # Apply chosen normalization method
        return self._normalize_embedding(emb)

    def _normalize_embedding(self, emb):
        """Apply robust normalization to handle extreme values."""
        if self.normalization_method == "min_max_norm":
            return min_max_norm(emb)
        elif self.normalization_method == "log_norm_zscore":
            return log_norm_zscore(emb)
        elif self.normalization_method == "clip_norm1000":
            emb_norm = np.clip(emb, -1000, 1000)
            return 20 * (emb_norm - (-1000)) / (1000 - (-1000)) - 10
        elif self.normalization_method == "robust_zscore":
            # Per-sample normalization (best for ESM embeddings)
            mean = emb.mean()
            std = emb.std()
            emb_norm = (emb - mean) / (std + 1e-8)
            # Clip outliers after normalization
            emb_norm = np.clip(emb_norm, -5, 5)
            return emb_norm
        else:
            return emb  # No normalization


    def _generate_batch(self, batch_indices):
        n = len(batch_indices)
        data = {"pep_blossom62": np.zeros((n, self.max_pep_len, 23), np.float32),
                "pep_mask": np.full((n, self.max_pep_len), PAD_TOKEN, dtype=np.float32),
                "mhc_emb": np.zeros((n, self.max_mhc_len, ESM_DIM), np.float32),
                "mhc_mask": np.full((n, self.max_mhc_len), PAD_TOKEN, dtype=np.float32),
                "pep_ohe_target": np.zeros((n, self.max_pep_len, 21), np.float32),
                "mhc_ohe_target": np.zeros((n, self.max_mhc_len, 21), np.float32),
                "labels": np.zeros((n, 1), np.float32),
                "sample_weights": np.zeros((n, 1), np.float32)}

        for i, master_idx in enumerate(batch_indices):
            pep_seq, emb_key, cleaned_key, mhc_seq = (self.long_mer_arr[master_idx].upper(),
                                                      self.emb_key_arr[master_idx],
                                                      self.cleaned_key_arr[master_idx],
                                                      self.mhc_seq_arr[master_idx])
            pep_len = len(pep_seq)
            data["pep_blossom62"][i] = seq_to_blossom62(pep_seq, max_seq_len=self.max_pep_len)
            #normalise pep
            #normalize only valid peptide positions to avoid touching padding
            # _pep_slice = data["pep_blossom62"][i, :pep_len]
            # pep_norm = 8 * (_pep_slice - (-4)) / (11 - (-4)) - 4 # MIN Max normalization of BLOSSUM62
            # _norm = np.linalg.norm(_pep_slice, axis=-1, keepdims=True)
            # _norm = np.maximum(_norm, 1e-12)
            # data["pep_blossom62"][i, :pep_len] = pep_norm
            data["pep_ohe_target"][i] = seq_to_onehot(pep_seq, max_seq_len=self.max_pep_len)
            data["pep_mask"][i, :pep_len] = NORM_TOKEN
            emb = self._get_embedding(emb_key, cleaned_key)
            L = emb.shape[0]
            data["mhc_emb"][i, :L] = emb
            data["mhc_ohe_target"][i] = seq_to_onehot(mhc_seq, max_seq_len=self.max_mhc_len)
            # data["mhc_mask"][i, ~np.all(data["mhc_ohe_target"][i] == PAD_INDEX_OHE, axis=-1)] = NORM_TOKEN
            is_padding = np.all(data["mhc_ohe_target"][i, :] == 0, axis=-1)
            data["mhc_mask"][i, ~is_padding] = NORM_TOKEN

            data["labels"][i, 0] = float(self.label_arr[master_idx])
            data["sample_weights"][i, 0] = self.class_weights_dict[int(self.label_arr[master_idx])]

        # Convert to tensors
        tensor_data = {k: tf.convert_to_tensor(v) for k, v in data.items()}

        # Apply dynamic masking if enabled
        if self.apply_masking:
            tensor_data = apply_dynamic_masking(tensor_data, emd_mask_d2=True)

        return tensor_data

def preprocess_df(df, seq_map, embed_map, workers=None, chunks=None):
    """
    Multiprocessed preprocessing. Splits df into chunks, processes in parallel, and
    preserves original order. Backward compatible with existing calls.
    """
    if workers is None:
        workers = max(1, os.cpu_count() // 2 or 1)
    if chunks is None:
        chunks = max(1, min(workers * 4, len(df)))  # more chunks than workers for load balancing

    parts = np.array_split(df, chunks)
    args = [(part, seq_map, embed_map, MHC_CLASS) for part in parts]

    if workers == 1 or len(parts) == 1:
        processed = [_preprocess_df_chunk(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            processed = list(ex.map(_preprocess_df_chunk, args))

    out = pd.concat(processed, axis=0)
    out = out.loc[df.index]  # preserve original row order
    return out


def _parse_tf_example(example_proto, mhc_embedding_table, tfrecord_dir):
    """
    Parse TFRecord example
    Fixes the typo (tf.record_dir -> tfrecord_dir) and correctly handles
    padding and one-hot targets from the start.
    """
    # THIS IS THE FIX: Changed tf.record_dir to the correct variable name tfrecord_dir
    metadata = load_metadata(tfrecord_dir)
    MAX_PEP_LEN = metadata['MAX_PEP_LEN']
    MAX_MHC_LEN = metadata['MAX_MHC_LEN']

    feature_description = {
        'pep_indices': tf.io.FixedLenFeature([MAX_PEP_LEN], tf.int64),
        'pep_ohe_indices': tf.io.FixedLenFeature([MAX_PEP_LEN], tf.int64),
        'mhc_ohe_indices': tf.io.FixedLenFeature([MAX_MHC_LEN], tf.int64),
        'embedding_id': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    parsed = tf.io.parse_single_example(example_proto, feature_description)
    pep_indices = parsed['pep_indices']
    pep_ohe_indices = parsed['pep_ohe_indices']
    mhc_ohe_indices = parsed['mhc_ohe_indices']

    # --- Create Feature Tensors ---
    embedding_id = tf.cast(parsed['embedding_id'], tf.int32)
    mhc_emb = tf.gather(mhc_embedding_table, embedding_id)
    pep_blossom62_input = tf.gather(BLOSUM62_TABLE, tf.cast(pep_indices, tf.int32))

    # --- Create One-Hot Targets and Fix UNK Column ---
    # Create one-hot targets
    pep_ohe_target = tf.one_hot(tf.cast(pep_ohe_indices, tf.int32), depth=21, dtype=tf.float32)
    mhc_ohe_target = tf.one_hot(tf.cast(mhc_ohe_indices, tf.int32), depth=21, dtype=tf.float32)
    # Zero out padding positions to match DataGenerator behavior
    # Where pep_ohe_indices == PAD_INDEX_OHE (20), set entire row to zeros
    pep_is_padding = tf.equal(pep_ohe_indices, tf.constant(PAD_INDEX_OHE, dtype=pep_ohe_indices.dtype))
    pep_ohe_target = tf.where(
        tf.expand_dims(pep_is_padding, axis=-1),  # Shape: (max_pep_len, 1)
        tf.zeros_like(pep_ohe_target),  # All zeros
        pep_ohe_target  # Keep original
    )

    mhc_is_padding = tf.equal(mhc_ohe_indices, tf.constant(PAD_INDEX_OHE, dtype=mhc_ohe_indices.dtype))
    mhc_ohe_target = tf.where(
        tf.expand_dims(mhc_is_padding, axis=-1),  # Shape: (max_mhc_len, 1)
        tf.zeros_like(mhc_ohe_target),  # All zeros
        mhc_ohe_target  # Keep original
    )

    # Zero out padding positions in pep_blossom62 to match DataGenerator
    pep_blossom62_input = tf.where(
        tf.expand_dims(pep_is_padding, axis=-1),  # Shape: (max_pep_len, 1)
        tf.zeros_like(pep_blossom62_input),       # All zeros for padding
        pep_blossom62_input                       # Keep original for valid
    )

    pep_mask = tf.where(pep_is_padding, PAD_TOKEN, NORM_TOKEN)
    mhc_mask = tf.where(mhc_is_padding, PAD_TOKEN, NORM_TOKEN)

    labels = tf.cast(parsed['label'], tf.float32)
    labels = tf.expand_dims(labels, axis=-1)

    return {
        "pep_blossom62": tf.cast(pep_blossom62_input, tf.float32),
        "pep_mask": tf.cast(pep_mask, tf.float32),
        "mhc_emb": tf.cast(mhc_emb, tf.float32),
        "mhc_mask": tf.cast(mhc_mask, tf.float32),
        "pep_ohe_target": pep_ohe_target,
        "mhc_ohe_target": mhc_ohe_target,
        "labels": labels
    }

def create_dataset_from_files(file_list, batch_size, shuffle=True, apply_masking=False):
    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
        cycle_length=len(file_list) if shuffle else 1,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(
        lambda x: _parse_tf_example(x, MHC_EMBEDDING_TABLE, tfrecord_dir),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if apply_masking:
        dataset = dataset.map(apply_dynamic_masking, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
# ──────────────────────────────────────────────────────────────────────


#### Training Configuration
#### ----------------------------------------------------------------------

def create_optimized_training_setup(run_config=None):
    """Create optimized model and training components"""
    if run_config is None:
        raise ValueError("RUN_CONFIG must be provided.")
    model = pmbind(
        max_pep_len=MAX_PEP_LEN,
        max_mhc_len=MAX_MHC_LEN,
        emb_dim=run_config["EMBED_DIM"],
        heads=run_config["HEADS"],
        noise_std=run_config["NOISE_STD"],
        drop_out_rate=run_config["DROPOUT_RATE"],
        l2_reg=run_config["L2_REG"],
        ESM_dim=ESM_DIM,
    )

    decay_steps = (metadata['train_samples'] // run_config["BATCH_SIZE"]) * 2
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=run_config["LEARNING_RATE"],
        first_decay_steps=decay_steps,
        alpha=0.1,
        t_mul=1.2,
        m_mul=0.9
    )
    base_optimizer = keras.optimizers.Lion(learning_rate=lr_schedule)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer) if mixed_precision else base_optimizer
    binary_loss_fn = tf.keras.losses.BinaryCrossentropy()
    return model, optimizer, binary_loss_fn

def train_step(batch_data, model, optimizer, binary_loss_fn, metrics, class_weights_dict, run_config=None):
    """Standard training step"""
    if run_config is None:
        raise ValueError("RUN_CONFIG must be provided.")

    x_batch_list = [batch_data['pep_blossom62'], batch_data['pep_mask'],
                    batch_data['mhc_emb'], batch_data['mhc_mask'],
                    batch_data['pep_ohe_target'], batch_data['mhc_ohe_target']]

    with tf.GradientTape() as tape:
        predictions = model(x_batch_list, training=True)

        # Use ANL_CE loss for reconstruction tasks
        pep_loss = masked_categorical_crossentropy(predictions['pep_ytrue_ypred'], batch_data['pep_mask'])
        mhc_loss = masked_categorical_crossentropy(predictions['mhc_ytrue_ypred'], batch_data['mhc_mask'])
        label_int = tf.cast(batch_data['labels'], tf.int32)
        sample_w = tf.where(tf.equal(label_int, 0), class_weights_dict[0], class_weights_dict[1])
        class_loss = binary_loss_fn(batch_data['labels'], predictions['cls_ypred'], sample_weight=tf.cast(sample_w, tf.float32))

        # Decay reconstruction loss over time (focus more on classification)
        recon_decay_w = (0.5 ** (run_config["EPOCHS"] / 5))  # Halve every 5 epochs
        total_loss = recon_decay_w * (run_config["PEP_LOSS_WEIGHT"] * pep_loss + run_config["MHC_LOSS_WEIGHT"] * mhc_loss) + run_config["CLS_LOSS_WEIGHT"] * class_loss

        # Use proper LossScaleOptimizer methods for mixed precision
        if mixed_precision:
            scaled_loss = optimizer.scale_loss(total_loss)
            grads = tape.gradient(scaled_loss, model.trainable_variables)
            # Gradient unscaling is handled automatically by apply_gradients
        else:
            grads = tape.gradient(total_loss, model.trainable_variables)

        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Update metrics
    metrics['loss'].update_state(total_loss)
    metrics['pep_recon_loss'].update_state(pep_loss)
    metrics['mhc_recon_loss'].update_state(mhc_loss)
    metrics['class_loss'].update_state(class_loss)
    metrics['auc'].update_state(batch_data['labels'], predictions['cls_ypred'])
    metrics['acc'].update_state(batch_data['labels'], predictions['cls_ypred'])
    metrics['mcc'].update_state(batch_data['labels'], predictions['cls_ypred'])

    return total_loss, pep_loss, mhc_loss, class_loss, metrics


def val_step_optimized(batch_data, model, binary_loss_fn, metrics, run_config=None):
    """Validation step with ANL_CE loss"""
    if run_config is None:
        raise ValueError("RUN_CONFIG must be provided.")
    x_batch_list = [batch_data['pep_blossom62'], batch_data['pep_mask'],
                    batch_data['mhc_emb'], batch_data['mhc_mask'],
                    batch_data['pep_ohe_target'], batch_data['mhc_ohe_target']]

    predictions = model(x_batch_list, training=False)

    pep_loss = masked_categorical_crossentropy(predictions['pep_ytrue_ypred'], batch_data['pep_mask'])
    mhc_loss = masked_categorical_crossentropy(predictions['mhc_ytrue_ypred'], batch_data['mhc_mask'])
    class_loss = binary_loss_fn(batch_data['labels'], predictions['cls_ypred'])

    total_loss = run_config["PEP_LOSS_WEIGHT"] * pep_loss + run_config["MHC_LOSS_WEIGHT"] * mhc_loss + run_config["CLS_LOSS_WEIGHT"] * class_loss

    # Update metrics
    metrics['loss'].update_state(total_loss)
    metrics['pep_recon_loss'].update_state(pep_loss)
    metrics['mhc_recon_loss'].update_state(mhc_loss)
    metrics['class_loss'].update_state(class_loss)
    metrics['auc'].update_state(batch_data['labels'], predictions['cls_ypred'])
    metrics['acc'].update_state(batch_data['labels'], predictions['cls_ypred'])
    metrics['mcc'].update_state(batch_data['labels'], predictions['cls_ypred'])

    return total_loss, pep_loss, mhc_loss, class_loss, predictions
#### ──────────────────────────────────────────────────────────────────────


### Load and preprocess datasets
### ----------------------------------------------------------------------
def create_datasets(run_config=None):
    if run_config is None:
        raise ValueError("RUN_CONFIG must be provided.")
    print("\nCreating datasets...")
    print("creating train dataset...")
    train_df = preprocess_df(pd.read_parquet(train_parquet_path), seq_map, embed_map)
    print(train_df["assigned_label"].value_counts())

    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.array([0, 1]),
        y=train_df['assigned_label'].values
    )
    class_weights_dict = {0: class_weights_array[0], 1: class_weights_array[1]}

    print("creating val dataset...")
    val_df = preprocess_df(pd.read_parquet(val_parquet_path), seq_map, embed_map)
    val_generator = OptimizedDataGenerator(
        df=val_df,  # Sample for demo
        seq_map=seq_map,
        embed_map=embed_map,
        max_pep_len=MAX_PEP_LEN,
        max_mhc_len=MAX_MHC_LEN,
        batch_size=128,
        apply_masking=False  # No masking for validation
    )
    # Bench datasets for live performance tracking during training
    # IEDB bench1
    print("creating bench1 dataset...")
    bench_df1 = preprocess_df(pd.read_parquet(bench1_parquet_path), seq_map, embed_map)
    bench_generator1 = OptimizedDataGenerator(
        df=bench_df1,
        seq_map=seq_map,
        embed_map=embed_map,
        max_pep_len=MAX_PEP_LEN,
        max_mhc_len=MAX_MHC_LEN,
        batch_size=128,
        apply_masking=False
    )
    # IEDB bench2
    print("creating bench2 dataset...")
    bench_df2 = preprocess_df(pd.read_parquet(bench2_parquet_path), seq_map, embed_map)
    bench_generator2 = OptimizedDataGenerator(
        df=bench_df2,
        seq_map=seq_map,
        embed_map=embed_map,
        max_pep_len=MAX_PEP_LEN,
        max_mhc_len=MAX_MHC_LEN,
        batch_size=128,
        apply_masking=False
    )
    # independent test set
    print("creating bench3 dataset...")
    bench_df3 = preprocess_df(pd.read_parquet(bench3_parquet_path), seq_map, embed_map)
    bench_generator3 = OptimizedDataGenerator(
        df=bench_df3,
        seq_map=seq_map,
        embed_map=embed_map,
        max_pep_len=MAX_PEP_LEN,
        max_mhc_len=MAX_MHC_LEN,
        batch_size=128,
        apply_masking=False
    )

    # Create datasets
    train_files = [os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.startswith('train_') and f.endswith('.tfrecord')]
    train_dataset_tf = create_dataset_from_files(train_files, run_config["BATCH_SIZE"], shuffle=True, apply_masking=True)
    print(f"Training TFRecord files: {train_files}")

    return train_dataset_tf, val_df, val_generator, class_weights_dict, (bench_generator1, bench_generator2, bench_generator3)
### ──────────────────────────────────────────────────────────────────────


### Main training loop
### ----------------------------------------------------------------------
def train(train_dataset_tf, val_df, val_generator=None, bench_generators=None, MODEL_SAVE_PATH="best_model.keras",
          run_config=None, epochs=20, batch_size=32, patience=5, class_weights_dict=None):
    if run_config is None:
        raise ValueError("RUN_CONFIG must be provided.")
    if class_weights_dict is None:
        class_weights_dict = {0: 1.0, 1: 1.0}

    print("\nStarting training...")

    def _build_metrics():
        # Create a fresh metrics dict each call (metrics objects must not be shared)
        return {
            'auc': tf.keras.metrics.AUC(),
            'acc': tf.metrics.BinaryAccuracy(),
            'mcc': BinaryMCC(),
            'loss': tf.keras.metrics.Mean(name='loss'),
            'pep_recon_loss': tf.keras.metrics.Mean(name='pep_recon_loss'),
            'mhc_recon_loss': tf.keras.metrics.Mean(name='mhc_recon_loss'),
            'class_loss': tf.keras.metrics.Mean(name='class_loss'),
        }

    train_metrics = _build_metrics()
    val_metrics = _build_metrics()
    bench1_metrics = _build_metrics()
    bench2_metrics = _build_metrics()
    bench3_metrics = _build_metrics()

    # Build unified history dictionary
    history = {}
    for prefix, mset in [
        ('train', train_metrics),
        ('val', val_metrics),
        ('bench1', bench1_metrics),
        ('bench2', bench2_metrics),
        ('bench3', bench3_metrics),
    ]:
        for k in mset.keys():
            key_name = k if prefix == 'train' else f"{prefix}_{k}"
            history[key_name] = []
    model, optimizer, binary_loss_fn, = create_optimized_training_setup(run_config)

    # model summary
    model.summary()

    best_val_auc = 0.0
    patience_counter = 0

    train_steps = metadata['train_samples'] // batch_size
    val_steps = len(val_df)//batch_size
    print(f"Train steps per epoch: {train_steps}, Validation steps: {val_steps}")

    print(f"\nCreated TFRecord datasets:")
    print(f"Training steps per epoch: {train_steps}")
    print(f"Validation steps per epoch: {val_steps}")
    print(f"Batch size: {batch_size}")

    print("\nStarting training with TFRecord pipeline and dynamic masking...")
    for epoch in range(epochs):
        for metric in train_metrics.values(): metric.reset_state()
        for metric in val_metrics.values(): metric.reset_state()

        # Training
        pbar = tqdm(train_dataset_tf, total=train_steps, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_data in pbar:
            total_loss, pep_loss, mhc_loss, class_loss, metrics = train_step(
                batch_data, model, optimizer, binary_loss_fn, train_metrics, run_config=run_config,
                class_weights_dict=class_weights_dict
            )
            pbar.set_postfix({
                "Loss": f"{train_metrics['loss'].result():.4f}", "AUC": f"{train_metrics['auc'].result():.4f}",
                "ACC": f"{train_metrics['acc'].result():.4f}", "MCC": f"{train_metrics['mcc'].result():.4f}",
            })

        # Validation
        pbar_val = tqdm(range(val_steps), desc=f"Epoch {epoch + 1}/{epochs} - Val", total=val_steps)
        for batch_idx in pbar_val:
            batch_data = val_generator[batch_idx]
            val_total_loss, val_pep_loss, val_mhc_loss, val_class_loss, _ = val_step_optimized(
                batch_data, model, binary_loss_fn, val_metrics, run_config=run_config
            )
            pbar_val.set_postfix({
                "Val_Loss": f"{val_metrics['loss'].result():.4f}", "Val_AUC": f"{val_metrics['auc'].result():.4f}",
                "Val_ACC": f"{val_metrics['acc'].result():.4f}", "Val_MCC": f"{val_metrics['mcc'].result():.4f}",
            })

        # Benchmark evaluation (after validation completes)
        if bench_generators:
            bench_names = ['bench1', 'bench2', 'bench3']
            bench_metrics_list = [bench1_metrics, bench2_metrics, bench3_metrics]

            for bench_name, bench_gen, bench_metrics in zip(bench_names, bench_generators, bench_metrics_list):
                for metric in bench_metrics.values():
                    metric.reset_state()

                pbar_bench = tqdm(range(len(bench_gen)), desc=f"Epoch {epoch + 1}/{epochs} - {bench_name}", total=len(bench_gen))
                for batch_idx in range(len(bench_gen)):
                    batch_data = bench_gen[batch_idx]
                    val_step_optimized(batch_data, model, binary_loss_fn, bench_metrics, run_config=run_config)
                    pbar_bench.set_postfix({
                        f"{bench_name}_Loss": f"{bench_metrics['loss'].result():.4f}",
                        f"{bench_name}_AUC": f"{bench_metrics['auc'].result():.4f}",
                        f"{bench_name}_ACC": f"{bench_metrics['acc'].result():.4f}",
                        f"{bench_name}_MCC": f"{bench_metrics['mcc'].result():.4f}",
                    })
                bench_results = {key: value.result().numpy() for key, value in bench_metrics.items()}
                for key, value in bench_results.items():
                    history[f"{bench_name}_{key}"].append(value)

        train_results = {key: value.result().numpy() for key, value in train_metrics.items()}
        val_results = {key: value.result().numpy() for key, value in val_metrics.items()}
        for key, value in train_results.items(): history[key].append(value)
        for key, value in val_results.items(): history[f"val_{key}"].append(value)

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Loss: {train_results['loss']:.4f} - AUC: {train_results['auc']:.4f} - ACC: {train_results['acc']:.4f} - MCC: {train_results['mcc']:.4f} - "
              f"Val Loss: {val_results['loss']:.4f} - Val AUC: {val_results['auc']:.4f} - Val ACC: {val_results['acc']:.4f} - Val MCC: {val_results['mcc']:.4f}")

        current_val_auc = val_results['auc']
        if current_val_auc > best_val_auc:
            print(f"Validation AUC improved from {best_val_auc:.4f} to {current_val_auc:.4f}. Saving model...")
            best_val_auc = current_val_auc
            patience_counter = 0
            model.save(MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            print(f"Validation AUC did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered. Training has been halted.")
            break
    print("Training finished!")

    # Save final model and history
    model.save(MODEL_SAVE_PATH.replace(".keras", "_final.keras"))
    # Ensure history is saved in the same directory as the final model
    model_dir = os.path.dirname(MODEL_SAVE_PATH)
    if not model_dir:
        model_dir = "."
    os.makedirs(model_dir, exist_ok=True)
    history_path = os.path.join(model_dir, os.path.splitext(os.path.basename(MODEL_SAVE_PATH))[0] + "_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    visualize_training_history(history, os.path.dirname(MODEL_SAVE_PATH))


# save std.out to file
def init_run_logger(log_dir, run_name):
    """Redirect stdout & stderr to a tee log file (still shows in console)."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.log")

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                    s.flush()
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    log_file = open(log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    print(f"[logger] Streaming stdout/stderr to {log_path}")
    return log_file


# ──────────────────────────────────────────────────────────────────────
# Main Execution
# ----------------------------------------------------------------------
def main(args):
    """Main function to run the training pipeline."""
    RUN_CONFIG = {
        "MHC_CLASS": 1, "EPOCHS": 120, "BATCH_SIZE": 1024, "LEARNING_RATE": 1e-3, "PATIENCE": 15,
        "EMBED_DIM": 32, "HEADS": 8, "NOISE_STD": 0.1, "L2_REG": 0.003, "EMBEDDING_NORM": "robust_zscore", # robust_zscore, clip_norm1000, None
        "CLS_LOSS_WEIGHT": 1.0, "PEP_LOSS_WEIGHT": 0.01, "MHC_LOSS_WEIGHT": 0.01, "DROPOUT_RATE": 0.18,
        "description": "tfrecord base"
    }

    base_output_folder = "../results/PMBind_runs/"
    run_id_base = 0

    log_file = init_run_logger(base_output_folder, f"run_{run_id_base + args.fold}_mhc{RUN_CONFIG['MHC_CLASS']}_seed{args.fold}")
    seed_to_run = args.fold  # Get fold from args
    run_id = run_id_base + seed_to_run
    run_name = f"run_{run_id}_mhc{RUN_CONFIG['MHC_CLASS']}_seed{seed_to_run}"
    out_dir = os.path.join(base_output_folder, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Starting run: {run_name}\nOutput directory: {out_dir}")

    # save run config
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(RUN_CONFIG, f, indent=4)

    if not os.path.exists(tfrecord_dir) or not os.path.exists(os.path.join(tfrecord_dir, 'metadata.json')):
        print(f"Error: TFRecord directory not found or is incomplete: {tfrecord_dir}")
        print("Please run the `create_tfrecords.py` script first.")
        sys.exit(1)

    train_dataset_tf, val_df, val_generator, class_weights_dict, bench_generators = create_datasets(RUN_CONFIG)

    train(
        MODEL_SAVE_PATH=os.path.join(out_dir, "best_model.keras"),
        run_config=RUN_CONFIG,
        train_dataset_tf=train_dataset_tf,
        val_df=val_df,
        val_generator=val_generator,
        bench_generators=bench_generators,
        epochs=RUN_CONFIG["EPOCHS"],
        batch_size=RUN_CONFIG["BATCH_SIZE"],
        patience=RUN_CONFIG["PATIENCE"],
        class_weights_dict=class_weights_dict,
    )
    if 'log_file' in locals() and not log_file.closed:
        print("[logger] Closing log file.")
        log_file.close()


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth for each GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    try:
        tf.config.experimental.enable_tensor_float_32_execution(True)
    except:
        pass
    parser = argparse.ArgumentParser(description="Run training for specified folds.")
    parser.add_argument("--fold", type=int, required=True, help="Fold number to run (e.g., 0-4).")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to model weights (.h5 file) to resume training from.")
    parser.add_argument("--subset", type=float, default=1.0, help="Subset percentage of training data to use.")
    parser.add_argument("--ls_param", type=float, default=0.15, help="Label smoothing parameter.")
    parser.add_argument("--as_param", type=float, default=5.0, help="Asymmetric loss scaling parameter.")
    args = parser.parse_args()
    main(args)