import csv
import sys
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

# --- Import necessary utilities and model definitions ---
from utils import (NORM_TOKEN, PAD_TOKEN, PAD_VALUE, BLOSUM62,
                   AMINO_ACID_VOCAB, PAD_INDEX, AA)
from models import pmbind_multitask as pmbind
from visualizations import visualize_inference_results

# --- Globals for the high-performance tf.data pipeline ---
MHC_EMBEDDING_TABLE = None
BLOSUM62_TABLE = None
MAX_PEP_LEN = 0
MAX_MHC_LEN = 0
ESM_DIM = 0


# --- On-the-fly TFRecord Parsing Function (identical to training) ---
def _parse_tf_example(example_proto):
    feature_description = {
        'pep_indices': tf.io.FixedLenFeature([], tf.string),
        'mhc_indices': tf.io.FixedLenFeature([], tf.string),
        'embedding_id': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    pep_indices = tf.io.parse_tensor(parsed['pep_indices'], out_type=tf.int8)
    mhc_indices = tf.io.parse_tensor(parsed['mhc_indices'], out_type=tf.int8)

    embedding_id = tf.cast(parsed['embedding_id'], tf.int32)
    mhc_emb = tf.gather(MHC_EMBEDDING_TABLE, embedding_id)
    mhc_emb = tf.cast(mhc_emb, tf.float32)

    pep_blossom62_input = tf.gather(BLOSUM62_TABLE, tf.cast(pep_indices, tf.int32))

    vocab_size_ohe = len(AA)
    # Create dummy pep_ohe_target and mhc_ohe_target as zeros for inference
    pep_ohe_target = tf.zeros((tf.shape(pep_indices)[0], vocab_size_ohe), dtype=tf.float32)
    mhc_ohe_target = tf.zeros((tf.shape(mhc_indices)[0], vocab_size_ohe), dtype=tf.float32)

    pep_mask = tf.where(pep_indices == PAD_INDEX, PAD_TOKEN, NORM_TOKEN)
    mhc_mask = tf.where(tf.reduce_all(mhc_emb == PAD_VALUE, axis=-1), PAD_TOKEN, NORM_TOKEN)

    labels = tf.cast(parsed['label'], tf.int32)
    labels = tf.expand_dims(labels, axis=-1)

    return {
        "pep_blossom62": pep_blossom62_input, "pep_mask": pep_mask,
        "mhc_emb": mhc_emb, "mhc_mask": mhc_mask,
        "pep_ohe_target": pep_ohe_target, "mhc_ohe_target": mhc_ohe_target,
        "labels": labels
    }


def create_inference_dataset_from_tfrecords(tfrecord_pattern, batch_size):
    """Creates a high-performance tf.data.Dataset for inference from TFRecords."""
    file_list = tf.io.gfile.glob(tfrecord_pattern)
    if not file_list:
        raise FileNotFoundError(f"No TFRecord files found for pattern: {tfrecord_pattern}")

    dataset = tf.data.TFRecordDataset(file_list, compression_type="GZIP", num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(_parse_tf_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def infer_on_dataset(model, tfrecord_dir, original_parquet_path, dataset_name, out_dir, batch_size):
    """Runs the full inference pipeline using the FAST TFRecord method."""
    global MHC_EMBEDDING_TABLE, BLOSUM62_TABLE

    print(f"\n--- Starting FAST inference on: {dataset_name} ---")
    os.makedirs(out_dir, exist_ok=True)

    # Note: The lookup table and TFRecords must correspond to the original_parquet_path
    # This assumes create_tfrecords.py was run on that parquet file.
    lookup_name = "validation_mhc_embedding_lookup.npz" if "validation" in dataset_name else "train_mhc_embedding_lookup.npz"
    lookup_path = os.path.join(tfrecord_dir, lookup_name)
    if not os.path.exists(lookup_path):
        raise FileNotFoundError(f"Required embedding lookup file not found: {lookup_path}")

    with np.load(lookup_path) as data:
        num_embeddings = len(data.files)
        table = np.zeros((num_embeddings, MAX_MHC_LEN, ESM_DIM), dtype=np.float16)
        for i in range(num_embeddings):
            table[i] = data[str(i)]
    MHC_EMBEDDING_TABLE = tf.constant(table)

    # Create the high-performance dataset
    shard_name = "validation_shard_*.tfrecord" if "validation" in dataset_name else "train_shard_*.tfrecord"
    tfrecord_pattern = os.path.join(tfrecord_dir, shard_name)
    dataset = create_inference_dataset_from_tfrecords(tfrecord_pattern, batch_size)

    # Run batched inference
    all_predictions, all_labels = [], []
    for batch in tqdm(dataset, desc=f"Predicting on {dataset_name}", file=sys.stdout):
        outputs = model(batch, training=False)
        all_predictions.append(outputs["cls_ypred"].numpy())
        all_labels.append(batch["labels"].numpy())

    all_predictions = np.concatenate(all_predictions, axis=0).squeeze()
    all_labels = np.concatenate(all_labels, axis=0).squeeze()

    # Load original data to merge predictions with metadata
    df_original = pd.read_parquet(original_parquet_path)
    # The TFRecord creation process filters out bad data, so the result might be shorter.
    df_results = df_original.iloc[:len(all_predictions)].copy()
    df_results["prediction_score"] = all_predictions
    df_results["prediction_label"] = (all_predictions >= 0.5).astype(int)

    output_path = os.path.join(out_dir, f"inference_results_{dataset_name}.csv")
    df_results.to_csv(output_path, index=False)
    print(f"✓ Inference results saved to {output_path}")

    # Visualization and summary
    if 'assigned_label' in df_results.columns:
        visualize_inference_results(df_results, all_labels, all_predictions, out_dir, dataset_name)

        auc = roc_auc_score(all_labels, all_predictions)
        ap = average_precision_score(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, df_results["prediction_label"])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        summary_path = os.path.join(os.path.dirname(out_dir.rstrip('/')), "inference_summary.csv")
        is_new_file = not os.path.exists(summary_path)
        with open(summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if is_new_file:
                writer.writerow(['dataset', 'num_samples', 'AUC', 'AP', 'TP', 'TN', 'FP', 'FN'])
            writer.writerow([dataset_name, len(df_results), f"{auc:.4f}", f"{ap:.4f}", tp, tn, fp, fn])
        print(f"✓ Summary for {dataset_name} appended to {summary_path}")


def run_all_inferences(run_dir, config, fold):
    """Orchestrates high-performance inference on all datasets for a given model run."""
    global MAX_PEP_LEN, MAX_MHC_LEN, ESM_DIM, BLOSUM62_TABLE

    print(f"\n{'=' * 80}\nStarting High-Performance Inference for Fold {fold}\n{'=' * 80}")

    # Load metadata from the TRAINING TFRecord directory, as this defines the model shape
    train_tfrecord_dir = f"../data/tfrecords/fold_{fold:02d}"
    with open(os.path.join(train_tfrecord_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    MAX_PEP_LEN, MAX_MHC_LEN, ESM_DIM = metadata['MAX_PEP_LEN'], metadata['MAX_MHC_LEN'], metadata['ESM_DIM']

    # Initialize the BLOSUM62 lookup table once
    _blosum_vectors = [BLOSUM62[aa] for aa in AMINO_ACID_VOCAB]
    _blosum_vectors.append([PAD_VALUE] * len(_blosum_vectors[0]))
    BLOSUM62_TABLE = tf.constant(np.array(_blosum_vectors), dtype=tf.float32)

    # Load and build the trained model
    model = pmbind(max_pep_len=MAX_PEP_LEN, max_mhc_len=MAX_MHC_LEN, emb_dim=config['EMBED_DIM'],
                   heads=config['HEADS'], noise_std=0.0, latent_dim=config['EMBED_DIM'] * 2,
                   ESM_dim=ESM_DIM, drop_out_rate=0.0)

    blosum_dim = len(next(iter(BLOSUM62.values())))
    dummy_input = {
        "pep_blossom62": tf.zeros((1, MAX_PEP_LEN, blosum_dim), dtype=tf.float32),
        "pep_mask": tf.zeros((1, MAX_PEP_LEN), dtype=tf.float32),
        "mhc_emb": tf.zeros((1, MAX_MHC_LEN, ESM_DIM), dtype=tf.float32),
        "mhc_mask": tf.zeros((1, MAX_MHC_LEN), dtype=tf.float32),
    }
    model(dummy_input)

    model_weights_path = os.path.join(run_dir, "best_model.weights.h5")
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights not found at {model_weights_path}.")
    model.load_weights(model_weights_path)
    print("✓ Trained model loaded successfully.")

    # --- Define all datasets to be processed ---
    # This dictionary maps a dataset name to its TFRecord directory and original Parquet path
    mhc_class = config['MHC_CLASS']
    parquet_base_dir = f"../data/cross_validation_dataset/mhc{mhc_class}"
    tfrecord_base_dir = f"../data/tfrecords"

    datasets = {
        "validation": {
            "tfrecord_dir": f"{tfrecord_base_dir}/fold_{fold:02d}",
            "parquet_path": f"{parquet_base_dir}/cv_folds/fold_{fold:02d}_val.parquet"
        },
        "test": {
            "tfrecord_dir": f"{tfrecord_base_dir}/test_set",  # Assumes you created this
            "parquet_path": f"{parquet_base_dir}/test_set_rarest_alleles.parquet"
        },
    }

    # --- Loop and run inference on each dataset ---
    for name, paths in datasets.items():
        if os.path.exists(paths["tfrecord_dir"]) and os.path.exists(paths["parquet_path"]):
            infer_out_dir = os.path.join(run_dir, f"inference_{name}")
            infer_on_dataset(model, paths["tfrecord_dir"], paths["parquet_path"], name, infer_out_dir,
                             config['BATCH_SIZE'])
        else:
            print(f"Warning: Artifacts for '{name}' not found, skipping.")
            print(f"  - Searched for TFRecords in: {paths['tfrecord_dir']}")
            print(f"  - Searched for Parquet file: {paths['parquet_path']}")
            print(f"  - Please run create_tfrecords.py on this dataset first.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py <path_to_run_directory>")
        sys.exit(1)

    run_dir = sys.argv[1]

    fold_match = re.search(r'fold(\d+)', run_dir)
    if not fold_match:
        print(f"Error: Could not automatically determine fold number from run directory path: {run_dir}")
        sys.exit(1)

    fold = int(fold_match.group(1))

    config_path = os.path.join(run_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    run_all_inferences(run_dir, config, fold)