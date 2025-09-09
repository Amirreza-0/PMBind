import numpy as np
import pandas as pd
import tensorflow as tf
import pyarrow.parquet as pq
import os
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial
import json

# Local utilities for sequence and key processing
from utils import (get_embed_key, clean_key, PAD_VALUE,
                   seq_to_indices, seq_to_ohe_indices)

# --- Globals for worker processes ---
EMB_DB = None
MHC_CLASS = 1
ESM_DIM = 0
MAX_PEP_LEN = 0
MAX_MHC_LEN = 0
EMBED_MAP = None
SEQ_MAP = None


# --- TFRecord Helper Functions ---
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))): value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# --- Multiprocessing Worker Setup and Functions ---
def init_worker(npz_path, seq_map_data, embed_map_data, mhc_class, esm_dim, max_pep, max_mhc):
    """Initializes globals in each worker process for efficient data access."""
    global EMB_DB, SEQ_MAP, EMBED_MAP, MHC_CLASS, ESM_DIM, MAX_PEP_LEN, MAX_MHC_LEN
    EMB_DB = np.load(npz_path, mmap_mode="r")
    SEQ_MAP = seq_map_data
    EMBED_MAP = embed_map_data
    MHC_CLASS = mhc_class
    ESM_DIM = esm_dim
    MAX_PEP_LEN = max_pep
    MAX_MHC_LEN = max_mhc


def get_embedding_for_worker(emb_key, cleaned_key):
    """Helper for workers to retrieve embeddings, including Class II logic."""
    # This function now assumes emb_key is never None because we filter upstream.
    if MHC_CLASS == 2:
        key_parts = cleaned_key.split('_')
        if len(key_parts) >= 2:
            embd_key1 = get_embed_key(key_parts[0], EMBED_MAP)
            embd_key2 = get_embed_key(key_parts[1], EMBED_MAP)
            if embd_key1 and embd_key2 and embd_key1 in EMB_DB and embd_key2 in EMB_DB:
                emb1, emb2 = EMB_DB[embd_key1], EMB_DB[embd_key2]
                return np.concatenate([emb1, emb2], axis=0)
    if emb_key in EMB_DB:
        return EMB_DB[emb_key]
    # Fallback, though this should ideally not be reached
    return np.zeros((1, ESM_DIM))


def process_chunk_to_tfrecord(chunk_df, output_path):
    """Worker Function: Converts a DataFrame chunk to a TFRecord file."""
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(output_path, options=options) as writer:
        for _, row in chunk_df.iterrows():
            # Indices for BLOSUM62 input
            pep_indices = seq_to_indices(row['long_mer'], MAX_PEP_LEN)
            mhc_indices = seq_to_indices(row['_mhc_seq'], MAX_MHC_LEN)
            # Indices for the 21-dim OHE target
            pep_ohe_indices = seq_to_ohe_indices(row['long_mer'], MAX_PEP_LEN)
            mhc_ohe_indices = seq_to_ohe_indices(row['_mhc_seq'], MAX_MHC_LEN)
            feature = {
                'pep_indices': _bytes_feature(tf.io.serialize_tensor(pep_indices)),
                'pep_ohe_indices': _bytes_feature(tf.io.serialize_tensor(pep_ohe_indices)),
                'mhc_indices': _bytes_feature(tf.io.serialize_tensor(mhc_indices)),
                'mhc_ohe_indices': _bytes_feature(tf.io.serialize_tensor(mhc_ohe_indices)),
                'embedding_id': _int64_feature(int(row['embedding_id'])),
                'label': _int64_feature(int(row['assigned_label'])),
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example_proto.SerializeToString())
    return output_path


def process_chunk_preprocess(chunk, seq_map, embed_map, mhc_class):
    """
    Worker function for the initial DataFrame preprocessing.
    This is the location of the critical fix.
    """
    chunk['_cleaned_key'] = chunk.apply(lambda r: clean_key(r.get('mhc_allele', r['allele'])), axis=1)
    chunk['_emb_key'] = chunk['_cleaned_key'].apply(lambda k: get_embed_key(k, embed_map))

    # Safely handle sequence lookup, especially for None keys
    if mhc_class == 2:
        def get_mhc_seq_class2(key):
            if key is None: return ''
            key_parts = key.split('_')
            if len(key_parts) >= 2:
                key1 = get_embed_key(key_parts[0], seq_map)
                key2 = get_embed_key(key_parts[1], seq_map)
                seq1 = seq_map.get(key1, '') if key1 else ''
                seq2 = seq_map.get(key2, '') if key2 else ''
                return seq1 + seq2
            return ''

        chunk['_mhc_seq'] = chunk['_cleaned_key'].apply(get_mhc_seq_class2)
    else:
        chunk['_mhc_seq'] = chunk['_emb_key'].apply(lambda k: seq_map.get(k, '') if k is not None else '')

    # --- THE FIX ---
    # Filter out any rows where we failed to find a valid embedding key or sequence.
    # This prevents 'None' from ever entering the main DataFrame.
    return chunk.dropna(subset=['_emb_key']).query('_mhc_seq != ""')


def preprocess_df(df, seq_map, embed_map, mhc_class, num_workers=None):
    """Preprocesses the input DataFrame in parallel to find correct keys and sequences."""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() // 2)
    chunk_size = max(1, len(df) // (num_workers * 4))
    chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]
    process_func = partial(process_chunk_preprocess, seq_map=seq_map, embed_map=embed_map, mhc_class=mhc_class)
    with mp.Pool(processes=num_workers) as pool:
        processed_chunks = list(tqdm(pool.imap(process_func, chunks), total=len(chunks), desc="Preprocessing Chunks"))
    return pd.concat(processed_chunks, ignore_index=True)


def create_artifacts(df_path, output_dir, name, args):
    """Processes a full dataset to create all necessary artifacts."""
    print(f"\n--- Processing {name} dataset from {df_path} ---")

    # Initialize EMB_DB for the main process
    global EMB_DB
    if EMB_DB is None:
        EMB_DB = np.load(args.embed_npz, mmap_mode="r")

    print(f"Loading and preprocessing {name} DataFrame...")
    df_raw = pd.read_parquet(df_path)
    df_full = preprocess_df(df_raw, SEQ_MAP, EMBED_MAP, MHC_CLASS)
    print(f"✓ Preprocessing complete for {name}. Found {len(df_full)} valid rows.")

    print(f"Creating central MHC embedding lookup file for {name}...")
    unique_emb_keys = df_full['_emb_key'].unique()
    embedding_dict, key_to_id_map = {}, {}
    for i, key in enumerate(tqdm(unique_emb_keys, desc=f"Extracting unique embeddings for {name}")):
        cleaned_key = df_full[df_full['_emb_key'] == key]['_cleaned_key'].iloc[0]
        # This function call is now safe because 'key' can no longer be None
        emb = get_embedding_for_worker(key, cleaned_key)
        padded_emb = np.full((MAX_MHC_LEN, ESM_DIM), PAD_VALUE, dtype=np.float16)
        padded_emb[:emb.shape[0]] = emb.astype(np.float16)
        embedding_dict[str(i)] = padded_emb
        key_to_id_map[key] = i

    lookup_path = os.path.join(output_dir, f"{name}_mhc_embedding_lookup.npz")
    np.savez_compressed(lookup_path, **embedding_dict)
    print(f"✓ Saved {len(unique_emb_keys)} unique embeddings to {lookup_path}")

    df_full['embedding_id'] = df_full['_emb_key'].map(key_to_id_map)

    num_workers = max(1, mp.cpu_count() // 2)
    chunk_size = int(np.ceil(len(df_full) / (num_workers * 4)))
    chunks = [df_full.iloc[i:i + chunk_size] for i in range(0, len(df_full), chunk_size)]
    worker_args = [(chunk, os.path.join(output_dir, f"{name}_shard_{i:04d}.tfrecord")) for i, chunk in
                   enumerate(chunks)]
    print(f"Starting TFRecord creation for {name} with {num_workers} processes...")
    init_args = (args.embed_npz, SEQ_MAP, EMBED_MAP, MHC_CLASS, ESM_DIM, MAX_PEP_LEN, MAX_MHC_LEN)
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:
        list(tqdm(pool.starmap(process_chunk_to_tfrecord, worker_args), total=len(chunks),
                  desc=f"Writing {name} TFRecords"))

    print(f"✓ All TFRecord shards for {name} created successfully in {output_dir}")


def main(args):
    """Main execution function."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    global MHC_CLASS, ESM_DIM, MAX_PEP_LEN, MAX_MHC_LEN, SEQ_MAP, EMBED_MAP

    print("Loading shared resources (embeddings, sequence maps)...")
    emb_db_local = np.load(args.embed_npz)
    MHC_CLASS = args.mhc_class
    ESM_DIM = int(next(iter(emb_db_local.values())).shape[1])

    SEQ_MAP = {clean_key(k): v for k, v in
               pd.read_csv(args.seq_csv, index_col="allele")["mhc_sequence"].to_dict().items()}
    EMBED_MAP = {k: v for k, v in pd.read_csv(args.embed_key, index_col="key")["mhc_sequence"].to_dict().items()}

    print("Calculating maximum sequence lengths from datasets...")
    df_train = pd.read_parquet(args.train_path)
    df_val = pd.read_parquet(args.val_path)
    MAX_PEP_LEN = int(pd.concat([df_train["long_mer"], df_val["long_mer"]]).str.len().max()) + 2
    MAX_MHC_LEN = 500 if MHC_CLASS == 2 else int(max(len(seq) for seq in SEQ_MAP.values())) + 2
    print(f"Using MAX_PEP_LEN={MAX_PEP_LEN}, MAX_MHC_LEN={MAX_MHC_LEN}, ESM_DIM={ESM_DIM}")

    os.makedirs(args.output_dir, exist_ok=True)

    metadata = {'MAX_PEP_LEN': MAX_PEP_LEN, 'MAX_MHC_LEN': MAX_MHC_LEN, 'ESM_DIM': ESM_DIM, 'MHC_CLASS': MHC_CLASS,
                'train_samples': len(df_train), 'val_samples': len(df_val)}
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    create_artifacts(args.train_path, args.output_dir, "train", args)
    create_artifacts(args.val_path, args.output_dir, "validation", args)

    print("\nAll artifacts created successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Parquet datasets to an optimized TFRecord format.")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--embed_npz", type=str, required=True)
    parser.add_argument("--seq_csv", type=str, required=True)
    parser.add_argument("--embed_key", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mhc_class", type=int, default=1, choices=[1, 2])
    args = parser.parse_args()
    main(args)