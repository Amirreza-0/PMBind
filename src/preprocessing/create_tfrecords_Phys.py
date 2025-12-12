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
import time

# Local utilities for sequence and key processing
from utils_Phys import (get_embed_key, get_seq, clean_key, PAD_VALUE, PAD_INDEX_23, PAD_INDEX_OHE, NORM_TOKEN,
                   seq_to_indices_AA_ENCODINGS, seq_to_ohe_indices, get_mhc_seq_class2, get_embed_key_class2, seq_to_aa_encodings)

# --- Globals for worker processes ---
MHC_CLASS = 1
ESM_DIM = 14  # Dimension of AA_ENCODINGS from utils_Phys (replacing ESM embeddings)
MAX_PEP_LEN = 0
MAX_MHC_LEN = 0
EMBED_MAP = None
SEQ_MAP = None


# --- TFRecord Helper Functions ---
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(values):
    """Returns a int64_list from a list of int / uint."""
    arr = np.asarray(values, dtype=np.int64).ravel().tolist()
    return tf.train.Feature(int64_list=tf.train.Int64List(value=arr))


def process_embedding_batch(emb_keys_batch, start_idx, mhc_class):
    """Process a batch of embedding keys to create AA encodings in parallel."""
    # Use global variables initialized by init_worker
    global EMBED_MAP, SEQ_MAP, ESM_DIM, MAX_MHC_LEN, PAD_VALUE

    batch_embeddings = {}
    batch_key_to_id = {}

    for i, key in enumerate(emb_keys_batch):
        emb_id = start_idx + i

        # Get MHC sequence for this key
        if mhc_class == 1:
            mhc_seq = get_seq(key, SEQ_MAP)
        else:
            mhc_seq = get_mhc_seq_class2(key, EMBED_MAP, SEQ_MAP)

        # Create AA encoding for the MHC sequence instead of loading ESM embeddings
        mhc_encoding = seq_to_aa_encodings(mhc_seq, MAX_MHC_LEN)  # (MAX_MHC_LEN, ESM_DIM=14)

        # Create padded embedding
        padded_emb = np.full((MAX_MHC_LEN, ESM_DIM), PAD_VALUE, dtype=np.float32)
        padded_emb[:mhc_encoding.shape[0]] = mhc_encoding.astype(np.float32)

        # Optimized padding detection - only compute what we need
        seq_len = len(mhc_seq)
        if seq_len < MAX_MHC_LEN:
            padded_emb[seq_len:] = PAD_VALUE  # Set positions beyond sequence length to 0

        batch_embeddings[str(emb_id)] = padded_emb
        batch_key_to_id[key] = emb_id

    return batch_embeddings, batch_key_to_id


# --- Multiprocessing Worker Setup and Functions ---
def init_worker(seq_map_data, embed_map_data, mhc_class, esm_dim, max_pep, max_mhc):
    """Initializes globals in each worker process for efficient data access."""
    global SEQ_MAP, EMBED_MAP, MHC_CLASS, ESM_DIM, MAX_PEP_LEN, MAX_MHC_LEN, PAD_VALUE
    SEQ_MAP = seq_map_data
    EMBED_MAP = embed_map_data
    MHC_CLASS = mhc_class
    ESM_DIM = esm_dim  # This is now 14 (AA_ENCODINGS dimension)
    MAX_PEP_LEN = max_pep
    MAX_MHC_LEN = max_mhc


def process_chunk_to_tfrecord(chunk_df, output_path):
    """Worker Function: Converts a DataFrame chunk to a TFRecord file with optimized processing."""
    options = tf.io.TFRecordOptions(compression_type="GZIP")

    # Pre-allocate arrays to avoid repeated memory allocation
    chunk_size = len(chunk_df)
    features_list = []

    # Vectorized operations where possible
    long_mers = chunk_df['long_mer'].values
    mhc_seqs = chunk_df['_mhc_seq'].values
    embedding_ids = chunk_df['embedding_id'].values.astype(np.int64)
    labels = chunk_df['assigned_label'].values.astype(np.int64)

    with tf.io.TFRecordWriter(output_path, options=options) as writer:
        for i in range(chunk_size):
            # Batch process indices to reduce function call overhead
            pep_indices = seq_to_indices_AA_ENCODINGS(long_mers[i], MAX_PEP_LEN)
            pep_ohe_indices = seq_to_ohe_indices(long_mers[i], MAX_PEP_LEN)
            mhc_ohe_indices = seq_to_ohe_indices(mhc_seqs[i], MAX_MHC_LEN)

            feature = {
                'pep_indices': _int64_list_feature(pep_indices),
                'pep_ohe_indices': _int64_list_feature(pep_ohe_indices),
                'mhc_ohe_indices': _int64_list_feature(mhc_ohe_indices),
                'embedding_id': _int64_feature(embedding_ids[i]),
                'label': _int64_feature(labels[i]),
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example_proto.SerializeToString())

    return output_path


def process_chunk_preprocess(chunk, seq_map, embed_map, mhc_class):
    """
    Worker function for the initial DataFrame preprocessing.
    This is the location of the critical fix.
    """
    chunk['_cleaned_key'] = chunk['allele'].apply(lambda k: clean_key(k))

    # Safely handle sequence lookup, especially for None keys
    if mhc_class == 2:
        chunk['_emb_key'] = chunk['_cleaned_key'].apply(lambda k: get_embed_key_class2(k, embed_map))
        chunk['_mhc_seq'] = chunk['_cleaned_key'].apply(lambda k: get_mhc_seq_class2(k, embed_map, seq_map))
    else:
        chunk['_emb_key'] = chunk['_cleaned_key'].apply(lambda k: get_embed_key(k, embed_map))
        chunk['_mhc_seq'] = chunk['_cleaned_key'].apply(lambda k: get_seq(k, seq_map) if k is not None else '')

    # Print the dropped rows for debugging
    dropped_no_emb_key = chunk[chunk['_emb_key'].isna()]
    dropped_no_mhc_seq = chunk[chunk['_mhc_seq'] == ""]
    if not dropped_no_emb_key.empty:
        print("Dropped rows with no emb_key unique alleles:")
        print(dropped_no_emb_key['allele'].unique())
        print(dropped_no_emb_key[['allele', '_cleaned_key', '_emb_key']])
        raise ValueError("Some alleles could not be mapped to an embedding key. See above for details.")
    if not dropped_no_mhc_seq.empty:
        print("Dropped rows with no mhc_seq unique alleles:")
        print(dropped_no_mhc_seq['allele'].unique())
        print(dropped_no_mhc_seq[['allele', '_cleaned_key', '_mhc_seq']])
        raise ValueError("Some alleles could not be mapped to an MHC sequence. See above for details.")
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
    start_time = time.time()
    print(f"\n--- Processing {name} dataset from {df_path} ---")

    # Step 1: Data preprocessing
    preprocess_start = time.time()
    print(f"Loading and preprocessing {name} DataFrame...")
    df_raw = pd.read_parquet(df_path)
    df_full = preprocess_df(df_raw, SEQ_MAP, EMBED_MAP, MHC_CLASS)
    preprocess_time = time.time() - preprocess_start
    print(f"✓ Preprocessing complete for {name}. Found {len(df_full)} valid rows. ({preprocess_time:.1f}s)")

    # Step 2: Create embeddings lookup (now using AA encodings instead of ESM)
    embedding_start = time.time()
    print(f"Creating central MHC embedding lookup file for {name}...")
    unique_emb_keys = df_full['_emb_key'].unique()

    # Process embeddings in parallel batches for better performance
    num_workers = max(1, mp.cpu_count() // 2)
    batch_size = max(1, len(unique_emb_keys) // (num_workers * 2))

    embedding_dict, key_to_id_map = {}, {}

    # Create batches for parallel processing
    key_batches = [unique_emb_keys[i:i + batch_size] for i in range(0, len(unique_emb_keys), batch_size)]

    # Process embeddings in parallel
    init_args = (SEQ_MAP, EMBED_MAP, MHC_CLASS, ESM_DIM, MAX_PEP_LEN, MAX_MHC_LEN)
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:
        batch_args = [(batch, i * batch_size, MHC_CLASS)
                      for i, batch in enumerate(key_batches)]

        results = list(tqdm(
            pool.starmap(process_embedding_batch, batch_args),
            total=len(key_batches),
            desc=f"Processing embedding batches for {name}"
        ))

    # Merge results from all batches
    for batch_embeddings, batch_key_to_id in results:
        embedding_dict.update(batch_embeddings)
        key_to_id_map.update(batch_key_to_id)

    lookup_path = os.path.join(output_dir, f"{name}_mhc_embedding_lookup.npz")
    np.savez_compressed(lookup_path, **embedding_dict)
    embedding_time = time.time() - embedding_start
    print(f"✓ Saved {len(unique_emb_keys)} unique embeddings to {lookup_path} ({embedding_time:.1f}s)")

    # Step 3: Create TFRecord files
    tfrecord_start = time.time()
    df_full['embedding_id'] = df_full['_emb_key'].map(key_to_id_map)

    # Intelligent chunk sizing based on data size and available resources
    num_workers = max(1, mp.cpu_count() // 2)

    # Optimize chunk size based on dataset size - larger chunks for efficiency, but not too large for memory
    target_chunks_per_worker = 4
    min_chunk_size = 1000  # Minimum chunk size for efficiency
    max_chunk_size = 50000  # Maximum chunk size to avoid memory issues

    optimal_chunk_size = max(min_chunk_size,
                             min(max_chunk_size,
                                 len(df_full) // (num_workers * target_chunks_per_worker)))

    chunks = [df_full.iloc[i:i + optimal_chunk_size] for i in range(0, len(df_full), optimal_chunk_size)]

    print(f"Processing {len(df_full)} samples in {len(chunks)} chunks of ~{optimal_chunk_size} samples each")

    worker_args = [(chunk, os.path.join(output_dir, f"{name}_shard_{i:04d}.tfrecord")) for i, chunk in
                   enumerate(chunks)]
    print(f"Starting TFRecord creation for {name} with {num_workers} processes...")
    init_args = (SEQ_MAP, EMBED_MAP, MHC_CLASS, ESM_DIM, MAX_PEP_LEN, MAX_MHC_LEN)
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:
        list(tqdm(pool.starmap(process_chunk_to_tfrecord, worker_args), total=len(chunks),
                  desc=f"Writing {name} TFRecords"))

    tfrecord_time = time.time() - tfrecord_start
    total_time = time.time() - start_time

    print(f"✓ All TFRecord shards for {name} created successfully in {output_dir}")
    print(f"Performance Summary for {name}:")
    print(f"  • Preprocessing: {preprocess_time:.1f}s")
    print(f"  • Embedding processing: {embedding_time:.1f}s")
    print(f"  • TFRecord creation: {tfrecord_time:.1f}s")
    print(f"  • Total time: {total_time:.1f}s")
    print(f"  • Processing rate: {len(df_full) / total_time:.0f} samples/second")


def main(args):
    """Main execution function."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    global MHC_CLASS, ESM_DIM, MAX_PEP_LEN, MAX_MHC_LEN, SEQ_MAP, EMBED_MAP

    print("Loading shared resources (sequence maps)...")
    MHC_CLASS = args.mhc_class
    ESM_DIM = 14  # Dimension of AA_ENCODINGS from utils_Phys (replacing ESM embeddings)

    SEQ_MAP = {clean_key(k): v for k, v in
               pd.read_csv(args.seq_csv, index_col="allele")["mhc_sequence"].to_dict().items()}
    EMBED_MAP = {k: v for k, v in pd.read_csv(args.embed_key, index_col="key")["mhc_sequence"].to_dict().items()}

    print("Calculating maximum sequence lengths from datasets...")
    df_train = pd.read_parquet(args.train_path)
    MAX_PEP_LEN = int(df_train["long_mer"].str.len().max()) + 2  # + 2 Buffer
    MAX_MHC_LEN = 500 if MHC_CLASS == 2 else int(max(len(seq) for seq in SEQ_MAP.values())) + 2  # + 2 Buffer
    print(f"Using MAX_PEP_LEN={MAX_PEP_LEN}, MAX_MHC_LEN={MAX_MHC_LEN}, ESM_DIM={ESM_DIM}")

    os.makedirs(args.output_dir, exist_ok=True)

    metadata = {'MAX_PEP_LEN': MAX_PEP_LEN, 'MAX_MHC_LEN': MAX_MHC_LEN, 'ESM_DIM': ESM_DIM, 'MHC_CLASS': MHC_CLASS,
                'train_samples': len(df_train)}
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    create_artifacts(args.train_path, args.output_dir, "train", args)

    print("\nAll artifacts created successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Parquet datasets to an optimized TFRecord format with AA encodings.")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--seq_csv", type=str, required=True)
    parser.add_argument("--embed_key", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mhc_class", type=int, default=1, choices=[1, 2])
    args = parser.parse_args()
    main(args)