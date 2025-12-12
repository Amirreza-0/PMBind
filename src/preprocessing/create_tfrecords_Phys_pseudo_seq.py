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

# Local utilities
from utils_Phys import (get_seq, clean_key, PAD_VALUE, PAD_INDEX_23, PAD_INDEX_OHE, NORM_TOKEN,
                   seq_to_indices_AA_ENCODINGS, seq_to_ohe_indices, seq_to_aa_encodings)

# --- Globals for worker processes ---
MHC_CLASS = 1
ESM_DIM = 14
MAX_PEP_LEN = 0
MAX_MHC_LEN = 0
KEY_TO_SEQ = None   # Maps CSV 'key' -> Sequence
ALLELE_TO_KEY = None # Maps Train 'allele' -> CSV 'key'


# --- Helper for Fuzzy Matching ---
def normalize_allele(s):
    """Removes *, :, -, spaces, and dots for robust comparison."""
    return str(s).replace("*", "").replace(":", "").replace("-", "").replace(" ", "").replace(".", "").upper()


# --- TFRecord Helper Functions ---
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(values):
    arr = np.asarray(values, dtype=np.int64).ravel().tolist()
    return tf.train.Feature(int64_list=tf.train.Int64List(value=arr))


def process_embedding_batch(keys_batch, start_idx):
    """Process a batch of embedding keys (from the CSV 'key' column) to create encodings."""
    global KEY_TO_SEQ, ESM_DIM, MAX_MHC_LEN, PAD_VALUE

    batch_embeddings = {}
    batch_key_to_id = {}

    for i, key in enumerate(keys_batch):
        emb_id = start_idx + i

        # Lookup sequence using the CSV key
        mhc_seq = KEY_TO_SEQ.get(key, "")
        
        if mhc_seq == "":
            mhc_encoding = np.zeros((0, ESM_DIM))
        else:
            mhc_encoding = seq_to_aa_encodings(mhc_seq, MAX_MHC_LEN)

        # Create padded embedding
        padded_emb = np.full((MAX_MHC_LEN, ESM_DIM), PAD_VALUE, dtype=np.float32)
        padded_emb[:mhc_encoding.shape[0]] = mhc_encoding.astype(np.float32)

        seq_len = len(mhc_seq)
        if seq_len < MAX_MHC_LEN:
            padded_emb[seq_len:] = PAD_VALUE

        batch_embeddings[str(emb_id)] = padded_emb
        batch_key_to_id[key] = emb_id

    return batch_embeddings, batch_key_to_id


# --- Multiprocessing Worker Setup and Functions ---
def init_worker(key_to_seq_data, allele_to_key_data, mhc_class, esm_dim, max_pep, max_mhc):
    """Initializes globals in each worker process."""
    global KEY_TO_SEQ, ALLELE_TO_KEY, MHC_CLASS, ESM_DIM, MAX_PEP_LEN, MAX_MHC_LEN, PAD_VALUE
    KEY_TO_SEQ = key_to_seq_data
    ALLELE_TO_KEY = allele_to_key_data
    MHC_CLASS = mhc_class
    ESM_DIM = esm_dim
    MAX_PEP_LEN = max_pep
    MAX_MHC_LEN = max_mhc


def process_chunk_to_tfrecord(chunk_df, output_path):
    """Worker Function: Converts a DataFrame chunk to a TFRecord file."""
    options = tf.io.TFRecordOptions(compression_type="GZIP")

    chunk_size = len(chunk_df)
    
    long_mers = chunk_df['long_mer'].values
    mhc_seqs = chunk_df['_mhc_seq'].values
    embedding_ids = chunk_df['embedding_id'].values.astype(np.int64)
    labels = chunk_df['assigned_label'].values.astype(np.int64)

    with tf.io.TFRecordWriter(output_path, options=options) as writer:
        for i in range(chunk_size):
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


def process_chunk_preprocess(chunk, allele_to_key_map, key_to_seq_map):
    """
    Worker function for DataFrame preprocessing.
    Maps: Train Allele -> CSV Key -> Sequence
    """
    chunk['_cleaned_key'] = chunk['allele'].apply(lambda k: clean_key(k))

    # 1. Map cleaned train allele to the CSV Key (using the resolved map)
    chunk['_emb_key'] = chunk['_cleaned_key'].map(allele_to_key_map)
    
    # 2. Map CSV Key to Sequence (for validity checking and OHE generation)
    chunk['_mhc_seq'] = chunk['_emb_key'].map(key_to_seq_map)

    # Validation
    dropped = chunk[chunk['_emb_key'].isna() | chunk['_mhc_seq'].isna() | (chunk['_mhc_seq'] == "")]
    if not dropped.empty:
        print("Dropped rows with no matching key or sequence:")
        print(dropped['allele'].unique())
        raise ValueError("Some alleles could not be mapped to a key/sequence.")
        
    return chunk.dropna(subset=['_emb_key', '_mhc_seq'])


def preprocess_df(df, allele_to_key, key_to_seq, num_workers=None):
    """Preprocesses the input DataFrame in parallel."""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() // 2)
    chunk_size = max(1, len(df) // (num_workers * 4))
    chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]
    
    process_func = partial(process_chunk_preprocess, allele_to_key_map=allele_to_key, key_to_seq_map=key_to_seq)
    
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
    
    # Pass both maps to the preprocessor
    df_full = preprocess_df(df_raw, ALLELE_TO_KEY, KEY_TO_SEQ)
    
    preprocess_time = time.time() - preprocess_start
    print(f"✓ Preprocessing complete for {name}. Found {len(df_full)} valid rows. ({preprocess_time:.1f}s)")

    # Step 2: Create embeddings lookup
    embedding_start = time.time()
    print(f"Creating central MHC embedding lookup file for {name} using CSV keys...")
    
    # The _emb_key is now the 'key' from the CSV
    unique_keys = df_full['_emb_key'].unique()

    num_workers = max(1, mp.cpu_count() // 2)
    batch_size = max(1, len(unique_keys) // (num_workers * 2))

    embedding_dict, key_to_id_map = {}, {}
    key_batches = [unique_keys[i:i + batch_size] for i in range(0, len(unique_keys), batch_size)]

    init_args = (KEY_TO_SEQ, ALLELE_TO_KEY, MHC_CLASS, ESM_DIM, MAX_PEP_LEN, MAX_MHC_LEN)
    
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:
        batch_args = [(batch, i * batch_size) for i, batch in enumerate(key_batches)]

        results = list(tqdm(
            pool.starmap(process_embedding_batch, batch_args),
            total=len(key_batches),
            desc=f"Processing embedding batches for {name}"
        ))

    for batch_embeddings, batch_key_to_id in results:
        embedding_dict.update(batch_embeddings)
        key_to_id_map.update(batch_key_to_id)

    lookup_path = os.path.join(output_dir, f"{name}_mhc_embedding_lookup.npz")
    np.savez_compressed(lookup_path, **embedding_dict)
    embedding_time = time.time() - embedding_start
    print(f"✓ Saved {len(unique_keys)} unique embeddings to {lookup_path} ({embedding_time:.1f}s)")

    # Step 3: Create TFRecord files
    tfrecord_start = time.time()
    df_full['embedding_id'] = df_full['_emb_key'].map(key_to_id_map)

    num_workers = max(1, mp.cpu_count() // 2)
    target_chunks_per_worker = 4
    min_chunk_size = 1000
    max_chunk_size = 50000

    optimal_chunk_size = max(min_chunk_size,
                             min(max_chunk_size,
                                 len(df_full) // (num_workers * target_chunks_per_worker)))

    chunks = [df_full.iloc[i:i + optimal_chunk_size] for i in range(0, len(df_full), optimal_chunk_size)]

    print(f"Processing {len(df_full)} samples in {len(chunks)} chunks of ~{optimal_chunk_size} samples each")

    worker_args = [(chunk, os.path.join(output_dir, f"{name}_shard_{i:04d}.tfrecord")) for i, chunk in
                   enumerate(chunks)]
    
    print(f"Starting TFRecord creation for {name} with {num_workers} processes...")
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


def main(args):
    """Main execution function."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    global MHC_CLASS, ESM_DIM, MAX_PEP_LEN, MAX_MHC_LEN, KEY_TO_SEQ, ALLELE_TO_KEY

    print("Loading shared resources (sequence maps)...")
    MHC_CLASS = args.mhc_class
    ESM_DIM = 14

    # 1. Load CSV and build Lookups
    #    KEY_TO_SEQ: Maps the official 'key' -> 'mhc_sequence'
    #    norm_map:   Maps normalized 'allele' -> official 'key' (for fuzzy matching)
    print(f"Loading sequences from {args.seq_csv}...")
    df_seqs = pd.read_csv(args.seq_csv)
    
    KEY_TO_SEQ = {}
    norm_map = {}

    for a, seq, key in zip(df_seqs['allele'], df_seqs['mhc_sequence'], df_seqs['key']):
        if isinstance(seq, str) and isinstance(key, str):
            KEY_TO_SEQ[key] = seq.replace(".", "")
            norm_map[normalize_allele(a)] = key

    # 2. Load Train Data to identify which alleles we need to map
    print("Calculating maximum sequence lengths from datasets...")
    df_train = pd.read_parquet(args.train_path)
    MAX_PEP_LEN = int(df_train["long_mer"].str.len().max()) + 2
    
    # 3. Resolve Train Alleles -> CSV Keys
    print("Resolving allele keys with fuzzy matching...")
    ALLELE_TO_KEY = {}
    unique_train_alleles = df_train['allele'].unique()
    resolved_count = 0

    for allele in unique_train_alleles:
        norm_a = normalize_allele(allele)
        found_key = None
        
        # A. Exact Normalized Match
        if norm_a in norm_map:
            found_key = norm_map[norm_a]
        else:
            # B. Fuzzy Match (Train starts with CSV allele?)
            for base_k_norm, base_key in norm_map.items():
                if norm_a.startswith(base_k_norm):
                    found_key = base_key
                    break
            
            # C. Fuzzy Match (CSV allele starts with Train?)
            if found_key is None:
                for base_k_norm, base_key in norm_map.items():
                    if base_k_norm.startswith(norm_a):
                        found_key = base_key
                        break
        
        if found_key is not None:
            # We map the *cleaned* train allele to the *CSV key*
            ALLELE_TO_KEY[clean_key(allele)] = found_key
            resolved_count += 1
        else:
            print(f"Warning: Could not resolve key for allele: {allele}")

    print(f"Resolved keys for {resolved_count}/{len(unique_train_alleles)} unique alleles.")

    # Set Max MHC Len based on the sequences we are actually using
    if not KEY_TO_SEQ:
        raise ValueError("No sequences loaded from CSV!")
    
    MAX_MHC_LEN = 500 if MHC_CLASS == 2 else int(max(len(seq) for seq in KEY_TO_SEQ.values())) + 2
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
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mhc_class", type=int, default=1, choices=[1, 2])
    args = parser.parse_args()
    main(args)