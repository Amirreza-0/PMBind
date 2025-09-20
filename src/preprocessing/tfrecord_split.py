#!/usr/bin/env python3
"""
Script to split TFRecord files into separate files for positive and negative samples.
This version is optimized for maximum I/O performance by using a batched-processing
approach that leverages the tf.data API for reading and Python for efficient writing.

Usage: python split_tfrecords.py --data_dir /path/to/tfrecords --output_dir /path/to/output
"""

import tensorflow as tf
import glob
import os
import argparse
import time
import json
from collections import defaultdict
from tqdm import tqdm

# Use a large batch size for efficient data transfer from TF to Python.
# Adjust based on your system's RAM. 32768 is a good starting point.
BATCH_SIZE = 32768


def create_writers(output_dir, num_negative_files=57):
    """Create and open TFRecordWriter objects for all output files."""
    os.makedirs(output_dir, exist_ok=True)

    positive_path = os.path.join(output_dir, "positive_samples.tfrecord")
    writers = {
        'positive': tf.io.TFRecordWriter(positive_path, options='GZIP'),
        'negative': [
            tf.io.TFRecordWriter(os.path.join(output_dir, f"negative_samples_{i:02d}.tfrecord"), options='GZIP')
            for i in range(num_negative_files)
        ]
    }

    print(f"Created writer for positive samples: {positive_path}")
    for i in range(num_negative_files):
        print(
            f"Created writer for negative samples {i}: {os.path.join(output_dir, f'negative_samples_{i:02d}.tfrecord')}")

    return writers


def close_writers(writers):
    """Safely close all TFRecord writers."""
    writers['positive'].close()
    for writer in writers['negative']:
        writer.close()


def parse_record_and_label(serialized_example):
    """Parse a TFRecord to get both the raw record and its label."""
    label = tf.io.parse_single_example(
        serialized_example, {'label': tf.io.FixedLenFeature([], tf.int64)}
    )['label']
    return serialized_example, label


def split_tfrecords_fastest(file_pattern, output_dir, dataset_name="Dataset", total_samples=None):
    """
    The definitive fast version. Uses a pure tf.data pipeline for reading/parsing
    and a batched Python loop for ultra-efficient writing.
    """
    files = sorted(glob.glob(file_pattern))
    if not files:
        print(f"No files found for pattern: {file_pattern}")
        return defaultdict(int)

    print(f"\nProcessing {len(files)} files from {dataset_name} using the fastest method...")

    writers = create_writers(output_dir)
    stats = defaultdict(int)
    negative_file_index = 0

    # 1. Create a highly parallel and efficient input pipeline in TensorFlow.
    # This pipeline handles reading, decompression, and parsing in the C++ backend.
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False  # Performance is better when order doesn't matter
    )
    # Map the parsing function and then batch everything.
    dataset = dataset.map(parse_record_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    # Prefetch batches to keep the GPU/CPU busy.
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # 2. Iterate over the dataset. `as_numpy_iterator` efficiently converts
    # batches of Tensors to NumPy arrays.
    pbar = tqdm(total=total_samples, desc=f"Splitting {dataset_name}", unit=" records")

    for records_batch, labels_batch in dataset.as_numpy_iterator():
        # This loop is in Python, operating on a batch of records in memory (NumPy arrays).
        # It's extremely fast because the data transfer from TF was done in one large chunk.
        for record, label in zip(records_batch, labels_batch):
            if label == 1:
                writers['positive'].write(record)
                stats['positive'] += 1
            else:
                writer_idx = negative_file_index % 57
                writers['negative'][writer_idx].write(record)
                stats['negative'] += 1
                negative_file_index += 1

        pbar.update(len(records_batch))

    pbar.close()
    stats['total'] = stats['positive'] + stats['negative']

    close_writers(writers)

    print("\nNegative samples distribution:")
    neg_counts = [0] * 57
    # Calculate the final distribution accurately
    for i in range(stats['negative']):
        neg_counts[i % 57] += 1
    for i, count in enumerate(neg_counts):
        print(f"  negative_samples_{i:02d}.tfrecord: {count:,} samples")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Split TFRecord files into positive and negative samples (Fastest Version)')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing input TFRecord files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save split TFRecord files')
    parser.add_argument('--train_pattern', type=str, default='train_shard_*.tfrecord',
                        help='Pattern for training files')
    parser.add_argument('--val_pattern', type=str, default='validation_shard_*.tfrecord',
                        help='Pattern for validation files')
    parser.add_argument('--split_train', action='store_true', default=True, help='Split training set')
    parser.add_argument('--split_val', action='store_true', default=False, help='Split validation set')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} does not exist")
        return

    metadata = {}
    metadata_path = os.path.join(args.data_dir, 'metadata.json')
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Successfully loaded metadata from {metadata_path}")
    except FileNotFoundError:
        print("Warning: metadata.json not found. Progress bars may not show accurate totals.")

    print("=" * 60)
    print("TFRecord Splitter - Positive/Negative Separation (FASTEST MODE)")
    print("=" * 60)

    start_time = time.time()
    all_stats = {}

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ Found {len(gpus)} GPU(s). TensorFlow `tf.data` pipeline will leverage it.")
    else:
        print("Info: No GPU found. `tf.data` pipeline will run on CPU.")

    # Always use the fastest, most reliable function
    split_function = split_tfrecords_fastest

    if args.split_train:
        train_pattern = os.path.join(args.data_dir, args.train_pattern)
        train_output_dir = os.path.join(args.output_dir, 'train')
        train_samples = metadata.get('train_samples', None)  # Or the correct key from your json
        train_stats = split_function(train_pattern, train_output_dir, "Training", train_samples)
        all_stats['train'] = train_stats

    if args.split_val:
        val_pattern = os.path.join(args.data_dir, args.val_pattern)
        val_output_dir = os.path.join(args.output_dir, 'validation')
        val_samples = metadata.get('val_samples', None)  # Or the correct key from your json
        val_stats = split_function(val_pattern, val_output_dir, "Validation", val_samples)
        all_stats['validation'] = val_stats

    if not args.split_val:
        # save validation as one file if not splitting
        valid_pattern = os.path.join(args.data_dir, args.val_pattern)
        valid_output_dir = os.path.join(args.output_dir, 'validation')
        os.makedirs(valid_output_dir, exist_ok=True)
        valid_files = sorted(glob.glob(valid_pattern))
        if valid_files:
            out_path = os.path.join(valid_output_dir, "validation_samples.tfrecord")
            valid_writer = tf.io.TFRecordWriter(out_path, options='GZIP')
            total_valid = 0
            pos_valid = 0
            neg_valid = 0
            for vf in valid_files:
                for record in tf.data.TFRecordDataset(vf, compression_type="GZIP"):
                    # count labels for accurate stats
                    parsed = tf.io.parse_single_example(
                        record, {'label': tf.io.FixedLenFeature([], tf.int64)}
                    )
                    if int(parsed['label'].numpy()) == 1:
                        pos_valid += 1
                    else:
                        neg_valid += 1
                    valid_writer.write(record.numpy())
                    total_valid += 1
            valid_writer.close()
            all_stats['validation'] = {
                'total': total_valid,
                'positive': pos_valid,
                'negative': neg_valid,
                'single_file': True
            }
            print(f"\nValidation samples saved to {out_path} with {total_valid} samples.")

    end_time = time.time()

    print("\n" + "=" * 60)
    print(f"SPLITTING COMPLETED (Total time: {end_time - start_time:.2f} seconds)")
    print("=" * 60)

    for name, stats in all_stats.items():
        print(f"\n{name.upper()} Set Results:")
        print(f"  Total samples processed: {stats['total']:,}")
        print(f"  Positive samples: {stats['positive']:,} → 1 file")
        print(f"  Negative samples: {stats['negative']:,} → 57 files")

    stats_path = os.path.join(args.output_dir, 'split_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=4)
    print(f"\n✓ Statistics saved to {stats_path}")

    print("\n✅ All files have been successfully split!")


if __name__ == "__main__":
    main()