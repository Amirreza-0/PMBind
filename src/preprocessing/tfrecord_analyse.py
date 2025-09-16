#!/usr/bin/env python3
"""
Script to efficiently count positive and negative samples in TFRecord files.
This version uses a parallelized, batched tf.data pipeline for maximum speed
and includes a tqdm progress bar for visibility.

Usage: python count_samples_final.py --data_dir /path/to/tfrecords
"""

import tensorflow as tf
import glob
import os
import argparse
from tqdm import tqdm
import time
import json

# Define a large batch size. The bigger the better, as long as it fits in memory.
# This value determines how many records are processed in one go.
BATCH_SIZE = 16384


def parse_tfrecord_for_label(example_proto):
    """Parse only the label from a TFRecord example."""
    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    return parsed['label']


def count_samples_with_progress(file_pattern, description="Processing", total_samples=None):
    """
    Efficiently count samples using a parallel, batched tf.data pipeline
    with a tqdm progress bar for visibility.

    Args:
        file_pattern: Glob pattern for TFRecord files.
        description: Description for the tqdm progress bar.
        total_samples: Total number of samples for an accurate progress bar.

    Returns:
        tuple: (positive_count, negative_count, total_count)
    """
    files = sorted(glob.glob(file_pattern))
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return 0, 0, 0

    print(f"\nFound {len(files)} files. Building high-performance pipeline...")

    # 1. Create a dataset of filenames.
    dataset = tf.data.Dataset.from_tensor_slices(files)

    # 2. Use interleave to read from multiple files in parallel.
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )

    # 3. Use map to parse records in parallel.
    dataset = dataset.map(parse_tfrecord_for_label, num_parallel_calls=tf.data.AUTOTUNE)

    # 4. Batch records into large Tensors. This is key for performance.
    dataset = dataset.batch(BATCH_SIZE)

    # 5. Prefetch next batches while the current one is being processed.
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Initialize Python counters
    positive_count = 0
    negative_count = 0
    total_count = 0

    # Determine total number of batches for tqdm
    total_batches = (total_samples // BATCH_SIZE) + 1 if total_samples else None

    # 6. Iterate through the batched dataset with a progress bar
    pbar = tqdm(dataset, total=total_batches, desc=f"Analyzing {description} set")
    for labels_batch in pbar:
        # labels_batch is a Tensor of shape [BATCH_SIZE]

        # Use the highly optimized bincount to count 0s and 1s in the tensor.
        # It's much faster than iterating or using tf.reduce_sum.
        counts = tf.math.bincount(tf.cast(labels_batch, tf.int32), minlength=2)

        neg_in_batch = counts[0].numpy()
        pos_in_batch = counts[1].numpy()

        negative_count += neg_in_batch
        positive_count += pos_in_batch
        total_count += (neg_in_batch + pos_in_batch)

    return positive_count, negative_count, total_count


def main():
    parser = argparse.ArgumentParser(description='Count positive and negative samples in TFRecord files')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing TFRecord files')
    parser.add_argument('--train_pattern', type=str, default='train_shard_*.tfrecord',
                        help='Pattern for training files')
    parser.add_argument('--val_pattern', type=str, default='validation_shard_*.tfrecord',
                        help='Pattern for validation files')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} does not exist")
        return

    # Try to load metadata for accurate progress bars
    metadata = {}
    metadata_path = os.path.join(args.data_dir, 'metadata.json')
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Successfully loaded metadata from {metadata_path}")
    except FileNotFoundError:
        print("Warning: metadata.json not found. Progress bars will not show percentages.")

    train_samples_total = metadata.get('train_samples')
    val_samples_total = metadata.get('val_samples')

    print("=" * 60)
    start_time = time.time()

    train_pattern = os.path.join(args.data_dir, args.train_pattern)
    train_pos, train_neg, train_total = count_samples_with_progress(
        train_pattern, "Training", train_samples_total
    )

    val_pattern = os.path.join(args.data_dir, args.val_pattern)
    val_pos, val_neg, val_total = count_samples_with_progress(
        val_pattern, "Validation", val_samples_total
    )

    end_time = time.time()

    print("\n" + "=" * 60)
    print(f"SUMMARY (Completed in {end_time - start_time:.2f} seconds)")
    print("=" * 60)

    print("\nTraining Set:")
    print(f"  Total samples:    {train_total:,}")
    print(
        f"  Positive samples: {train_pos:,} ({train_pos / train_total * 100:.2f}%)" if train_total > 0 else "  Positive samples: 0")
    print(
        f"  Negative samples: {train_neg:,} ({train_neg / train_total * 100:.2f}%)" if train_total > 0 else "  Negative samples: 0")

    print("\nValidation Set:")
    print(f"  Total samples:    {val_total:,}")
    print(
        f"  Positive samples: {val_pos:,} ({val_pos / val_total * 100:.2f}%)" if val_total > 0 else "  Positive samples: 0")
    print(
        f"  Negative samples: {val_neg:,} ({val_neg / val_total * 100:.2f}%)" if val_total > 0 else "  Negative samples: 0")


if __name__ == "__main__":
    main()