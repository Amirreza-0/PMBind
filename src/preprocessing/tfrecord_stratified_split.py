#!/usr/bin/env python
"""
Creates pre-stratified batch files from training TFRecords to avoid runtime batching computation.

This script:
1. Reads positive and negative training TFRecord files
2. Creates stratified batches with specified positive/negative ratios
3. Saves each batch as a separate TFRecord file
4. Generates metadata for efficient training

Usage:
    python tfrecord_stratified_split.py --tfrecord_dir /path/to/tfrecords --batch_size 1024 --pos_ratio 0.02
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import shutil


def count_records_in_tfrecord(file_path: str) -> int:
    """Count the number of records in a TFRecord file."""
    count = 0
    try:
        for _ in tf.data.TFRecordDataset(file_path, compression_type="GZIP"):
            count += 1
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0
    return count


def read_records_from_tfrecord(file_path: str) -> List[bytes]:
    """Read all records from a TFRecord file and return as list of byte strings."""
    records = []
    try:
        dataset = tf.data.TFRecordDataset(file_path, compression_type="GZIP")
        for record in dataset:
            records.append(record.numpy())
    except Exception as e:
        print(f"Error reading records from {file_path}: {e}")
        return []
    return records


def parse_record_label(record_bytes: bytes) -> int:
    """Parse a TFRecord and extract the label."""
    try:
        feature_description = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'pep_indices': tf.io.FixedLenFeature([], tf.string),
            'pep_ohe_indices': tf.io.FixedLenFeature([], tf.string),
            'mhc_indices': tf.io.FixedLenFeature([], tf.string),
            'mhc_ohe_indices': tf.io.FixedLenFeature([], tf.string),
            'embedding_id': tf.io.FixedLenFeature([], tf.int64),
        }

        parsed = tf.io.parse_single_example(record_bytes, feature_description)
        label = int(parsed['label'].numpy())
        return label
    except Exception as e:
        print(f"Error parsing record: {e}")
        return -1


def write_tfrecord_batch(records: List[bytes], output_path: str):
    """Write a list of records to a compressed TFRecord file."""
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(output_path, options=options) as writer:
        for record in records:
            writer.write(record)


def create_stratified_batches(
    positive_files: List[str],
    negative_files: List[str],
    output_dir: str,
    batch_size: int = 1024,
    pos_ratio: float = 0.02,
    num_epochs: int = 100,
    seed: int = 42
):
    """
    Create stratified batch files from positive and negative TFRecord files.

    Args:
        positive_files: List of positive TFRecord file paths
        negative_files: List of negative TFRecord file paths
        output_dir: Directory to save batch files
        batch_size: Number of samples per batch
        pos_ratio: Ratio of positive samples per batch
        num_epochs: Number of epoch's worth of batches to create
        seed: Random seed for reproducibility
    """
    print(f"Creating stratified batches with batch_size={batch_size}, pos_ratio={pos_ratio}")

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)

    # Calculate samples per batch
    pos_per_batch = max(1, int(batch_size * pos_ratio))
    neg_per_batch = batch_size - pos_per_batch

    print(f"Each batch will contain: {pos_per_batch} positive, {neg_per_batch} negative samples")

    # Create output directory
    batches_dir = Path(output_dir) / "stratified_batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    # Load all positive records
    print("Loading positive samples...")
    all_positive_records = []
    for pos_file in tqdm(positive_files, desc="Reading positive files"):
        records = read_records_from_tfrecord(pos_file)
        all_positive_records.extend(records)

    print(f"Loaded {len(all_positive_records):,} positive samples")

    # Load all negative records
    print("Loading negative samples...")
    all_negative_records = []
    for neg_file in tqdm(negative_files, desc="Reading negative files"):
        records = read_records_from_tfrecord(neg_file)
        all_negative_records.extend(records)

    print(f"Loaded {len(all_negative_records):,} negative samples")

    # Verify we have enough samples
    if len(all_positive_records) < pos_per_batch:
        raise ValueError(f"Not enough positive samples: need {pos_per_batch}, have {len(all_positive_records)}")

    if len(all_negative_records) < neg_per_batch:
        raise ValueError(f"Not enough negative samples: need {neg_per_batch}, have {len(all_negative_records)}")

    # Calculate number of batches we can create
    max_batches_from_pos = len(all_positive_records) // pos_per_batch
    max_batches_from_neg = len(all_negative_records) // neg_per_batch

    # Calculate target number of batches (num_epochs worth)
    batches_per_epoch = min(1000, max_batches_from_pos, max_batches_from_neg)  # Max 1000 batches per epoch
    total_target_batches = num_epochs * batches_per_epoch

    print(f"Creating {total_target_batches:,} batches ({batches_per_epoch} per epoch × {num_epochs} epochs)")

    # Create batches
    batch_files = []
    batch_metadata = []

    for batch_idx in tqdm(range(total_target_batches), desc="Creating batches"):
        # Sample positive records (with replacement if needed)
        if len(all_positive_records) >= pos_per_batch:
            sampled_pos = random.sample(all_positive_records, pos_per_batch)
        else:
            # With replacement if not enough unique samples
            sampled_pos = random.choices(all_positive_records, k=pos_per_batch)

        # Sample negative records (with replacement if needed)
        if len(all_negative_records) >= neg_per_batch:
            sampled_neg = random.sample(all_negative_records, neg_per_batch)
        else:
            # With replacement if not enough unique samples
            sampled_neg = random.choices(all_negative_records, k=neg_per_batch)

        # Combine and shuffle batch
        batch_records = sampled_pos + sampled_neg
        random.shuffle(batch_records)

        # Write batch to file
        batch_filename = f"batch_{batch_idx:06d}.tfrecord"
        batch_path = batches_dir / batch_filename
        write_tfrecord_batch(batch_records, str(batch_path))

        batch_files.append(str(batch_path))
        batch_metadata.append({
            'batch_id': batch_idx,
            'filename': batch_filename,
            'num_samples': len(batch_records),
            'positive_samples': pos_per_batch,
            'negative_samples': neg_per_batch,
            'epoch': batch_idx // batches_per_epoch
        })

    # Save metadata
    metadata = {
        'batch_size': batch_size,
        'pos_ratio': pos_ratio,
        'pos_per_batch': pos_per_batch,
        'neg_per_batch': neg_per_batch,
        'total_batches': len(batch_files),
        'batches_per_epoch': batches_per_epoch,
        'num_epochs': num_epochs,
        'total_positive_samples': len(all_positive_records),
        'total_negative_samples': len(all_negative_records),
        'batches_dir': str(batches_dir),
        'batch_files': [os.path.basename(f) for f in batch_files],
        'creation_seed': seed
    }

    metadata_path = batches_dir / "batches_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save detailed batch info
    batch_details_path = batches_dir / "batch_details.json"
    with open(batch_details_path, 'w') as f:
        json.dump(batch_metadata, f, indent=2)

    print(f"\n✅ Successfully created {len(batch_files):,} stratified batch files")
    print(f"📁 Saved to: {batches_dir}")
    print(f"📊 Metadata saved to: {metadata_path}")
    print(f"🔍 Each batch: {batch_size} samples ({pos_per_batch} pos, {neg_per_batch} neg)")
    print(f"📈 {batches_per_epoch} batches per epoch × {num_epochs} epochs")

    return str(batches_dir), metadata


def main():
    """Main function to create stratified batches from command line arguments."""
    parser = argparse.ArgumentParser(description="Create stratified batch files from TFRecord data")

    parser.add_argument("--tfrecord_dir", type=str, required=True,
                       help="Directory containing train/positive_samples.tfrecord and train/negative_samples_*.tfrecord")
    parser.add_argument("--batch_size", type=int, default=1024,
                       help="Number of samples per batch (default: 1024)")
    parser.add_argument("--pos_ratio", type=float, default=0.02,
                       help="Ratio of positive samples per batch (default: 0.02)")
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of epochs worth of batches to create (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: tfrecord_dir)")

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.tfrecord_dir

    # Find input files
    tfrecord_dir = Path(args.tfrecord_dir)
    positive_file = tfrecord_dir / "train" / "positive_samples.tfrecord"
    negative_pattern = tfrecord_dir / "train" / "negative_samples_*.tfrecord"

    # Check if files exist
    if not positive_file.exists():
        print(f"❌ Error: Positive samples file not found: {positive_file}")
        sys.exit(1)

    negative_files = list(tfrecord_dir.glob("train/negative_samples_*.tfrecord"))
    if not negative_files:
        print(f"❌ Error: No negative samples files found in: {tfrecord_dir / 'train'}")
        sys.exit(1)

    print(f"📂 Input directory: {tfrecord_dir}")
    print(f"✅ Found positive file: {positive_file}")
    print(f"✅ Found {len(negative_files)} negative files")

    # Verify files are readable
    print("\n🔍 Verifying input files...")
    pos_count = count_records_in_tfrecord(str(positive_file))
    print(f"   Positive samples: {pos_count:,}")

    total_neg_count = 0
    for neg_file in negative_files:
        neg_count = count_records_in_tfrecord(str(neg_file))
        total_neg_count += neg_count
        print(f"   {neg_file.name}: {neg_count:,}")

    print(f"   Total negative samples: {total_neg_count:,}")
    print(f"   Class ratio: {pos_count / (pos_count + total_neg_count):.4f} positive")

    if pos_count == 0 or total_neg_count == 0:
        print("❌ Error: No samples found in input files")
        sys.exit(1)

    # Create stratified batches
    try:
        batches_dir, metadata = create_stratified_batches(
            positive_files=[str(positive_file)],
            negative_files=[str(f) for f in negative_files],
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            pos_ratio=args.pos_ratio,
            num_epochs=args.num_epochs,
            seed=args.seed
        )

        print(f"\n🎉 Successfully created stratified batches!")
        print(f"📁 Use this directory in training: {batches_dir}")

    except Exception as e:
        print(f"❌ Error creating batches: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()