#!/usr/bin/env python
"""
accumulative_attention.py: Calculate and visualize accumulative attention scores for 9-mer peptides.

This script performs accumulative attention analysis on binder 9-mer peptides for a specific MHC allele.
For each position in the peptide, it substitutes all 20 amino acids and measures the attention scores,
then visualizes the results as a heatmap showing which amino acids at which positions receive the most attention.

Usage:
    python accumulative_attention.py \
        --model_weights_path <path_to_model> \
        --config_path <path_to_config> \
        --df_path <path_to_parquet> \
        --out_dir <output_directory> \
        --allele <MHC_allele> \
        --allele_seq_path <path_to_allele_sequences> \
        --embedding_key_path <path_to_embedding_keys> \
        --embedding_npz_path <path_to_embeddings> \
        [--batch_size <batch_size>] \
        [--max_peptides <max_number_of_peptides>]

"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pyarrow.parquet as pq
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports - reuse functions from other scripts
from utils import (
    seq_to_onehot, get_embed_key, clean_key, seq_to_blossom62,
    NORM_TOKEN, MASK_TOKEN, PAD_TOKEN, PAD_VALUE,
    load_embedding_db, AA
)
from models import pmbind_multitask_plus as pmbind

# Constants
AA_20 = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
PEP_LEN = 9  # Fixed 9-mer peptides

# Global variables (will be set in main)
EMB_DB_p = None
ESM_DIM = None
MHC_CLASS = 1
EMB_NORM_METHOD = "robust_zscore"


class OptimizedDataGenerator(keras.utils.Sequence):
    """Optimized data generator for inference without masking."""

    def __init__(self, df, seq_map, embed_map, max_pep_len, max_mhc_len, batch_size,
                 normalization_method=None):
        super().__init__()
        self.df = df
        self.seq_map = seq_map
        self.embed_map = embed_map
        self.max_pep_len = max_pep_len
        self.max_mhc_len = max_mhc_len
        self.batch_size = batch_size
        self.long_mer_arr = df['long_mer'].to_numpy()
        self.emb_key_arr = df['_emb_key'].to_numpy()
        self.cleaned_key_arr = df['_cleaned_key'].to_numpy()
        self.mhc_seq_arr = df['_mhc_seq'].to_numpy()
        self.label_arr = df['assigned_label'].to_numpy() if 'assigned_label' in df.columns else np.zeros(len(df))
        self.indices = np.arange(len(df))
        self.normalization_method = normalization_method

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.indices))
        return self._generate_batch(self.indices[start_idx:end_idx])

    def _get_embedding(self, emb_key, cleaned_key):
        """Get embedding and apply normalization."""
        try:
            emb = EMB_DB_p[emb_key]
        except KeyError:
            raise KeyError(f"embedding not found for emb_key {emb_key}, cleaned_key: {cleaned_key}")
        return self._normalize_embedding(emb)

    def _normalize_embedding(self, emb):
        """Apply robust normalization to handle extreme values."""
        if self.normalization_method == "robust_zscore":
            mean = emb.mean()
            std = emb.std()
            emb_norm = (emb - mean) / (std + 1e-8)
            emb_norm = np.clip(emb_norm, -5, 5)
            return emb_norm
        else:
            return emb

    def _generate_batch(self, batch_indices):
        n = len(batch_indices)
        data = {
            "pep_blossom62": np.zeros((n, self.max_pep_len, 23), np.float32),
            "pep_mask": np.full((n, self.max_pep_len), PAD_TOKEN, dtype=np.float32),
            "mhc_emb": np.zeros((n, self.max_mhc_len, ESM_DIM), np.float32),
            "mhc_mask": np.full((n, self.max_mhc_len), PAD_TOKEN, dtype=np.float32),
            "pep_ohe_target": np.zeros((n, self.max_pep_len, 21), np.float32),
            "mhc_ohe_target": np.zeros((n, self.max_mhc_len, 21), np.float32),
            "labels": np.zeros((n, 1), np.float32),
            "sample_weights": np.ones((n, 1), np.float32)
        }

        for i, master_idx in enumerate(batch_indices):
            pep_seq = self.long_mer_arr[master_idx].upper()
            emb_key = self.emb_key_arr[master_idx]
            cleaned_key = self.cleaned_key_arr[master_idx]
            mhc_seq = self.mhc_seq_arr[master_idx]
            pep_len = len(pep_seq)

            # Peptide features
            data["pep_blossom62"][i] = seq_to_blossom62(pep_seq, max_seq_len=self.max_pep_len)
            data["pep_ohe_target"][i] = seq_to_onehot(pep_seq, max_seq_len=self.max_pep_len)
            data["pep_mask"][i, :pep_len] = NORM_TOKEN

            # MHC features
            emb = self._get_embedding(emb_key, cleaned_key)
            L = emb.shape[0]
            data["mhc_emb"][i, :L] = emb
            data["mhc_ohe_target"][i] = seq_to_onehot(mhc_seq, max_seq_len=self.max_mhc_len)
            is_padding = np.all(data["mhc_ohe_target"][i, :] == 0, axis=-1)
            data["mhc_mask"][i, ~is_padding] = NORM_TOKEN

            data["labels"][i, 0] = float(self.label_arr[master_idx])

        # Convert to tensors
        tensor_data = {k: tf.convert_to_tensor(v) for k, v in data.items()}
        return tensor_data


def preprocess_df(df, seq_map, embed_map):
    """Preprocess dataframe to add MHC sequence and embedding keys."""
    df['_cleaned_key'] = df.apply(
        lambda r: r.get('mhc_embedding_key', r['allele'].replace(' ', '').replace('*', '').replace(':', '')),
        axis=1
    )
    df['_emb_key'] = df['_cleaned_key'].apply(lambda k: get_embed_key(clean_key(k), embed_map))
    df['_mhc_seq'] = df['_emb_key'].apply(lambda k: seq_map.get(k, ''))
    return df


def load_model_and_config(model_weights_path, config_path, embedding_npz_path):
    """Load model and configuration."""
    global ESM_DIM, MHC_CLASS, EMB_DB_p

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    MHC_CLASS = config.get("MHC_CLASS", 1)

    # Load embeddings
    print("Loading MHC embedding database...")
    EMB_DB_p = load_embedding_db(embedding_npz_path)
    ESM_DIM = int(next(iter(EMB_DB_p.values())).shape[1])
    print(f"✓ Loaded embedding database with ESM_DIM={ESM_DIM}")

    # Load model
    if model_weights_path.endswith('.keras'):
        print(f"Loading full model from {model_weights_path}...")
        model = keras.models.load_model(model_weights_path, compile=False)
        print("✓ Full model loaded successfully.")

        # Extract dimensions from loaded model
        max_pep_len = model.input[0].shape[1]
        max_mhc_len = model.input[2].shape[1]
        embed_dim = model.get_layer("pep_dense_embed").output.shape[-1]

        try:
            attn_layer = model.get_layer("pmhc_2d_masked_attention2")
            heads = attn_layer.heads
        except:
            try:
                attn_layer = model.get_layer("pmhc_2d_masked_attention")
                heads = attn_layer.heads
            except:
                heads = config.get("HEADS", 8)
    else:
        # For weights-only files, build model from config
        max_pep_len = config.get("MAX_PEP_LEN", 15)
        max_mhc_len = config.get("MAX_MHC_LEN", 400)
        embed_dim = config.get("EMBED_DIM", 32)
        heads = config.get("HEADS", 8)

        print(f"Building model and loading weights from {model_weights_path}...")
        model = pmbind(
            max_pep_len=max_pep_len,
            max_mhc_len=max_mhc_len,
            emb_dim=embed_dim,
            heads=heads,
            noise_std=0.0,
            drop_out_rate=0.0,
            l2_reg=config.get("L2_REG", 0.0),
            ESM_dim=ESM_DIM
        )
        model.load_weights(model_weights_path)
        print("✓ Model weights loaded successfully.")

    print(f"Model configuration: max_pep_len={max_pep_len}, max_mhc_len={max_mhc_len}, "
          f"embed_dim={embed_dim}, heads={heads}")

    return model, config, max_pep_len, max_mhc_len


def calculate_attention_score_batch(model, peptides, position, allele, seq_map, embed_map,
                                    max_pep_len, max_mhc_len, batch_size):
    """
    Calculate attention scores for a batch of peptides at a specific position.

    Args:
        model: The trained PMBind model
        peptides: List of peptide sequences
        position: Position in the peptide to analyze (0-indexed)
        allele: MHC allele
        seq_map: Sequence mapping dictionary
        embed_map: Embedding key mapping dictionary
        max_pep_len: Maximum peptide length
        max_mhc_len: Maximum MHC length
        batch_size: Batch size for inference

    Returns:
        Array of attention scores for each peptide at the specified position
    """
    # Create temporary dataframe for these peptides
    temp_df = pd.DataFrame({
        'long_mer': peptides,
        'allele': [allele] * len(peptides),
        'assigned_label': [1] * len(peptides)  # Dummy label
    })

    # Preprocess
    temp_df = preprocess_df(temp_df, seq_map, embed_map)

    # Create data generator
    gen = OptimizedDataGenerator(
        temp_df, seq_map, embed_map, max_pep_len, max_mhc_len,
        batch_size=batch_size, normalization_method=EMB_NORM_METHOD
    )

    # Run inference
    all_attn_scores = []
    for batch in gen:
        outputs = model(batch, training=False)
        attn_weights = outputs["attn_weights"].numpy()  # Shape: (B, heads, P+M, P+M)

        # Extract attention scores for the specified position
        # Sum across all heads and MHC positions to get overall attention for peptide position
        # attn_weights[:, :, position, :] gives attention from position to all other positions
        # We sum across heads (axis=1) and all target positions (axis=-1) to get total attention
        scores = np.sum(attn_weights[:, :, position, :], axis=(1, 2))
        all_attn_scores.append(scores)

    return np.concatenate(all_attn_scores)


def calculate_accumulative_attention(model, peptides, allele, seq_map, embed_map,
                                     max_pep_len, max_mhc_len, batch_size):
    """
    Calculate accumulative attention scores for all positions and amino acid substitutions.

    Args:
        model: The trained PMBind model
        peptides: List of 9-mer peptide sequences (all binders for the MHC)
        allele: MHC allele
        seq_map: Sequence mapping dictionary
        embed_map: Embedding key mapping dictionary
        max_pep_len: Maximum peptide length
        max_mhc_len: Maximum MHC length
        batch_size: Batch size for inference

    Returns:
        Dictionary with:
            - 'matrix': 9x20 matrix of accumulative attention scores
            - 'position_scores': List of dicts mapping amino acid to score for each position
    """
    print(f"\nCalculating accumulative attention scores for {len(peptides)} peptides...")
    print(f"MHC Allele: {allele}")

    # Convert peptides to numpy array for efficient manipulation
    peptides_array = np.array([list(pep) for pep in peptides])
    all_position_scores = []

    # For each position in the 9-mer
    for pos in tqdm(range(PEP_LEN), desc="Processing positions", file=sys.stdout):
        pos_attn_scores = {}

        # For each amino acid substitution
        for sub_aa in AA_20:
            # Create mutated peptides by replacing position pos with sub_aa
            mutated_array = peptides_array.copy()
            mutated_array[:, pos] = sub_aa
            mutated_peptides = [''.join(pep) for pep in mutated_array]

            # Calculate attention scores for this substitution
            scores = calculate_attention_score_batch(
                model, mutated_peptides, pos, allele, seq_map, embed_map,
                max_pep_len, max_mhc_len, batch_size
            )

            # Sum across all peptides to get accumulative score
            accumulative_score = np.sum(scores)
            pos_attn_scores[sub_aa] = accumulative_score

        all_position_scores.append(pos_attn_scores)
        print(f"  Position {pos}: {pos_attn_scores}")

    # Convert to matrix format (9 positions x 20 amino acids)
    attn_matrix = np.array([
        [all_position_scores[pos][aa] for aa in AA_20]
        for pos in range(PEP_LEN)
    ])

    return {
        'matrix': attn_matrix,
        'position_scores': all_position_scores
    }


def visualize_accumulative_attention(attn_matrix, out_dir, allele):
    """
    Generate and save heatmap visualization of accumulative attention scores.

    Args:
        attn_matrix: 9x20 matrix of accumulative attention scores
        out_dir: Output directory
        allele: MHC allele name
    """
    os.makedirs(out_dir, exist_ok=True)

    # Create heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        attn_matrix,
        annot=True,
        fmt='.2f',
        xticklabels=list(AA_20),
        yticklabels=[f"Pos {i+1}" for i in range(PEP_LEN)],
        cmap="YlGnBu",
        cbar_kws={'label': 'Accumulative Attention Score'}
    )
    plt.title(f"Accumulative Attention Scores Heatmap\nMHC Allele: {allele}", fontsize=16)
    plt.xlabel("Amino Acid Substitution", fontsize=12)
    plt.ylabel("Peptide Position", fontsize=12)
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(out_dir, f"accumulative_attention_{clean_key(allele)}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Heatmap saved to: {output_path}")

    # Also save as a more compact version without annotations
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        attn_matrix,
        annot=False,
        xticklabels=list(AA_20),
        yticklabels=[f"Pos {i+1}" for i in range(PEP_LEN)],
        cmap="YlGnBu",
        cbar_kws={'label': 'Accumulative Attention Score'}
    )
    plt.title(f"Accumulative Attention Scores Heatmap\nMHC Allele: {allele}", fontsize=14)
    plt.xlabel("Amino Acid Substitution", fontsize=11)
    plt.ylabel("Peptide Position", fontsize=11)
    plt.tight_layout()

    output_path_compact = os.path.join(out_dir, f"accumulative_attention_{clean_key(allele)}_compact.png")
    plt.savefig(output_path_compact, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Compact heatmap saved to: {output_path_compact}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate and visualize accumulative attention scores for 9-mer peptides"
    )
    parser.add_argument("--model_weights_path", required=True, help="Path to model weights or .keras file")
    parser.add_argument("--config_path", required=True, help="Path to model configuration JSON")
    parser.add_argument("--df_path", required=True, help="Path to parquet file with peptide data")
    parser.add_argument("--out_dir", required=True, help="Output directory for results")
    parser.add_argument("--allele", required=True, help="MHC allele to analyze (e.g., HLA-A*02:01)")
    parser.add_argument("--allele_seq_path", required=True, help="Path to allele sequences CSV")
    parser.add_argument("--embedding_key_path", required=True, help="Path to embedding keys CSV")
    parser.add_argument("--embedding_npz_path", required=True, help="Path to embeddings NPZ file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference (default: 32)")
    parser.add_argument("--max_peptides", type=int, default=None,
                       help="Maximum number of peptides to use (default: all)")

    args = parser.parse_args()

    # Configure GPU
    if gpus := tf.config.list_physical_devices('GPU'):
        try:
            [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
            print(f"✓ Memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU error: {e}")

    # Enable mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f'✓ Mixed precision enabled: {policy}')

    # Load model and configuration
    model, config, max_pep_len, max_mhc_len = load_model_and_config(
        args.model_weights_path, args.config_path, args.embedding_npz_path
    )

    # Load sequence mappings
    print("\nLoading sequence mappings...")
    seq_map = {clean_key(k): v for k, v in
               pd.read_csv(args.allele_seq_path, index_col="allele")["mhc_sequence"].to_dict().items()}
    embed_map = pd.read_csv(args.embedding_key_path, index_col="key")["mhc_sequence"].to_dict()
    print("✓ Sequence mappings loaded")

    # Load and filter data
    print(f"\nLoading data from {args.df_path}...")
    df = pq.ParquetFile(args.df_path).read().to_pandas()
    print(f"✓ Loaded {len(df)} total samples")

    # Filter for specified allele and 9-mer binders
    allele_cleaned = clean_key(args.allele)
    df['_allele_cleaned'] = df['allele'].apply(clean_key)
    df_filtered = df[
        (df['_allele_cleaned'] == allele_cleaned) &
        (df['long_mer'].str.len() == PEP_LEN)
    ]

    # Further filter for binders if label column exists
    if 'assigned_label' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['assigned_label'] == 1]
        print(f"✓ Filtered to {len(df_filtered)} 9-mer binders for allele {args.allele}")
    else:
        print(f"✓ Filtered to {len(df_filtered)} 9-mer peptides for allele {args.allele}")

    if len(df_filtered) == 0:
        print(f"ERROR: No 9-mer peptides found for allele {args.allele}")
        sys.exit(1)

    # Limit number of peptides if specified
    if args.max_peptides and len(df_filtered) > args.max_peptides:
        df_filtered = df_filtered.sample(n=args.max_peptides, random_state=42)
        print(f"✓ Randomly selected {args.max_peptides} peptides for analysis")

    peptides = df_filtered['long_mer'].tolist()

    # Calculate accumulative attention
    results = calculate_accumulative_attention(
        model, peptides, args.allele, seq_map, embed_map,
        max_pep_len, max_mhc_len, args.batch_size
    )

    # Save results
    os.makedirs(args.out_dir, exist_ok=True)

    # Save matrix as CSV
    matrix_df = pd.DataFrame(
        results['matrix'],
        columns=list(AA_20),
        index=[f"Pos_{i+1}" for i in range(PEP_LEN)]
    )
    csv_path = os.path.join(args.out_dir, f"accumulative_attention_{clean_key(args.allele)}.csv")
    matrix_df.to_csv(csv_path)
    print(f"\n✓ Attention matrix saved to: {csv_path}")

    # Save detailed scores as JSON
    json_path = os.path.join(args.out_dir, f"accumulative_attention_{clean_key(args.allele)}.json")
    with open(json_path, 'w') as f:
        json.dump({
            'allele': args.allele,
            'num_peptides': len(peptides),
            'position_scores': results['position_scores']
        }, f, indent=2)
    print(f"✓ Detailed scores saved to: {json_path}")

    # Generate visualizations
    visualize_accumulative_attention(results['matrix'], args.out_dir, args.allele)

    print(f"\n✓ All results saved to: {args.out_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
