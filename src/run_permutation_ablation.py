#!/usr/bin/env python
"""
run_permutation_ablation.py: Permutation and Ablation Studies for PMBind Model

This script performs comprehensive permutation and ablation studies to understand:
1. Feature importance through permutation testing
2. Component importance through ablation testing
3. Model interpretability and robustness

Permutation Studies:
- Peptide sequence permutation
- MHC embedding permutation
- Individual BLOSUM62 feature permutation
- Anchor position permutation
- Per-position amino acid permutation

Ablation Studies:
- Attention head ablation
- Input feature ablation (peptide/MHC)
- Position-specific ablation
- Embedding dimension ablation

Usage:
    python src/run_permutation_ablation.py \
        --model_weights_path results/model/best_model.weights.h5 \
        --config_path results/model/run_config.json \
        --df_path data/test_set.parquet \
        --out_dir results/permutation_ablation \
        --allele_seq_path data/alleles/aligned_PMGen_class_1.csv \
        --embedding_key_path results/ESM/mhc1_encodings.csv \
        --embedding_npz_path results/ESM/mhc1_encodings.npz \
        --n_permutations 10 \
        --batch_size 256
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow import keras
import pyarrow.parquet as pq
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

# Local imports
from utils import (seq_to_onehot, get_embed_key, NORM_TOKEN, MASK_TOKEN, PAD_TOKEN,
                   clean_key, seq_to_blossom62, load_embedding_db)
from models import pmbind_multitask_plus as pmbind
from infer import OptimizedDataGenerator, preprocess_df
from visualizations import (visualize_permutation_importance, visualize_ablation_results,
                           visualize_position_importance, visualize_feature_importance_heatmap)

# Enable mixed precision
mixed_precision = True
if mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f'Mixed precision enabled: {policy}')


class PermutationStudy:
    """Performs permutation-based feature importance analysis."""

    def __init__(self, model, data_generator, df, metric='auc'):
        self.model = model
        self.data_generator = data_generator
        self.df = df
        self.metric = metric
        self.baseline_score = None

    def calculate_baseline(self):
        """Calculate baseline performance without permutation."""
        print("\n=== Calculating Baseline Performance ===")
        predictions, labels = [], []

        for batch in tqdm(self.data_generator, desc="Baseline inference"):
            outputs = self.model(batch, training=False)
            predictions.append(outputs["cls_ypred"].numpy())
            labels.append(batch["labels"].numpy())

        predictions = np.concatenate(predictions).squeeze()
        labels = np.concatenate(labels).squeeze()

        if self.metric == 'auc':
            self.baseline_score = roc_auc_score(labels, predictions)
        elif self.metric == 'accuracy':
            self.baseline_score = accuracy_score(labels, (predictions >= 0.5).astype(int))
        elif self.metric == 'ap':
            self.baseline_score = average_precision_score(labels, predictions)

        print(f"Baseline {self.metric.upper()}: {self.baseline_score:.4f}")
        return self.baseline_score, predictions, labels

    def permute_peptide_sequences(self, n_permutations=10):
        """Permute entire peptide sequences."""
        print("\n=== Permuting Peptide Sequences ===")
        results = []

        for i in range(n_permutations):
            print(f"Permutation {i+1}/{n_permutations}")
            predictions, labels = [], []

            for batch_idx, batch in enumerate(tqdm(self.data_generator, desc=f"Perm {i+1}")):
                # Create permuted batch
                permuted_batch = {k: v.numpy() for k, v in batch.items()}

                # Permute peptide BLOSUM62 features
                batch_size = permuted_batch['pep_blossom62'].shape[0]
                perm_indices = np.random.permutation(batch_size)
                permuted_batch['pep_blossom62'] = permuted_batch['pep_blossom62'][perm_indices]
                permuted_batch['pep_mask'] = permuted_batch['pep_mask'][perm_indices]
                permuted_batch['pep_ohe_target'] = permuted_batch['pep_ohe_target'][perm_indices]

                # Convert back to tensors
                permuted_batch = {k: tf.convert_to_tensor(v) for k, v in permuted_batch.items()}

                outputs = self.model(permuted_batch, training=False)
                predictions.append(outputs["cls_ypred"].numpy())
                labels.append(batch["labels"].numpy())

            predictions = np.concatenate(predictions).squeeze()
            labels = np.concatenate(labels).squeeze()

            if self.metric == 'auc':
                score = roc_auc_score(labels, predictions)
            elif self.metric == 'accuracy':
                score = accuracy_score(labels, (predictions >= 0.5).astype(int))
            elif self.metric == 'ap':
                score = average_precision_score(labels, predictions)

            importance = self.baseline_score - score
            results.append({'permutation': i, 'score': score, 'importance': importance})
            print(f"  Score: {score:.4f}, Importance: {importance:.4f}")

        return pd.DataFrame(results)

    def permute_mhc_embeddings(self, n_permutations=10):
        """Permute MHC embeddings."""
        print("\n=== Permuting MHC Embeddings ===")
        results = []

        for i in range(n_permutations):
            print(f"Permutation {i+1}/{n_permutations}")
            predictions, labels = [], []

            for batch in tqdm(self.data_generator, desc=f"Perm {i+1}"):
                permuted_batch = {k: v.numpy() for k, v in batch.items()}

                # Permute MHC embeddings
                batch_size = permuted_batch['mhc_emb'].shape[0]
                perm_indices = np.random.permutation(batch_size)
                permuted_batch['mhc_emb'] = permuted_batch['mhc_emb'][perm_indices]
                permuted_batch['mhc_mask'] = permuted_batch['mhc_mask'][perm_indices]
                permuted_batch['mhc_ohe_target'] = permuted_batch['mhc_ohe_target'][perm_indices]

                permuted_batch = {k: tf.convert_to_tensor(v) for k, v in permuted_batch.items()}

                outputs = self.model(permuted_batch, training=False)
                predictions.append(outputs["cls_ypred"].numpy())
                labels.append(batch["labels"].numpy())

            predictions = np.concatenate(predictions).squeeze()
            labels = np.concatenate(labels).squeeze()

            if self.metric == 'auc':
                score = roc_auc_score(labels, predictions)
            elif self.metric == 'accuracy':
                score = accuracy_score(labels, (predictions >= 0.5).astype(int))
            elif self.metric == 'ap':
                score = average_precision_score(labels, predictions)

            importance = self.baseline_score - score
            results.append({'permutation': i, 'score': score, 'importance': importance})
            print(f"  Score: {score:.4f}, Importance: {importance:.4f}")

        return pd.DataFrame(results)

    def permute_per_position(self, n_permutations=5):
        """Permute each peptide position independently."""
        print("\n=== Permuting Per-Position ===")
        max_pep_len = self.data_generator.max_pep_len
        position_results = []

        for pos in range(max_pep_len):
            print(f"\nPosition {pos+1}/{max_pep_len}")
            pos_scores = []

            for perm_idx in range(n_permutations):
                predictions, labels = [], []

                for batch in tqdm(self.data_generator, desc=f"Pos {pos+1}, Perm {perm_idx+1}", leave=False):
                    permuted_batch = {k: v.numpy() for k, v in batch.items()}

                    # Permute specific position
                    batch_size = permuted_batch['pep_blossom62'].shape[0]
                    perm_indices = np.random.permutation(batch_size)
                    permuted_batch['pep_blossom62'][:, pos, :] = permuted_batch['pep_blossom62'][perm_indices, pos, :]

                    permuted_batch = {k: tf.convert_to_tensor(v) for k, v in permuted_batch.items()}

                    outputs = self.model(permuted_batch, training=False)
                    predictions.append(outputs["cls_ypred"].numpy())
                    labels.append(batch["labels"].numpy())

                predictions = np.concatenate(predictions).squeeze()
                labels = np.concatenate(labels).squeeze()

                if self.metric == 'auc':
                    score = roc_auc_score(labels, predictions)
                elif self.metric == 'accuracy':
                    score = accuracy_score(labels, (predictions >= 0.5).astype(int))
                elif self.metric == 'ap':
                    score = average_precision_score(labels, predictions)

                pos_scores.append(score)

            avg_score = np.mean(pos_scores)
            std_score = np.std(pos_scores)
            importance = self.baseline_score - avg_score

            position_results.append({
                'position': pos,
                'avg_score': avg_score,
                'std_score': std_score,
                'importance': importance
            })
            print(f"  Position {pos}: Avg Score={avg_score:.4f}±{std_score:.4f}, Importance={importance:.4f}")

        return pd.DataFrame(position_results)

    def permute_blosum_features(self, n_permutations=5):
        """Permute individual BLOSUM62 feature dimensions."""
        print("\n=== Permuting BLOSUM62 Features ===")
        n_features = 23  # BLOSUM62 has 23 dimensions
        feature_results = []

        for feat_idx in range(n_features):
            print(f"\nFeature {feat_idx+1}/{n_features}")
            feat_scores = []

            for perm_idx in range(n_permutations):
                predictions, labels = [], []

                for batch in tqdm(self.data_generator, desc=f"Feature {feat_idx+1}, Perm {perm_idx+1}", leave=False):
                    permuted_batch = {k: v.numpy() for k, v in batch.items()}

                    # Permute specific BLOSUM feature across all positions
                    batch_size = permuted_batch['pep_blossom62'].shape[0]
                    perm_indices = np.random.permutation(batch_size)
                    permuted_batch['pep_blossom62'][:, :, feat_idx] = permuted_batch['pep_blossom62'][perm_indices, :, feat_idx]

                    permuted_batch = {k: tf.convert_to_tensor(v) for k, v in permuted_batch.items()}

                    outputs = self.model(permuted_batch, training=False)
                    predictions.append(outputs["cls_ypred"].numpy())
                    labels.append(batch["labels"].numpy())

                predictions = np.concatenate(predictions).squeeze()
                labels = np.concatenate(labels).squeeze()

                if self.metric == 'auc':
                    score = roc_auc_score(labels, predictions)
                elif self.metric == 'accuracy':
                    score = accuracy_score(labels, (predictions >= 0.5).astype(int))
                elif self.metric == 'ap':
                    score = average_precision_score(labels, predictions)

                feat_scores.append(score)

            avg_score = np.mean(feat_scores)
            std_score = np.std(feat_scores)
            importance = self.baseline_score - avg_score

            feature_results.append({
                'feature_idx': feat_idx,
                'avg_score': avg_score,
                'std_score': std_score,
                'importance': importance
            })
            print(f"  Feature {feat_idx}: Avg Score={avg_score:.4f}±{std_score:.4f}, Importance={importance:.4f}")

        return pd.DataFrame(feature_results)


class AblationStudy:
    """Performs ablation-based component importance analysis."""

    def __init__(self, model, data_generator, df, metric='auc'):
        self.model = model
        self.data_generator = data_generator
        self.df = df
        self.metric = metric
        self.baseline_score = None

    def calculate_baseline(self):
        """Calculate baseline performance."""
        print("\n=== Calculating Baseline Performance (Ablation) ===")
        predictions, labels = [], []

        for batch in tqdm(self.data_generator, desc="Baseline inference"):
            outputs = self.model(batch, training=False)
            predictions.append(outputs["cls_ypred"].numpy())
            labels.append(batch["labels"].numpy())

        predictions = np.concatenate(predictions).squeeze()
        labels = np.concatenate(labels).squeeze()

        if self.metric == 'auc':
            self.baseline_score = roc_auc_score(labels, predictions)
        elif self.metric == 'accuracy':
            self.baseline_score = accuracy_score(labels, (predictions >= 0.5).astype(int))
        elif self.metric == 'ap':
            self.baseline_score = average_precision_score(labels, predictions)

        print(f"Baseline {self.metric.upper()}: {self.baseline_score:.4f}")
        return self.baseline_score, predictions, labels

    def ablate_peptide_input(self):
        """Zero out peptide input completely."""
        print("\n=== Ablating Peptide Input ===")
        predictions, labels = [], []

        for batch in tqdm(self.data_generator, desc="Peptide ablation"):
            ablated_batch = {k: v.numpy() for k, v in batch.items()}

            # Zero out peptide features
            ablated_batch['pep_blossom62'] = np.zeros_like(ablated_batch['pep_blossom62'])

            ablated_batch = {k: tf.convert_to_tensor(v) for k, v in ablated_batch.items()}

            outputs = self.model(ablated_batch, training=False)
            predictions.append(outputs["cls_ypred"].numpy())
            labels.append(batch["labels"].numpy())

        predictions = np.concatenate(predictions).squeeze()
        labels = np.concatenate(labels).squeeze()

        if self.metric == 'auc':
            score = roc_auc_score(labels, predictions)
        elif self.metric == 'accuracy':
            score = accuracy_score(labels, (predictions >= 0.5).astype(int))
        elif self.metric == 'ap':
            score = average_precision_score(labels, predictions)

        importance = self.baseline_score - score
        print(f"Peptide Ablation - Score: {score:.4f}, Importance: {importance:.4f}")

        return {'component': 'peptide_input', 'score': score, 'importance': importance}

    def ablate_mhc_input(self):
        """Zero out MHC input completely."""
        print("\n=== Ablating MHC Input ===")
        predictions, labels = [], []

        for batch in tqdm(self.data_generator, desc="MHC ablation"):
            ablated_batch = {k: v.numpy() for k, v in batch.items()}

            # Zero out MHC embeddings
            ablated_batch['mhc_emb'] = np.zeros_like(ablated_batch['mhc_emb'])

            ablated_batch = {k: tf.convert_to_tensor(v) for k, v in ablated_batch.items()}

            outputs = self.model(ablated_batch, training=False)
            predictions.append(outputs["cls_ypred"].numpy())
            labels.append(batch["labels"].numpy())

        predictions = np.concatenate(predictions).squeeze()
        labels = np.concatenate(labels).squeeze()

        if self.metric == 'auc':
            score = roc_auc_score(labels, predictions)
        elif self.metric == 'accuracy':
            score = accuracy_score(labels, (predictions >= 0.5).astype(int))
        elif self.metric == 'ap':
            score = average_precision_score(labels, predictions)

        importance = self.baseline_score - score
        print(f"MHC Ablation - Score: {score:.4f}, Importance: {importance:.4f}")

        return {'component': 'mhc_input', 'score': score, 'importance': importance}

    def ablate_per_position(self):
        """Ablate each peptide position independently."""
        print("\n=== Ablating Per-Position ===")
        max_pep_len = self.data_generator.max_pep_len
        position_results = []

        for pos in range(max_pep_len):
            print(f"\nAblating Position {pos+1}/{max_pep_len}")
            predictions, labels = [], []

            for batch in tqdm(self.data_generator, desc=f"Pos {pos+1} ablation"):
                ablated_batch = {k: v.numpy() for k, v in batch.items()}

                # Zero out specific position
                ablated_batch['pep_blossom62'][:, pos, :] = 0

                ablated_batch = {k: tf.convert_to_tensor(v) for k, v in ablated_batch.items()}

                outputs = self.model(ablated_batch, training=False)
                predictions.append(outputs["cls_ypred"].numpy())
                labels.append(batch["labels"].numpy())

            predictions = np.concatenate(predictions).squeeze()
            labels = np.concatenate(labels).squeeze()

            if self.metric == 'auc':
                score = roc_auc_score(labels, predictions)
            elif self.metric == 'accuracy':
                score = accuracy_score(labels, (predictions >= 0.5).astype(int))
            elif self.metric == 'ap':
                score = average_precision_score(labels, predictions)

            importance = self.baseline_score - score
            position_results.append({
                'position': pos,
                'score': score,
                'importance': importance
            })
            print(f"  Position {pos}: Score={score:.4f}, Importance={importance:.4f}")

        return pd.DataFrame(position_results)

    def ablate_anchor_positions(self):
        """Ablate known anchor positions (P2 and P-omega)."""
        print("\n=== Ablating Anchor Positions ===")
        results = []

        # Ablate P2 (position 1, 0-indexed)
        print("\nAblating P2 (position 2)")
        predictions, labels = [], []

        for batch in tqdm(self.data_generator, desc="P2 ablation"):
            ablated_batch = {k: v.numpy() for k, v in batch.items()}
            ablated_batch['pep_blossom62'][:, 1, :] = 0  # P2 is index 1
            ablated_batch = {k: tf.convert_to_tensor(v) for k, v in ablated_batch.items()}

            outputs = self.model(ablated_batch, training=False)
            predictions.append(outputs["cls_ypred"].numpy())
            labels.append(batch["labels"].numpy())

        predictions = np.concatenate(predictions).squeeze()
        labels = np.concatenate(labels).squeeze()

        if self.metric == 'auc':
            score = roc_auc_score(labels, predictions)
        elif self.metric == 'accuracy':
            score = accuracy_score(labels, (predictions >= 0.5).astype(int))
        elif self.metric == 'ap':
            score = average_precision_score(labels, predictions)

        importance = self.baseline_score - score
        results.append({'anchor': 'P2', 'score': score, 'importance': importance})
        print(f"P2 Ablation - Score: {score:.4f}, Importance: {importance:.4f}")

        # Ablate P-omega (last position)
        print("\nAblating P-omega (C-terminal)")
        predictions, labels = [], []

        for batch in tqdm(self.data_generator, desc="P-omega ablation"):
            ablated_batch = {k: v.numpy() for k, v in batch.items()}
            # Find actual last position per sample based on mask
            masks = ablated_batch['pep_mask']
            for i in range(masks.shape[0]):
                valid_positions = np.where(masks[i] != PAD_TOKEN)[0]
                if len(valid_positions) > 0:
                    last_pos = valid_positions[-1]
                    ablated_batch['pep_blossom62'][i, last_pos, :] = 0

            ablated_batch = {k: tf.convert_to_tensor(v) for k, v in ablated_batch.items()}

            outputs = self.model(ablated_batch, training=False)
            predictions.append(outputs["cls_ypred"].numpy())
            labels.append(batch["labels"].numpy())

        predictions = np.concatenate(predictions).squeeze()
        labels = np.concatenate(labels).squeeze()

        if self.metric == 'auc':
            score = roc_auc_score(labels, predictions)
        elif self.metric == 'accuracy':
            score = accuracy_score(labels, (predictions >= 0.5).astype(int))
        elif self.metric == 'ap':
            score = average_precision_score(labels, predictions)

        importance = self.baseline_score - score
        results.append({'anchor': 'P-omega', 'score': score, 'importance': importance})
        print(f"P-omega Ablation - Score: {score:.4f}, Importance: {importance:.4f}")

        # Ablate both
        print("\nAblating Both Anchors (P2 + P-omega)")
        predictions, labels = [], []

        for batch in tqdm(self.data_generator, desc="Both anchors ablation"):
            ablated_batch = {k: v.numpy() for k, v in batch.items()}
            ablated_batch['pep_blossom62'][:, 1, :] = 0  # P2
            # P-omega
            masks = ablated_batch['pep_mask']
            for i in range(masks.shape[0]):
                valid_positions = np.where(masks[i] != PAD_TOKEN)[0]
                if len(valid_positions) > 0:
                    last_pos = valid_positions[-1]
                    ablated_batch['pep_blossom62'][i, last_pos, :] = 0

            ablated_batch = {k: tf.convert_to_tensor(v) for k, v in ablated_batch.items()}

            outputs = self.model(ablated_batch, training=False)
            predictions.append(outputs["cls_ypred"].numpy())
            labels.append(batch["labels"].numpy())

        predictions = np.concatenate(predictions).squeeze()
        labels = np.concatenate(labels).squeeze()

        if self.metric == 'auc':
            score = roc_auc_score(labels, predictions)
        elif self.metric == 'accuracy':
            score = accuracy_score(labels, (predictions >= 0.5).astype(int))
        elif self.metric == 'ap':
            score = average_precision_score(labels, predictions)

        importance = self.baseline_score - score
        results.append({'anchor': 'Both', 'score': score, 'importance': importance})
        print(f"Both Anchors Ablation - Score: {score:.4f}, Importance: {importance:.4f}")

        return pd.DataFrame(results)


def load_model_and_data(model_weights_path, config_path, df_path, allele_seq_path,
                       embedding_key_path, embedding_npz_path, batch_size, emb_norm_method="robust_zscore"):
    """Load model and prepare data generator."""

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    mhc_class = config.get("MHC_CLASS", 1)

    # Load embeddings
    global EMB_DB_p, ESM_DIM
    EMB_DB_p = load_embedding_db(embedding_npz_path)
    ESM_DIM = int(next(iter(EMB_DB_p.values())).shape[1])
    print(f"Loaded embedding database with ESM_DIM={ESM_DIM}")

    # Load sequence mappings
    print("Loading sequence mappings...")
    seq_map = {clean_key(k): v for k, v in
               pd.read_csv(allele_seq_path, index_col="allele")["mhc_sequence"].to_dict().items()}
    embed_map = pd.read_csv(embedding_key_path, index_col="key")["mhc_sequence"].to_dict()
    print("✓ Sequence mappings loaded")

    # Load model
    if model_weights_path.endswith('.keras'):
        print(f"Loading full model from {model_weights_path}...")
        model = keras.models.load_model(model_weights_path, compile=False)
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
        # Build model from config and load weights
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

        # Build with dummy data
        df_dummy = pq.ParquetFile(df_path).read().to_pandas().head(1)
        df_dummy = preprocess_df(df_dummy, seq_map, embed_map)
        dummy_gen = OptimizedDataGenerator(df_dummy, seq_map, embed_map, max_pep_len, max_mhc_len, 1,
                                           apply_masking=False, normalization_method=emb_norm_method)
        model(dummy_gen[0], training=False)
        model.load_weights(model_weights_path)
        print("✓ Model weights loaded successfully.")

    print(f"Model config: max_pep_len={max_pep_len}, max_mhc_len={max_mhc_len}, embed_dim={embed_dim}, heads={heads}")

    # Load and preprocess data
    print(f"Loading data from {df_path}...")
    df = pq.ParquetFile(df_path).read().to_pandas()
    print(f"✓ Loaded {len(df)} samples")

    print("Preprocessing data...")
    df = preprocess_df(df, seq_map, embed_map)
    print("✓ Data preprocessing complete")

    # Create data generator
    data_gen = OptimizedDataGenerator(df, seq_map, embed_map, max_pep_len, max_mhc_len, batch_size,
                                      apply_masking=False, normalization_method=emb_norm_method)

    return model, data_gen, df, max_pep_len, max_mhc_len


def main():
    parser = argparse.ArgumentParser(description="PMBind Permutation and Ablation Studies")
    parser.add_argument("--model_weights_path", required=True, help="Path to model weights")
    parser.add_argument("--config_path", required=True, help="Path to run config JSON")
    parser.add_argument("--df_path", required=True, help="Path to test data parquet")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--allele_seq_path", required=True, help="Path to allele sequences")
    parser.add_argument("--embedding_key_path", required=True, help="Path to embedding key CSV")
    parser.add_argument("--embedding_npz_path", required=True, help="Path to embedding NPZ")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--n_permutations", type=int, default=10, help="Number of permutations")
    parser.add_argument("--metric", type=str, default="auc", choices=["auc", "accuracy", "ap"],
                       help="Evaluation metric")
    parser.add_argument("--skip_permutation", action="store_true", help="Skip permutation studies")
    parser.add_argument("--skip_ablation", action="store_true", help="Skip ablation studies")
    parser.add_argument("--emb_norm_method", type=str, default="robust_zscore",
                       help="Embedding normalization method")

    args = parser.parse_args()

    # Configure GPU
    if gpus := tf.config.list_physical_devices('GPU'):
        try:
            [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
            print(f"✓ Memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU error: {e}")

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model and data
    print("\n" + "="*60)
    print("PERMUTATION AND ABLATION STUDIES")
    print("="*60)

    model, data_gen, df, max_pep_len, max_mhc_len = load_model_and_data(
        args.model_weights_path, args.config_path, args.df_path,
        args.allele_seq_path, args.embedding_key_path, args.embedding_npz_path,
        args.batch_size, args.emb_norm_method
    )

    # ========== PERMUTATION STUDIES ==========
    if not args.skip_permutation:
        print("\n" + "="*60)
        print("PERMUTATION STUDIES")
        print("="*60)

        perm_study = PermutationStudy(model, data_gen, df, metric=args.metric)
        baseline_score, baseline_preds, baseline_labels = perm_study.calculate_baseline()

        # 1. Peptide sequence permutation
        pep_perm_results = perm_study.permute_peptide_sequences(args.n_permutations)
        pep_perm_results.to_csv(os.path.join(args.out_dir, "permutation_peptide.csv"), index=False)

        # 2. MHC embedding permutation
        mhc_perm_results = perm_study.permute_mhc_embeddings(args.n_permutations)
        mhc_perm_results.to_csv(os.path.join(args.out_dir, "permutation_mhc.csv"), index=False)

        # 3. Per-position permutation
        pos_perm_results = perm_study.permute_per_position(n_permutations=5)
        pos_perm_results.to_csv(os.path.join(args.out_dir, "permutation_positions.csv"), index=False)

        # 4. BLOSUM feature permutation
        blosum_perm_results = perm_study.permute_blosum_features(n_permutations=5)
        blosum_perm_results.to_csv(os.path.join(args.out_dir, "permutation_blosum.csv"), index=False)

        # Visualize permutation results
        print("\nGenerating permutation visualizations...")
        visualize_permutation_importance(
            pep_results=pep_perm_results,
            mhc_results=mhc_perm_results,
            pos_results=pos_perm_results,
            blosum_results=blosum_perm_results,
            baseline_score=baseline_score,
            out_dir=os.path.join(args.out_dir, "visualizations"),
            metric=args.metric
        )

        # Position-specific visualization
        visualize_position_importance(
            pos_results=pos_perm_results,
            baseline_score=baseline_score,
            max_pep_len=max_pep_len,
            out_dir=os.path.join(args.out_dir, "visualizations"),
            study_type="permutation",
            metric=args.metric
        )

        # BLOSUM feature heatmap
        visualize_feature_importance_heatmap(
            blosum_results=blosum_perm_results,
            out_dir=os.path.join(args.out_dir, "visualizations")
        )

    # ========== ABLATION STUDIES ==========
    if not args.skip_ablation:
        print("\n" + "="*60)
        print("ABLATION STUDIES")
        print("="*60)

        ablation_study = AblationStudy(model, data_gen, df, metric=args.metric)
        baseline_score, baseline_preds, baseline_labels = ablation_study.calculate_baseline()

        # 1. Input ablation
        results = []
        results.append(ablation_study.ablate_peptide_input())
        results.append(ablation_study.ablate_mhc_input())
        input_ablation_df = pd.DataFrame(results)
        input_ablation_df.to_csv(os.path.join(args.out_dir, "ablation_inputs.csv"), index=False)

        # 2. Per-position ablation
        pos_ablation_results = ablation_study.ablate_per_position()
        pos_ablation_results.to_csv(os.path.join(args.out_dir, "ablation_positions.csv"), index=False)

        # 3. Anchor position ablation
        anchor_ablation_results = ablation_study.ablate_anchor_positions()
        anchor_ablation_results.to_csv(os.path.join(args.out_dir, "ablation_anchors.csv"), index=False)

        # Visualize ablation results
        print("\nGenerating ablation visualizations...")
        visualize_ablation_results(
            input_results=input_ablation_df,
            pos_results=pos_ablation_results,
            anchor_results=anchor_ablation_results,
            baseline_score=baseline_score,
            out_dir=os.path.join(args.out_dir, "visualizations"),
            metric=args.metric
        )

        # Position-specific visualization
        visualize_position_importance(
            pos_results=pos_ablation_results,
            baseline_score=baseline_score,
            max_pep_len=max_pep_len,
            out_dir=os.path.join(args.out_dir, "visualizations"),
            study_type="ablation",
            metric=args.metric
        )

    # Save summary
    summary = {
        'model_weights_path': args.model_weights_path,
        'config_path': args.config_path,
        'df_path': args.df_path,
        'n_samples': len(df),
        'metric': args.metric,
        'baseline_score': float(baseline_score),
        'n_permutations': args.n_permutations,
        'max_pep_len': int(max_pep_len),
        'max_mhc_len': int(max_mhc_len)
    }

    with open(os.path.join(args.out_dir, "study_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("STUDIES COMPLETED!")
    print("="*60)
    print(f"Results saved to: {args.out_dir}")
    print(f"Baseline {args.metric.upper()}: {baseline_score:.4f}")


if __name__ == "__main__":
    main()