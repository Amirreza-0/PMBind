#!/usr/bin/env python
"""
simple_infer.py: Direct parquet-based inference script for PMBind model.
This script performs inference on a single data file and generates visualizations
of the results and the model's latent space.
"""
import csv
import sys
import re
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pyarrow.parquet as pq
import json
import os
from tqdm import tqdm
import argparse
from functools import lru_cache

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score)

# Local imports (ensure these utility scripts are in the same directory or Python path)
from utils import (seq_to_onehot, get_embed_key, NORM_TOKEN, MASK_TOKEN, PAD_TOKEN, PAD_VALUE, MASK_VALUE,
                   clean_key, seq_to_blossom62, BLOSUM62, AA, OHE_to_seq_single,
                   masked_categorical_crossentropy, split_y_true_y_pred, get_mhc_seq_class2, load_metadata,
                   load_metadata, load_embedding_table, normalize_embedding_tf, AA_BLOSUM,
                   load_embedding_db, apply_dynamic_masking, min_max_norm, log_norm_zscore, _preprocess_df_chunk
                   )
from models import pmbind_multitask_v8 as pmbind
from visualizations import (_analyze_latents, visualize_attention_weights,
                            visualize_per_allele_metrics, visualize_anchor_positions_and_binding_pockets)

# --- Globals & Configuration ---
mixed_precision = True  # Enable mixed precision for significant speedup

if mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f'Mixed precision enabled: {policy}')

# Paths - load from json config
with open("infer_paths.json", "r") as f:
    config = json.load(f)

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
EMB_NORM_METHOD = "robust_zscore"  # clip_norm1000, None

## --- Constant lookup table for on-the-fly feature creation ---
_blosum_vectors = [np.asarray(BLOSUM62[aa], dtype=np.float32) for aa in AA_BLOSUM]
vector_len = _blosum_vectors[0].shape[0]
_blosum_vectors.append(np.full((vector_len,), PAD_VALUE, dtype=np.float32))
BLOSUM62_TABLE = tf.constant(np.stack(_blosum_vectors, axis=0), dtype=tf.float32)

# Load EMB_DB_p for DataGenerator
print("\nLoading MHC embedding lookup table for DataGenerator...")
EMB_DB_p = load_embedding_db(embedding_table_path)


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
        return tensor_data


def preprocess_df(df, seq_map, embed_map):
    df['_cleaned_key'] = df.apply(
        lambda r: r.get('mhc_embedding_key', r['allele'].replace(' ', '').replace('*', '').replace(':', '')), axis=1)
    df['_emb_key'] = df['_cleaned_key'].apply(lambda k: get_embed_key(clean_key(k), embed_map))
    if MHC_CLASS == 2:
        df['_mhc_seq'] = df['_cleaned_key'].apply(get_mhc_seq_class2)
    else:
        df['_mhc_seq'] = df['_emb_key'].apply(lambda k: seq_map.get(get_embed_key(clean_key(k), seq_map), ''))
    return df


# --- Visualization Functions ---
def visualize_inference_results(df, true_labels, scores, out_dir, name):
    """Generate and save evaluation plots, including score distributions."""
    print(f"Generating visualizations for {name} set...")

    # Confusion Matrix
    plt.figure()
    cm = confusion_matrix(true_labels, df["prediction_label"])
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"Confusion Matrix on {name} Set")
    plt.savefig(os.path.join(out_dir, f"cm_{name}.png"))
    plt.close()

    # ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(true_labels, scores)
    auc = roc_auc_score(true_labels, scores)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve on {name} Set')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(out_dir, f"roc_{name}.png"))
    plt.close()

    # Prediction Score Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='prediction_score', hue='assigned_label', kde=True, bins=50, palette="viridis")
    plt.title(f'Prediction Score Distribution on {name} Set')
    plt.xlabel('Predicted Probability (Class 1)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, f"dist_{name}.png"))
    plt.close()


def generate_sample_data_for_viz(df, max_pep_len, max_mhc_len, seq_map, embed_map):
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    gen = OptimizedDataGenerator(df, seq_map, embed_map, max_pep_len, max_mhc_len, 1, apply_masking=False,
                                    normalization_method=EMB_NORM_METHOD, class_weights_dict=None)
    data = gen[0]
    data = {k: v.numpy() for k, v in data.items()}
    pep_seq = OHE_to_seq_single(data['pep_ohe_target'][0], gap=True)
    mhc_seq = OHE_to_seq_single(data['mhc_ohe_target'][0], gap=True)
    return data, pep_seq, mhc_seq


def generate_attention_visualizations(model, df, seq_map, embed_map, max_pep_len, max_mhc_len, out_dir, name):
    """Generate attention weight visualizations for a sample of the data."""
    print("\n--- Generating attention weight visualizations ---")
    attention_out_dir = os.path.join(out_dir, "attention_weights")
    os.makedirs(attention_out_dir, exist_ok=True)

    # Select a few representative samples for attention visualization
    num_samples_to_viz = min(1, len(df))
    sample_indices = np.random.choice(len(df), num_samples_to_viz, replace=False) if len(df) > 5 else range(len(df))

    for i, sample_idx in enumerate(sample_indices):
        sample_row = df.iloc[sample_idx]
        sample_data, pep_seq, mhc_seq = generate_sample_data_for_viz(sample_row, max_pep_len, max_mhc_len, seq_map,
                                                                     embed_map)

        # Convert to tensors for model input
        model_input = {k: tf.convert_to_tensor(v) for k, v in sample_data.items()}

        # Get model outputs including attention weights
        outputs = model(model_input, training=False)

        if "attn_weights" in outputs:
            attn_weights = outputs["attn_weights"]

            # Create sample-specific output directory
            sample_out_dir = os.path.join(attention_out_dir, f"sample_{sample_idx}")
            os.makedirs(sample_out_dir, exist_ok=True)

            # Visualize attention weights
            # visualize_attention_weights(
            #     attn_weights=attn_weights,
            #     peptide_seq=pep_seq,
            #     mhc_seq=mhc_seq,
            #     max_pep_len=max_pep_len,
            #     max_mhc_len=max_mhc_len,
            #     out_dir=sample_out_dir,
            #     sample_idx=0,  # Since we're processing one sample at a time
            #     head_idx=None,  # Average across heads
            #     save_all_heads=True  # Save individual head visualizations
            # )

            # Visualize anchor positions and binding pockets using max pooling
            anchor_pocket_results = visualize_anchor_positions_and_binding_pockets(
                attn_weights=attn_weights,
                peptide_seq=pep_seq,
                mhc_seq=mhc_seq,
                max_pep_len=max_pep_len,
                max_mhc_len=max_mhc_len,
                out_dir=sample_out_dir,
                sample_idx=0,
                pooling='max',  # Show both max and mean pooling
                top_k_anchors=4,
                top_k_pockets=15
            )

            # Also save sample information
            with open(os.path.join(sample_out_dir, "sample_info.txt"), "w") as f:
                f.write(f"Sample Index: {sample_idx}\n")
                f.write(f"Peptide: {pep_seq}\n")
                f.write(f"MHC Allele: {sample_row['allele']}\n")
                f.write(f"MHC Sequence: {mhc_seq}\n")
                if 'assigned_label' in sample_row:
                    f.write(f"True Label: {sample_row['assigned_label']}\n")
                if 'prediction_score' in sample_row:
                    f.write(f"Prediction Score: {sample_row['prediction_score']:.4f}\n")
                if 'prediction_label' in sample_row:
                    f.write(f"Predicted Label: {sample_row['prediction_label']}\n")

                # Add anchor positions and binding pocket information
                f.write(f"\n--- Detected Anchor Positions ---\n")
                for anchor in anchor_pocket_results['anchor_positions']:
                    f.write(
                        f"  Position {anchor['position_1indexed']}: {anchor['amino_acid']} (score: {anchor['score']:.4f})\n")

                f.write(f"\n--- Detected MHC Binding Pockets ---\n")
                for pocket in anchor_pocket_results['binding_pockets']:
                    f.write(
                        f"  Position {pocket['position_1indexed']}: {pocket['amino_acid']} (score: {pocket['score']:.4f})\n")

            print(f"✓ Attention visualizations saved for sample {sample_idx}")
        else:
            print(f"Warning: Model does not output attention weights for sample {sample_idx}")

    print(f"✓ All attention weight visualizations saved to {attention_out_dir}")


def run_visualizations(df, latents_pooled, latents_seq, out_dir, name, max_pep_len, max_mhc_len, seq_map, embed_map,
                       source_col=None):
    print("\nGenerating latent space visualizations...")
    os.makedirs(out_dir, exist_ok=True)
    alleles = df['allele'].apply(clean_key).astype('category')
    unique_alleles = alleles.cat.categories
    highlight_mask = (df[source_col] == 'test').values if source_col and source_col in df.columns else None
    num_to_highlight = min(5, len(unique_alleles))
    np.random.seed(999)
    random_alleles_to_highlight = np.random.choice(unique_alleles, num_to_highlight, replace=False).tolist()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_alleles))) if len(unique_alleles) <= 20 else plt.cm.viridis(
        np.linspace(0, 1, len(unique_alleles)))
    allele_color_map = {allele: color for allele, color in zip(unique_alleles, colors)}
    for l_type, l_data in [("seq", latents_seq), ("pooled", latents_pooled)]:
        _analyze_latents(latents=l_data, df=df, alleles=alleles, allele_color_map=allele_color_map,
                         random_alleles_to_highlight=random_alleles_to_highlight, latent_type=l_type, out_dir=out_dir,
                         dataset_name=name, highlight_mask=highlight_mask)

    print("\n--- Generating supplementary plots (inputs, masks) ---")
    sample_idx = 0
    sample_row = df.iloc[sample_idx]
    sample_data, pep_seq, _ = generate_sample_data_for_viz(sample_row, max_pep_len, max_mhc_len, seq_map, embed_map)
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle(f'Input Data Sample (Peptide: {pep_seq}, Allele: {df.iloc[0]["allele"]})', fontsize=16)
    sns.heatmap(sample_data['pep_blossom62'][0].T, ax=axes[0, 0], cmap='gray_r')
    axes[0, 0].set_title('Peptide Input (BLOSUM62)')
    sns.heatmap(sample_data['pep_mask'][0][np.newaxis, :], ax=axes[0, 1], cmap='viridis', cbar=False)
    axes[0, 1].set_title('Peptide Mask')
    sns.heatmap(sample_data['mhc_emb'][0].T, ax=axes[1, 0], cmap='viridis')
    axes[1, 0].set_title('MHC Input (Embedding)')
    sns.heatmap(sample_data['mhc_mask'][0][np.newaxis, :], ax=axes[1, 1], cmap='viridis', cbar=False)
    axes[1, 1].set_title('MHC Mask')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, "input_mask_sample.png"))
    plt.close()
    print("✓ Input and mask plots saved.")


# --- Main Inference Function ---

def infer(model_weights_path, config_path, df_path, out_dir, name,
          allele_seq_path, embedding_key_path, embedding_npz_path,
          batch_size=10, source_col=None, allow_cache=False):
    global EMB_DB, MHC_CLASS, ESM_DIM
    os.makedirs(out_dir, exist_ok=True)

    # Load run configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract model parameters from config
    MHC_CLASS = config.get("MHC_CLASS", 1)

    # Load embeddings first to get ESM_DIM
    EMB_DB = load_embedding_db(embedding_npz_path)
    ESM_DIM = int(next(iter(EMB_DB.values())).shape[1])
    print(f"Loaded embedding database with ESM_DIM={ESM_DIM}")

    # Load sequence mappings (needed for preprocessing)
    print("Loading sequence mappings...")
    seq_map = {clean_key(k): v for k, v in
               pd.read_csv(allele_seq_path, index_col="allele")["mhc_sequence"].to_dict().items()}
    embed_map = pd.read_csv(embedding_key_path, index_col="key")["mhc_sequence"].to_dict()
    print("✓ Sequence mappings loaded")

    # Load model - handle both .keras and weights-only files
    if model_weights_path.endswith('.keras'):
        print(f"Loading full model from {model_weights_path}...")
        model = keras.models.load_model(model_weights_path, compile=False)
        print("✓ Full model loaded successfully.")

        # Extract actual dimensions from the loaded model's input shapes
        max_pep_len = model.input[0].shape[1]  # pep_blossom62 input
        max_mhc_len = model.input[2].shape[1]  # mhc_emb input
        embed_dim = model.get_layer("pep_dense_embed").output.shape[-1]

        # Try to infer number of heads from attention layer if possible
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
        # For weights-only files, use config values and build model
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

        # Build model with dummy data
        df_dummy = pq.ParquetFile(df_path).read().to_pandas().head(1)
        df_dummy = preprocess_df(df_dummy, seq_map, embed_map)
        dummy_gen = OptimizedDataGenerator(df_dummy, seq_map, embed_map, max_pep_len, max_mhc_len, 1,
                                           apply_masking=False, normalization_method=EMB_NORM_METHOD, class_weights_dict=None)
        model(dummy_gen[0], training=False)
        model.load_weights(model_weights_path)
        print("✓ Model weights loaded successfully.")

    print(
        f"Model configuration: MHC_CLASS={MHC_CLASS}, max_pep_len={max_pep_len}, max_mhc_len={max_mhc_len}, embed_dim={embed_dim}, heads={heads}")

    # Load and preprocess input data
    print(f"Loading data from {df_path}...")
    df_infer = pq.ParquetFile(df_path).read().to_pandas()
    print(f"✓ Loaded {len(df_infer)} samples")

    print("Preprocessing data...")
    df_infer = preprocess_df(df_infer, seq_map, embed_map)
    print("✓ Data preprocessing complete")

    # Define paths for cached latents
    latents_seq_path = os.path.join(out_dir, f"latents_seq_{name}.h5")
    latents_pooled_path = os.path.join(out_dir, f"latents_pooled_{name}.h5")

    # Run inference or load cached results
    if not (os.path.exists(latents_pooled_path) and allow_cache):
        print("Generating predictions and latents...")
        infer_gen = OptimizedDataGenerator(df_infer, seq_map, embed_map, max_pep_len, max_mhc_len, batch_size,
                                           apply_masking=False, normalization_method=EMB_NORM_METHOD, class_weights_dict=None)
        all_predictions, all_labels = [], []
        with h5py.File(latents_seq_path, 'w') as f_seq, h5py.File(latents_pooled_path, 'w') as f_pooled:
            d_seq = f_seq.create_dataset('latents', shape=(len(df_infer), max_pep_len + max_mhc_len, embed_dim),
                                         dtype='float32')
            d_pooled = f_pooled.create_dataset('latents', shape=(len(df_infer), max_pep_len + max_mhc_len + embed_dim),
                                               dtype='float32')
            for i, batch in enumerate(tqdm(infer_gen, desc=f"Inference on {name}", file=sys.stdout)):
                outputs = model(batch, training=False)
                all_predictions.append(outputs["cls_ypred"].numpy())
                all_labels.append(batch["labels"].numpy())
                start, end = i * batch_size, i * batch_size + len(batch["labels"])
                d_seq[start:end] = outputs["latent_seq"].numpy()
                d_pooled[start:end] = outputs["latent_vector"].numpy()
        all_predictions, all_labels = np.concatenate(all_predictions).squeeze(), np.concatenate(all_labels).squeeze()
        df_infer["prediction_score"] = all_predictions
        df_infer["prediction_label"] = (all_predictions >= 0.5).astype(int)
        df_infer.to_csv(os.path.join(out_dir, f"inference_results_{name}.csv"), index=False)
        print(f"✓ Inference results saved.")
    else:
        print("Loading cached predictions and latents...")
        df_infer = pd.read_csv(os.path.join(out_dir, f"inference_results_{name}.csv"))
        all_labels, all_predictions = df_infer["assigned_label"].values, df_infer["prediction_score"].values

    # Load latents
    with h5py.File(latents_seq_path, 'r') as f:
        latents_seq = f['latents'][:]
    with h5py.File(latents_pooled_path, 'r') as f:
        latents_pooled = f['latents'][:]

    # Generate evaluation visualizations if labels are available
    if "assigned_label" in df_infer.columns:
        visualize_inference_results(df_infer, all_labels, all_predictions, out_dir, name)

        # Generate per-allele AUC visualization
        print("\n--- Generating per-allele AUC visualizations ---")
        visualize_per_allele_metrics(
            df=df_infer,
            true_labels=all_labels,
            predictions=all_predictions,
            out_dir=os.path.join(out_dir, "visualizations"),
            dataset_name=name,
            allele_col='allele',
            min_samples=10,
            top_n=None  # Show all alleles with sufficient samples
        )

        # Calculate and save metrics summary
        auc = roc_auc_score(all_labels, all_predictions) if len(np.unique(all_labels)) > 1 else -1
        ap = average_precision_score(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, df_infer["prediction_label"])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        summary_path = os.path.join(os.path.dirname(out_dir), "inference_summary.csv")
        file_exists = os.path.isfile(summary_path)
        with open(summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['dataset', 'num_samples', 'AUC', 'AP', 'TP', 'TN', 'FP', 'FN'])
            writer.writerow([name, len(df_infer), f"{auc:.4f}", f"{ap:.4f}", tp, tn, fp, fn])
        print(f"✓ Summary updated.")

    # Run latent space visualizations
    run_visualizations(df_infer, latents_pooled, latents_seq, os.path.join(out_dir, "visualizations"), name,
                       max_pep_len, max_mhc_len, seq_map, embed_map, source_col)

    # Generate attention weight visualizations
    generate_attention_visualizations(model, df_infer, seq_map, embed_map, max_pep_len, max_mhc_len,
                                      os.path.join(out_dir, "visualizations"), name)

    # Save sequence predictions for first 10 samples
    pred_samples_df = df_infer.head(10)
    pred_data_gen = OptimizedDataGenerator(pred_samples_df, seq_map, embed_map, max_pep_len, max_mhc_len, 10,
                                           apply_masking=False, normalization_method=EMB_NORM_METHOD, class_weights_dict=None)
    pred_batch = pred_data_gen[0]
    model_outputs = model(pred_batch, training=False)

    if "pep_ytrue_ypred" in model_outputs and "mhc_ytrue_ypred" in model_outputs:
        pep_true, pep_pred_ohe = split_y_true_y_pred(model_outputs["pep_ytrue_ypred"].numpy())
        mhc_true, mhc_pred_ohe = split_y_true_y_pred(model_outputs["mhc_ytrue_ypred"].numpy())
        pep_masks_np = pred_batch["pep_mask"].numpy()
        mhc_masks_np = pred_batch["mhc_mask"].numpy()

        pred_list = []
        for i in range(10):
            allele = clean_key(pred_samples_df.iloc[i]['allele'])
            original_peptide_full = OHE_to_seq_single(pep_true[i], gap=True).replace("X", "-")
            predicted_peptide_full = OHE_to_seq_single(pep_pred_ohe[i], gap=True).replace("X", "-")
            pep_valid_mask = (pep_masks_np[i] != PAD_TOKEN) & (np.array(list(original_peptide_full)) != '-')
            original_peptide = "".join(np.array(list(original_peptide_full))[pep_valid_mask])
            predicted_peptide = "".join(np.array(list(predicted_peptide_full))[pep_valid_mask])

            original_mhc_full = OHE_to_seq_single(mhc_true[i], gap=True).replace("X", "-")
            predicted_mhc_full = OHE_to_seq_single(mhc_pred_ohe[i], gap=True).replace("X", "-")
            mhc_valid_mask = (mhc_masks_np[i] != PAD_TOKEN) & (np.array(list(original_mhc_full)) != '-')
            original_mhc = "".join(np.array(list(original_mhc_full))[mhc_valid_mask])
            predicted_mhc = "".join(np.array(list(predicted_mhc_full))[mhc_valid_mask])

            pred_list.append({
                "sample_index": int(pred_samples_df.index[i]), "allele": allele,
                "original_peptide": original_peptide, "predicted_peptide": predicted_peptide,
                "original_mhc": original_mhc, "predicted_mhc": predicted_mhc
            })

        predictions_df = pd.DataFrame(pred_list)
        predictions_output_path = os.path.join(out_dir, f"sequence_predictions_{name}.csv")
        predictions_df.to_csv(predictions_output_path, index=False)
        print(f"✓ Sequence predictions saved to {predictions_output_path}")
    else:
        print("Model does not output reconstruction predictions - skipping sequence prediction CSV")


def main():
    parser = argparse.ArgumentParser(description="PMBind Simple Inference Script")
    parser.add_argument("--model_weights_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--df_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--allele_seq_path", required=True)
    parser.add_argument("--embedding_key_path", required=True)
    parser.add_argument("--embedding_npz_path", required=True)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--source_col")
    parser.add_argument("--allow_cache", action='store_true', help="Use cached inference results if available")
    args = parser.parse_args()

    # Configure GPU memory growth
    if gpus := tf.config.list_physical_devices('GPU'):
        try:
            [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
            print(f"✓ Memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU error: {e}")

    infer(**vars(args))


if __name__ == "__main__":
    main()