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
                   clean_key, seq_to_blossom62, BLOSUM62, AMINO_ACID_VOCAB, PAD_INDEX, AA, OHE_to_seq_single,
                   masked_categorical_crossentropy, split_y_true_y_pred)
from models import pmbind_multitask_modified as pmbind
from visualizations import _analyze_latents, visualize_attention_weights

# --- Globals & Configuration ---
EMB_DB: np.lib.npyio.NpzFile | None = None
MHC_CLASS = 1
ESM_DIM = 1536


# --- Data Generator and Helper Functions ---

def load_embedding_db(npz_path: str):
    """Load embedding database with memory mapping."""
    return np.load(npz_path, mmap_mode="r")


class OptimizedDataGenerator(keras.utils.Sequence):
    """Optimized data generator for inference."""

    def __init__(self, df, seq_map, embed_map, max_pep_len, max_mhc_len, batch_size):
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
        self.label_arr = df['assigned_label'].to_numpy()
        self.indices = np.arange(len(df))

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        start_idx, end_idx = index * self.batch_size, min((index + 1) * self.batch_size, len(self.indices))
        return self._generate_batch(self.indices[start_idx:end_idx])

    @lru_cache(maxsize=128)
    def _get_embedding(self, emb_key, cleaned_key):
        if MHC_CLASS == 2:
            parts = cleaned_key.split('_')
            if len(parts) >= 2:
                k1, k2 = get_embed_key(clean_key(parts[0]), self.embed_map), get_embed_key(clean_key(parts[1]),
                                                                                           self.embed_map)
                return np.concatenate([EMB_DB[k1], EMB_DB[k2]], axis=0)
        return EMB_DB[emb_key]

    def _generate_batch(self, batch_indices):
        n = len(batch_indices)
        data = {"pep_blossom62": np.zeros((n, self.max_pep_len, 23), np.float32),
                "pep_mask": np.full((n, self.max_pep_len), PAD_TOKEN, dtype=np.float32),
                "mhc_emb": np.zeros((n, self.max_mhc_len, ESM_DIM), np.float32),
                "mhc_mask": np.full((n, self.max_mhc_len), PAD_TOKEN, dtype=np.float32),
                "pep_ohe_target": np.zeros((n, self.max_pep_len, 21), np.float32),
                "mhc_ohe_target": np.zeros((n, self.max_mhc_len, 21), np.float32), "labels": np.zeros((n, 1), np.int32)}
        for i, master_idx in enumerate(batch_indices):
            pep_seq, emb_key, cleaned_key, mhc_seq = self.long_mer_arr[master_idx].upper(), self.emb_key_arr[
                master_idx], self.cleaned_key_arr[master_idx], self.mhc_seq_arr[master_idx]
            pep_len = len(pep_seq)
            data["pep_blossom62"][i] = seq_to_blossom62(pep_seq, max_seq_len=self.max_pep_len)
            data["pep_ohe_target"][i] = seq_to_onehot(pep_seq, max_seq_len=self.max_pep_len)
            data["pep_mask"][i, :pep_len] = NORM_TOKEN
            emb = self._get_embedding(emb_key, cleaned_key)
            L = emb.shape[0]
            data["mhc_emb"][i, :L] = emb
            data["mhc_emb"][i, L:, :] = PAD_VALUE
            data["mhc_mask"][i, ~np.all(data["mhc_emb"][i] == PAD_VALUE, axis=-1)] = NORM_TOKEN
            data["mhc_ohe_target"][i] = seq_to_onehot(mhc_seq, max_seq_len=self.max_mhc_len)
            data["labels"][i, 0] = int(self.label_arr[master_idx])
        return {k: tf.convert_to_tensor(v) for k, v in data.items()}


def preprocess_df(df, seq_map, embed_map):
    df['_cleaned_key'] = df.apply(
        lambda r: r.get('mhc_embedding_key', r['allele'].replace(' ', '').replace('*', '').replace(':', '')), axis=1)
    df['_emb_key'] = df['_cleaned_key'].apply(lambda k: get_embed_key(clean_key(k), embed_map))
    if MHC_CLASS == 2:
        def get_mhc_seq_class2(key):
            parts = key.split('_')
            return seq_map.get(get_embed_key(clean_key(parts[0]), seq_map), '') + seq_map.get(
                get_embed_key(clean_key(parts[1]), seq_map), '') if len(parts) >= 2 else ''

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


def generate_sample_data_for_viz(df, idx, max_pep_len, max_mhc_len, seq_map, embed_map):
    sample_row = df.iloc[idx]
    pep_seq, emb_key, cleaned_key, mhc_seq = sample_row['long_mer'].upper(), sample_row['_emb_key'], sample_row[
        '_cleaned_key'], sample_row['_mhc_seq']
    data = {"pep_blossom62": np.zeros((1, max_pep_len, 23), np.float32),
            "pep_mask": np.full((1, max_pep_len), PAD_TOKEN, dtype=np.float32),
            "mhc_emb": np.zeros((1, max_mhc_len, ESM_DIM), np.float32),
            "mhc_mask": np.full((1, max_mhc_len), PAD_TOKEN, dtype=np.float32)}
    pep_len = len(pep_seq)
    data["pep_blossom62"][0] = seq_to_blossom62(pep_seq, max_seq_len=max_pep_len)
    data["pep_mask"][0, :pep_len] = NORM_TOKEN
    gen = OptimizedDataGenerator(df, seq_map, embed_map, max_pep_len, max_mhc_len, 1)
    emb = gen._get_embedding(emb_key, cleaned_key)
    L = emb.shape[0]
    data["mhc_emb"][0, :L] = emb
    data["mhc_emb"][0, L:, :] = PAD_VALUE
    data["mhc_mask"][0, ~np.all(data["mhc_emb"][0] == PAD_VALUE, axis=-1)] = NORM_TOKEN
    return data, pep_seq, mhc_seq


def generate_attention_visualizations(model, df, seq_map, embed_map, max_pep_len, max_mhc_len, out_dir, name):
    """Generate attention weight visualizations for a sample of the data."""
    print("\n--- Generating attention weight visualizations ---")
    attention_out_dir = os.path.join(out_dir, "attention_weights")
    os.makedirs(attention_out_dir, exist_ok=True)

    # Select a few representative samples for attention visualization
    num_samples_to_viz = min(5, len(df))
    sample_indices = np.random.choice(len(df), num_samples_to_viz, replace=False) if len(df) > 5 else range(len(df))

    for i, sample_idx in enumerate(sample_indices):
        sample_row = df.iloc[sample_idx]
        sample_data, pep_seq, mhc_seq = generate_sample_data_for_viz(df, sample_idx, max_pep_len, max_mhc_len, seq_map, embed_map)

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
            visualize_attention_weights(
                attn_weights=attn_weights,
                peptide_seq=pep_seq,
                mhc_seq=mhc_seq,
                max_pep_len=max_pep_len,
                max_mhc_len=max_mhc_len,
                out_dir=sample_out_dir,
                sample_idx=0,  # Since we're processing one sample at a time
                head_idx=None,  # Average across heads
                save_all_heads=True  # Save individual head visualizations
            )

            # Also save sample information
            with open(os.path.join(sample_out_dir, "sample_info.txt"), "w") as f:
                f.write(f"Sample Index: {sample_idx}\n")
                f.write(f"Peptide: {pep_seq}\n")
                f.write(f"MHC Allele: {sample_row['allele']}\n")
                f.write(f"MHC Sequence: {mhc_seq[:50]}{'...' if len(mhc_seq) > 50 else ''}\n")
                if 'assigned_label' in sample_row:
                    f.write(f"True Label: {sample_row['assigned_label']}\n")
                if 'prediction_score' in sample_row:
                    f.write(f"Prediction Score: {sample_row['prediction_score']:.4f}\n")
                if 'prediction_label' in sample_row:
                    f.write(f"Predicted Label: {sample_row['prediction_label']}\n")

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
    sample_data, pep_seq, _ = generate_sample_data_for_viz(df, 0, max_pep_len, max_mhc_len, seq_map, embed_map)
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
          batch_size=256, source_col=None, allow_cache=False):
    global EMB_DB, MHC_CLASS, ESM_DIM
    os.makedirs(out_dir, exist_ok=True)
    with open(config_path, 'r') as f:
        config = json.load(f)
    MHC_CLASS, max_pep_len, max_mhc_len, embed_dim, heads = config["MHC_CLASS"], config["MAX_PEP_LEN"], config[
        "MAX_MHC_LEN"], config["EMBED_DIM"], config["HEADS"]
    EMB_DB = load_embedding_db(embedding_npz_path)
    ESM_DIM = int(next(iter(EMB_DB.values())).shape[1])
    seq_map = {clean_key(k): v for k, v in
               pd.read_csv(allele_seq_path, index_col="allele")["mhc_sequence"].to_dict().items()}
    embed_map = pd.read_csv(embedding_key_path, index_col="key")["mhc_sequence"].to_dict()
    df = pq.ParquetFile(df_path).read().to_pandas()
    print("Preprocessing dataset...")
    df_infer = preprocess_df(df, seq_map, embed_map)
    model = pmbind(max_pep_len=max_pep_len, max_mhc_len=max_mhc_len, emb_dim=embed_dim, heads=heads, noise_std=0,
                   drop_out_rate=0.0, latent_dim=embed_dim * 2, ESM_dim=ESM_DIM)
    dummy_gen = OptimizedDataGenerator(df_infer.head(1), seq_map, embed_map, max_pep_len, max_mhc_len, 1)
    model(dummy_gen[0], training=False)
    model.load_weights(model_weights_path)
    print("Model loaded successfully.")

    latents_seq_path, latents_pooled_path = os.path.join(out_dir, f"latents_seq_{name}.h5"), os.path.join(out_dir,
                                                                                                          f"latents_pooled_{name}.h5")
    if not (os.path.exists(latents_pooled_path) and allow_cache):
        print("Generating predictions and latents...")
        infer_gen = OptimizedDataGenerator(df_infer, seq_map, embed_map, max_pep_len, max_mhc_len, batch_size)
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

    with h5py.File(latents_seq_path, 'r') as f:
        latents_seq = f['latents'][:]
    with h5py.File(latents_pooled_path, 'r') as f:
        latents_pooled = f['latents'][:]

    if "assigned_label" in df_infer.columns:
        visualize_inference_results(df_infer, all_labels, all_predictions, out_dir, name)
        auc = roc_auc_score(all_labels, all_predictions) if len(np.unique(all_labels)) > 1 else -1
        ap = average_precision_score(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, df_infer["prediction_label"])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        summary_path = os.path.join(os.path.dirname(out_dir), "inference_summary.csv")
        file_exists = os.path.isfile(summary_path)
        with open(summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(['dataset', 'num_samples', 'AUC', 'AP', 'TP', 'TN', 'FP', 'FN'])
            writer.writerow([name, len(df_infer), f"{auc:.4f}", f"{ap:.4f}", tp, tn, fp, fn])
        print(f"✓ Summary updated.")

    run_visualizations(df_infer, latents_pooled, latents_seq, os.path.join(out_dir, "visualizations"), name,
                       max_pep_len, max_mhc_len, seq_map, embed_map, source_col)

    # Generate attention weight visualizations
    generate_attention_visualizations(model, df_infer, seq_map, embed_map, max_pep_len, max_mhc_len,
                                    os.path.join(out_dir, "visualizations"), name)

    # --- save input and predictions of the first 10 samples in a csv file ---
    pred_samples_df = df_infer.head(10)
    pred_data_gen = OptimizedDataGenerator(pred_samples_df, seq_map, embed_map, max_pep_len, max_mhc_len, 10)
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
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--source_col")
    args = parser.parse_args()
    if gpus := tf.config.list_physical_devices('GPU'):
        try:
            [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
            print(
                f"✓ Memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU error: {e}")
    infer(**vars(args))


if __name__ == "__main__":
    main()