#!/usr/bin/env python
"""
Minimal PMBind training script for classification and reconstruction tasks.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pyarrow.parquet as pq
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import json

# Import your utility functions and model
from utils import (seq_to_onehot, get_embed_key, NORM_TOKEN, MASK_TOKEN, PAD_TOKEN,
                   PAD_VALUE, MASK_VALUE, masked_categorical_crossentropy, clean_key)
from models import pmbind_subtract_moe_auto

from focal_loss import binary_focal_loss

# Global embedding database
EMB_DB: np.lib.npyio.NpzFile | None = None


def load_embedding_db(npz_path: str):
    """Load embedding database from NPZ file."""
    return np.load(npz_path, mmap_mode="r")


def rows_to_tensors(rows: pd.DataFrame, max_pep_len: int, max_mhc_len: int, seq_map: dict[str, str],
                    embed_map: dict[str, str]) -> dict[str, tf.Tensor]:
    n = len(rows)
    # This dictionary contains ALL data needed for a batch, including targets.
    batch_data = {
        "pep_onehot": np.zeros((n, max_pep_len, 21), np.float32),
        "pep_mask": np.full((n, max_pep_len), PAD_TOKEN, dtype=np.float32),  # Initialize with PAD_TOKEN
        "mhc_emb": np.zeros((n, max_mhc_len, 1152), np.float32),
        "mhc_mask": np.full((n, max_mhc_len), PAD_TOKEN, dtype=np.float32),  # Initialize with PAD_TOKEN
        "mhc_onehot": np.zeros((n, max_mhc_len, 21), np.float32),  # reconstruction target
        "labels": rows["assigned_label"].values.astype(np.float32),  # classification target
    }

    for i, (_, r) in enumerate(rows.iterrows()):
        ### PEP
        # Process peptide sequence
        pep_seq = r["long_mer"].upper()
        pep_OHE = seq_to_onehot(pep_seq, max_seq_len=max_pep_len)
        batch_data["pep_onehot"][i] = pep_OHE

        # Create peptide mask: 1.0 for valid positions, PAD_TOKEN for padding
        pep_len = len(pep_seq)
        batch_data["pep_mask"][i, :pep_len] = NORM_TOKEN  # Valid positions get NORM_TOKEN (1.0)
        # Positions beyond sequence length remain PAD_TOKEN (-2.0)

        # Randomly mask 30% of valid peptide positions with MASK_TOKEN
        valid_positions = np.where(batch_data["pep_mask"][i] == NORM_TOKEN)[0]
        if len(valid_positions) > 0:
            mask_fraction = 0.30
            n_mask = max(1, int(mask_fraction * len(valid_positions)))  # At least 1 position
            mask_indices = np.random.choice(valid_positions, size=n_mask, replace=False)
            batch_data["pep_mask"][i, mask_indices] = MASK_TOKEN  # Masked positions get MASK_TOKEN (-1.0)
            # Zero out the corresponding one-hot encoding for masked positions
            batch_data["pep_onehot"][i, mask_indices, :] = MASK_VALUE

        ### MHC
        # print(f"Peptide mask for sample {i}: {batch_data['pep_mask'][i]}")  # Debugging line to check peptide mask
        # Process MHC embeddings and sequence
        if MHC_CLASS == 2:
            key_parts = r["mhc_embedding_key"].split("_")
            embd_key1 = get_embed_key(clean_key(key_parts[0]), embed_map)
            embd_key2 = get_embed_key(clean_key(key_parts[1]), embed_map)
            emb1 = EMB_DB[embd_key1]
            emb2 = EMB_DB[embd_key2]
            emb = np.concatenate([emb1, emb2], axis=0)
        else:
            embd_key = get_embed_key(clean_key(r["mhc_embedding_key"]), embed_map)
            emb = EMB_DB[embd_key]
        L = emb.shape[0]
        batch_data["mhc_emb"][i, :L] = emb
        # Set padding positions in embeddings to PAD_VALUE
        batch_data["mhc_emb"][i, L:, :] = PAD_VALUE
        # print(batch_data["mhc_emb"][i, L:, :])  # Debugging line to check padding values

        # Create MHC mask based on the embedding values.
        # A position is considered padding if its embedding vector is all PAD_VALUE.
        # This handles both padding within the sequence and padding at the end.
        is_padding = np.all(batch_data["mhc_emb"][i] == PAD_VALUE, axis=-1)
        batch_data["mhc_mask"][i, ~is_padding] = NORM_TOKEN
        # Positions where is_padding is True will retain their initial PAD_TOKEN value.

        # Randomly mask 20% of valid MHC positions with MASK_TOKEN
        valid_mhc_positions = np.where(batch_data["mhc_mask"][i] == NORM_TOKEN)[0]
        if len(valid_mhc_positions) > 0:
            mask_fraction = 0.20
            n_mask = max(1, int(mask_fraction * len(valid_mhc_positions)))  # At least 1 position
            mask_indices = np.random.choice(valid_mhc_positions, size=n_mask, replace=False)
            batch_data["mhc_mask"][i, mask_indices] = MASK_TOKEN  # Masked positions get MASK_TOKEN (-1.0)
            # Zero out the corresponding embeddings for masked positions
            batch_data["mhc_emb"][i, mask_indices, :] = MASK_VALUE

        # print(f"MHC mask for sample {i}: {batch_data['mhc_mask'][i]}")  # Debugging line to check MHC mask

        # Get MHC sequence and convert to one-hot
        if MHC_CLASS == 2:
            key_parts = r["mhc_embedding_key"].split("_")
            key_norm1 = get_embed_key(clean_key(key_parts[0]), seq_map)
            key_norm2 = get_embed_key(clean_key(key_parts[1]), seq_map)
            mhc_seq = seq_map[key_norm1] + seq_map[key_norm2]
            batch_data["mhc_onehot"][i] = seq_to_onehot(mhc_seq, max_seq_len=max_mhc_len)
        else:
            key_norm = get_embed_key(clean_key(r["mhc_embedding_key"]), seq_map)
            mhc_seq = seq_map[key_norm]
            batch_data["mhc_onehot"][i] = seq_to_onehot(mhc_seq, max_seq_len=max_mhc_len)

        # Classification labels (assuming binary classification)
        # batch_data["labels"][i] = r.get("assigned_label", -1.0)  # -1.0 for unknown labels
        batch_data['labels'][i] = r['assigned_label']


    return {k: tf.convert_to_tensor(v) for k, v in batch_data.items()}


def combined_loss_fn(true_preds, batch_data, reconstruction_weight=0.2, classification_weight=2.0):
    """Combined loss function for reconstruction and classification."""
    # Reconstruction losses
    pep_recon_loss = tf.reduce_mean(
        masked_categorical_crossentropy(true_preds["pep_ytrue_ypred"], batch_data["pep_mask"])
    )
    mhc_recon_loss = tf.reduce_mean(
        masked_categorical_crossentropy(true_preds["mhc_ytrue_ypred"], batch_data["mhc_mask"])
    )

    # Classification loss
    if "cls_ypred" in true_preds:
        classification_loss = tf.reduce_mean(
            binary_focal_loss(
                y_true=batch_data["labels"],
                y_pred=true_preds["cls_ypred"],
                from_logits=True,
                gamma=2.0 # focusing parameter
            )
        )
    else:
        print("No classification predictions found; skipping classification loss.")
        classification_loss = 0.0

    total_loss = (reconstruction_weight * (pep_recon_loss + mhc_recon_loss) +
                  classification_weight * classification_loss)

    return total_loss, pep_recon_loss, mhc_recon_loss, classification_loss


def plot_training_history(history, out_dir):
    """Plot comprehensive training history."""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[0, 1].plot(epochs, history['train_recon'], 'g-', label='Reconstruction Loss', linewidth=2)
    axes[0, 1].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Classification loss
    axes[1, 0].plot(epochs, history['train_class'], 'purple', label='Classification Loss', linewidth=2)
    axes[1, 0].set_title('Classification Loss', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate (if available)
    if 'learning_rate' in history:
        axes[1, 1].plot(epochs, history['learning_rate'], 'orange', linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
    else:
        axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Tracked', ha='center', va='center',
                       fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(model1, val_data, seq_map, embed_map, mhc_class, max_pep_len, max_mhc_len, out_dir):
    """Comprehensive model evaluation with visualizations."""
    print("\n=== Model Evaluation ===")

    # Get predictions on validation set
    predictions_list = []
    labels_list = []

    batch_size = 64
    for i in range(0, len(val_data), batch_size):
        batch_df = val_data.iloc[i:i+batch_size]
        batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)

        pred = model1(batch_data, training=False)

        if "cls_ypred" in pred:
            predictions_list.append(tf.nn.sigmoid(pred["cls_ypred"]).numpy().flatten())
            labels_list.append(batch_data["labels"].numpy())

    if predictions_list:
        y_pred_proba = np.concatenate(predictions_list)
        y_true = np.concatenate(labels_list)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Create evaluation plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        axes[0, 1].plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
        axes[0, 2].set_title('Confusion Matrix', fontweight='bold')
        axes[0, 2].set_ylabel('True Label')
        axes[0, 2].set_xlabel('Predicted Label')
        # Prediction Distribution
        axes[1, 0].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='Negative', color='red')
        axes[1, 0].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Positive', color='blue')
        axes[1, 0].set_xlabel('Prediction Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Prediction Distribution', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        # Calibration Plot
        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_proba, n_bins=10)
        axes[1, 1].plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        axes[1, 1].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        axes[1, 1].set_xlabel('Mean Predicted Probability')
        axes[1, 1].set_ylabel('Fraction of Positives')
        axes[1, 1].set_title('Calibration Plot', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        # Performance Metrics Summary
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred),
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc
        }
        # Display metrics
        metrics_text = '\n'.join([f'{k}: {v:.3f}' for k, v in metrics.items()])
        axes[1, 2].text(0.5, 0.5, metrics_text, ha='center', va='center',
                          fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Performance Metrics', fontweight='bold')
        axes[1, 2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'model_evaluation.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # Save metrics to JSON
        with open(os.path.join(out_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        return metrics


def train_minimal(train_path: str, validation_path: str, embed_npz: str, seq_csv: str,
                  embd_key_path: str, out_dir: str, pretrained_pmclus_weights: str | None = None,
                  mhc_class: int = 1,
                  epochs: int = 3, batch_size: int = 32, lr: float = 1e-4,
                  embed_dim: int = 32, heads: int = 8, noise_std: float = 0.1,
                  reconstruction_weight: float = 1.0, classification_weight: float = 1.0):
    """
    Minimal training function for combined reconstruction and classification.
    """
    global EMB_DB
    EMB_DB = load_embedding_db(embed_npz)

    # Load data and mappings
    seq_map = pd.read_csv(seq_csv, index_col="allele")["mhc_sequence"].to_dict()
    embed_map = pd.read_csv(embd_key_path, index_col="key")["mhc_sequence"].to_dict()
    seq_map = {clean_key(k): v for k, v in seq_map.items()}

    df_train = pq.ParquetFile(train_path).read().to_pandas()
    df_val = pq.ParquetFile(validation_path).read().to_pandas()
    print(f"Loaded {len(df_train)} training, {len(df_val)} validation samples.")

    # Calculate max lengths
    max_pep_len = int(pd.concat([df_train["long_mer"], df_val["long_mer"]]).str.len().max())
    max_mhc_len = 500 if mhc_class == 2 else int(next(iter(EMB_DB.values())).shape[0])
    print(f"Max peptide length: {max_pep_len}, Max MHC length: {max_mhc_len}")

    # Initialize model
    model = pmbind_subtract_moe_auto(
        max_pep_len, max_mhc_len,
        emb_dim=embed_dim, heads=heads,
        mask_token=MASK_TOKEN, pad_token=PAD_TOKEN,
        noise_std=noise_std
    )

    # Optimizer with learning rate scheduling
    num_train_steps = (len(df_train) // batch_size) * epochs
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr, decay_steps=num_train_steps, alpha=0.0
    )
    optimizer = keras.optimizers.Lion(lr_schedule)

    os.makedirs(out_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_weights_path = os.path.join(out_dir, "best_model.weights.h5")

    # Training loop
    train_indices = np.arange(len(df_train))
    val_indices = np.arange(len(df_val))

    history = {
        'train_loss': [], 'val_loss': [], 'train_recon': [], 'train_class': [],
        'train_pep_recon': [], 'train_mhc_recon': [], 'learning_rate': [],
        'train_auc': [], 'val_auc': []
    }

    train_auc_metric = tf.keras.metrics.AUC()
    val_auc_metric = tf.keras.metrics.AUC()

    for epoch in range(1, epochs + 1):
        np.random.shuffle(train_indices)
        print(f"\nEpoch {epoch}/{epochs}")

        # Training
        epoch_loss, epoch_recon, epoch_class, epoch_pep_recon, epoch_mhc_recon = 0, 0, 0, 0, 0
        num_steps = 0
        pbar = tqdm(range(0, len(train_indices), batch_size), desc=f"Training")

        batch_auc = None
        for step in pbar:
            batch_idx = train_indices[step:step + batch_size]
            batch_df = df_train.iloc[batch_idx]
            batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len,
                                         seq_map, embed_map)

            with tf.GradientTape() as tape:
                predictions = model(batch_data, training=True)
                # threshold = tf.reduce_mean(batch_data["labels"])
                # predictions["cls_ypred"] = tf.where(predictions["cls_ypred"] > threshold,
                #                                    1.,
                #                                    0.)
                total_loss, pep_loss, mhc_loss, class_loss = combined_loss_fn(predictions, batch_data,
                                                                              reconstruction_weight,
                                                                              classification_weight)
                if step == 0:
                    p = predictions['cls_ypred'].numpy().ravel()
                    print(f"[Debug] cls_ypred min={p.min():.4f} max={p.max():.4f} std={p.std():.4f}")
                # Update AUC metric (only if classification present)
                if "cls_ypred" in predictions and "labels" in batch_data:
                    train_auc_metric.update_state(batch_data["labels"], predictions["cls_ypred"])

                gradients = tape.gradient(total_loss, model.trainable_variables)
                gradients = [g if g is not None else tf.zeros_like(v)
                             for g, v in zip(gradients, model.trainable_variables)]
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        epoch_loss += total_loss.numpy()
        epoch_pep_recon += pep_loss.numpy()
        epoch_mhc_recon += mhc_loss.numpy()
        epoch_recon += (pep_loss.numpy() + mhc_loss.numpy())
        epoch_class += class_loss.numpy() if isinstance(class_loss, tf.Tensor) else class_loss
        num_steps += 1

        pbar.set_postfix(loss=f"{total_loss.numpy():.4f}")

        # Record training metrics
        current_lr = optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, 'numpy') else lr
        history['train_loss'].append(epoch_loss / num_steps)
        history['train_recon'].append(epoch_recon / num_steps)
        history['train_pep_recon'].append(epoch_pep_recon / num_steps)
        history['train_mhc_recon'].append(epoch_mhc_recon / num_steps)
        history['train_class'].append(epoch_class / num_steps)
        history['learning_rate'].append(current_lr)
        train_auc = float(train_auc_metric.result().numpy()) if train_auc_metric.result() is not None else 0.0
        history['train_auc'].append(train_auc)

        # Validation
        val_loss, val_steps = 0, 0
        val_auc_metric.reset_state()
        for val_step in range(0, len(val_indices), batch_size):
            batch_idx = val_indices[val_step:val_step + batch_size]
            batch_df = df_val.iloc[batch_idx]
            batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len,
                                         seq_map, embed_map)

            predictions = model(batch_data, training=False)
            total_loss, _, _, _ = combined_loss_fn(predictions, batch_data,
                                                   reconstruction_weight,
                                                   classification_weight)
            if "cls_ypred" in predictions and "labels" in batch_data:
                val_auc_metric.update_state(batch_data["labels"], predictions["cls_ypred"])

            val_loss += total_loss.numpy()
            val_steps += 1

        avg_val_loss = val_loss / val_steps
        history['val_loss'].append(avg_val_loss)


        val_auc = float(val_auc_metric.result().numpy()) if val_auc_metric.result() is not None else 0.0
        history['val_auc'].append(val_auc)

        print(f"Train Loss: {history['train_loss'][-1]:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Recon: {history['train_recon'][-1]:.4f}, "
              f"Class: {history['train_class'][-1]:.4f}, LR: {current_lr:.2e}, "
              f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_weights(best_weights_path)
            print(f"✓ Best model saved with val loss: {best_val_loss:.4f}")

    # Save final model
    final_weights_path = os.path.join(out_dir, "final_model.weights.h5")
    model.save_weights(final_weights_path)

    # Plot training history
    plot_training_history(history, out_dir)

    # Load best model for evaluation
    # Re-initialize model and load best weights for evaluation
    print("Loading best model for evaluation...")
    model = pmbind_subtract_moe_auto(
        max_pep_len, max_mhc_len,
        emb_dim=embed_dim, heads=heads,
        mask_token=MASK_TOKEN, pad_token=PAD_TOKEN,
        noise_std=noise_std
    )
    # Build the model by calling it on a sample batch
    sample_batch_df = df_val.iloc[:batch_size]
    sample_batch_data = rows_to_tensors(sample_batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)
    _ = model(sample_batch_data, training=False)
    model.load_weights(best_weights_path)

    # Comprehensive evaluation
    metrics = evaluate_model(model, df_val, seq_map, embed_map, mhc_class, max_pep_len, max_mhc_len, out_dir)

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(out_dir, 'training_history.csv'), index=False)

    print(f"✓ Training complete. Results saved to {out_dir}")
    print(f"✓ Best validation loss: {best_val_loss:.4f}")
    if metrics:
        print(f"✓ Final ROC-AUC: {metrics.get('ROC-AUC', 'N/A'):.3f}")
        print(f"✓ Final Accuracy: {metrics.get('Accuracy', 'N/A'):.3f}")

    return model, history


# def evaluate_model(model, val_data, seq_map, embed_map, mhc_class, max_pep_len, max_mhc_len, out_dir):
#     """Comprehensive model evaluation with visualizations."""
#     print("\n=== Model Evaluation ===")
#
#     # Get predictions on validation set
#     predictions_list = []
#     labels_list = []
#
#     batch_size = 64
#     for i in range(0, len(val_data), batch_size):
#         batch_df = val_data.iloc[i:i+batch_size]
#         batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len, seq_map, embed_map, mhc_class)
#
#         pred = model(batch_data, training=False)
#
#         if "cls_ypred" in pred:
#             predictions_list.append(tf.nn.sigmoid(pred["cls_ypred"]).numpy().flatten())
#             labels_list.append(batch_data["labels"].numpy())
#
#     if predictions_list:
#         y_pred_proba = np.concatenate(predictions_list)
#         y_true = np.concatenate(labels_list)
#         y_pred = (y_pred_proba > 0.5).astype(int)
#
#         # Create evaluation plots
#         fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#
#         # ROC Curve
#         fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
#         roc_auc = auc(fpr, tpr)
#         axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
#         axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
#         axes[0, 0].set_xlabel('False Positive Rate')
#         axes[0, 0].set_ylabel('True Positive Rate')
#         axes[0, 0].set_title('ROC Curve', fontweight='bold')
#         axes[0, 0].legend()
#         axes[0, 0].grid(True, alpha=0.3)
#
#         # Precision-Recall Curve
#         precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
#         pr_auc = auc(recall, precision)
#         axes[0, 1].plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
#         axes[0, 1].set_xlabel('Recall')
#         axes[0, 1].set_ylabel('Precision')
#         axes[0, 1].set_title('Precision-Recall Curve', fontweight='bold')
#         axes[0, 1].legend()
#         axes[0, 1].grid(True, alpha=0.3)
#
#         # Confusion Matrix
#         cm = confusion_matrix(y_true, y_pred)
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
#         axes[0, 2].set_title('Confusion Matrix', fontweight='bold')
#         axes[0, 2].set_ylabel('True Label')
#         axes[0, 2].set_xlabel('Predicted Label')
#
#         # Prediction Distribution
#         axes[1, 0].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='Negative', color='red')
#         axes[1, 0].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Positive', color='blue')
#         axes[1, 0].set_xlabel('Prediction Probability')
#         axes[1, 0].set_ylabel('Frequency')
#         axes[1, 0].set_title('Prediction Distribution', fontweight='bold')
#         axes[1, 0].legend()
#         axes[1, 0].grid(True, alpha=0.3)
#
#         # Calibration Plot
#         from sklearn.calibration import calibration_curve
#         fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_proba, n_bins=10)
#         axes[1, 1].plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
#         axes[1, 1].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
#         axes[1, 1].set_xlabel('Mean Predicted Probability')
#         axes[1, 1].set_ylabel('Fraction of Positives')
#         axes[1, 1].set_title('Calibration Plot', fontweight='bold')
#         axes[1, 1].legend()
#         axes[1, 1].grid(True, alpha=0.3)
#
#         # Performance Metrics Summary
#         from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#
#         metrics = {
#             'Accuracy': accuracy_score(y_true, y_pred),
#             'Precision': precision_score(y_true, y_pred),
#             'Recall': recall_score(y_true, y_pred),
#             'F1-Score': f1_score(y_true, y_pred),
#             'ROC-AUC': roc_auc,
#             'PR-AUC': pr_auc
#         }
#
#         # Display metrics
#         metrics_text = '\n'.join([f'{k}: {v:.3f}' for k, v in metrics.items()])
#         axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
#                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
#         axes[1, 2].set_title('Performance Metrics', fontweight='bold')
#         axes[1, 2].set_xlim(0, 1)
#         axes[1, 2].set_ylim(0, 1)
#         axes[1, 2].axis('off')
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(out_dir, 'model_evaluation.png'), dpi=300, bbox_inches='tight')
#         plt.show()
#
#         # Save metrics to file
#         with open(os.path.join(out_dir, 'evaluation_metrics.json'), 'w') as f:
#             json.dump(metrics, f, indent=2)
#
#         return metrics
#     else:
#         print("No classification predictions available for evaluation")
#         return {}
#
#
# def visualize_attention_weights(model, sample_data, out_dir):
#     """Visualize attention patterns if available."""
#     try:
#         # Get model predictions with attention weights
#         predictions = model(sample_data, training=False)
#
#         if hasattr(model, 'get_attention_weights'):
#             attention_weights = model.get_attention_weights(sample_data)
#
#             fig, axes = plt.subplots(2, 2, figsize=(15, 12))
#
#             # Plot attention heatmaps
#             for i, (name, weights) in enumerate(attention_weights.items()):
#                 if i >= 4:
#                     break
#                 ax = axes[i // 2, i % 2]
#
#                 # Average over heads and batch
#                 attn_avg = np.mean(weights, axis=(0, 1))
#
#                 sns.heatmap(attn_avg, ax=ax, cmap='viridis')
#                 ax.set_title(f'{name} Attention', fontweight='bold')
#                 ax.set_xlabel('Key Position')
#                 ax.set_ylabel('Query Position')
#
#             plt.tight_layout()
#             plt.savefig(os.path.join(out_dir, 'attention_weights.png'), dpi=300, bbox_inches='tight')
#             plt.show()
#
#     except Exception as e:
#         print(f"Could not visualize attention weights: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    MHC_CLASS = 1  # Set MHC class (1 or 2)
    config = {
        "MHC_CLASS": 1,
        "EPOCHS": 3,
        "BATCH_SIZE": 128,
        "LEARNING_RATE": 1e-3,
        "EMBED_DIM": 16,
        "HEADS": 2,
        "NOISE_STD": 0.1,
    }

    # --- Paths ---
    paths = {
        "train": f"../tests/binding_affinity_dataset_with_swapped_negatives{config['MHC_CLASS']}_train.parquet",
        "val": f"../tests/binding_affinity_dataset_with_swapped_negatives{config['MHC_CLASS']}_val.parquet",
        "embed_npz": f"/media/amirreza/lasse/ESM/esmc_600m/PMGen_whole_seq/mhc{config['MHC_CLASS']}_encodings.npz",
        "embed_key": f"/media/amirreza/lasse/ESM/esmc_600m/PMGen_whole_seq/mhc{config['MHC_CLASS']}_encodings.csv",
        "seq_csv": f"../data/alleles/aligned_PMGen_class_{config['MHC_CLASS']}.csv",
        "out_dir": f"../outputs/minimal_run_mhc{config['MHC_CLASS']}",
        "pretrained_pmclust": f"/media/amirreza/lasse/PMClust_runs/run_PMClust_ns_0.1_hds_4_zdim_21_L1_all/1/best_model.weights.h5"
    }

    # --- Train model ---
    model, history = train_minimal(
        train_path=paths["train"],
        validation_path=paths["val"],
        embed_npz=paths["embed_npz"],
        seq_csv=paths["seq_csv"],
        embd_key_path=paths["embed_key"],
        out_dir=paths["out_dir"],
        mhc_class=config["MHC_CLASS"],
        epochs=config["EPOCHS"],
        batch_size=config["BATCH_SIZE"],
        lr=config["LEARNING_RATE"],
        embed_dim=config["EMBED_DIM"],
        heads=config["HEADS"],
        noise_std=config["NOISE_STD"],
        pretrained_pmclus_weights=paths["pretrained_pmclust"],
        reconstruction_weight=0.1,
        classification_weight=4.0
    )


    print("Training completed successfully!")