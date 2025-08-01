#!/usr/bin/env python
"""
Minimal GradientTape training loop for `pmclust_subtract` **without using `tf.data.Dataset`**.
------------------------------------------------------------------------------------------------
This script shows the bare-bones path through your model:

1.  Load all rows from a Parquet file into **pandas**.
2.  Shuffle & slice with NumPy to create mini-batches.
3.  Convert every mini-batch to tensors on-the-fly and feed it through a
    `tf.GradientTape` loop.

Everything runs in eager mode – no `Dataset.from_generator`, no
prefetching pipelines – so the control-flow is crystal-clear when you
step through with a debugger.

NOTE: This version has been made self-contained and runnable. It includes
the user's model structure with fixes and placeholder custom layers.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pyarrow.parquet as pq
import os

from models import pmclust_subtract

from utils import seq_to_onehot, get_embed_key, NORM_TOKEN, MASK_TOKEN, PAD_TOKEN, PAD_VALUE, \
    masked_categorical_crossentropy, OHE_to_seq_single, split_y_true_y_pred, OHE_to_seq, clean_key

# ──────────────────────────────────────────────────────────────────────
# 4. DATA PREPARATION & TRAINING LOOP (Corrected)
# ----------------------------------------------------------------------
EMB_DB: np.lib.npyio.NpzFile | None = None


def load_embedding_db(npz_path: str):
    return np.load(npz_path, mmap_mode="r")


def random_mask(length: int, mask_fraction: float = 0.15) -> np.ndarray:
    out = np.full((length,), NORM_TOKEN, dtype=np.float32)
    n_mask = int(mask_fraction * length)
    if n_mask > 0:
        idx = np.random.choice(length, n_mask, replace=False)
        out[idx] = MASK_TOKEN
    return out


def rows_to_tensors(rows: pd.DataFrame, max_pep_len: int, max_mhc_len: int, seq_map: dict[str, str], embed_map: dict[str, str]) -> dict[str, tf.Tensor]:
    n = len(rows)
    # This dictionary contains ALL data needed for a batch, including targets.
    batch_data = {
        "pep_onehot": np.zeros((n, max_pep_len, 21), np.float32),
        "pep_mask": np.zeros((n, max_pep_len), np.float32),
        "mhc_emb": np.zeros((n, max_mhc_len, 1152), np.float32),
        "mhc_mask": np.zeros((n, max_mhc_len), np.float32),
        "mhc_onehot": np.zeros((n, max_mhc_len, 21), np.float32),  # Target
    }

    for i, (_, r) in enumerate(rows.iterrows()):
        pep_seq = r["long_mer"].upper()
        pep_OHE = seq_to_onehot(pep_seq, max_seq_len=max_pep_len)
        batch_data["pep_onehot"][i] = pep_OHE
        pep_mask = np.ones((max_pep_len,), dtype=np.float32)
        pep_mask[np.all(pep_OHE == PAD_VALUE, axis=-1)] = PAD_TOKEN  # Set padding to PAD_TOKEN
        batch_data["pep_mask"][i] = pep_mask

        embd_key = get_embed_key(r["mhc_embedding_key"], embed_map)
        emb = EMB_DB[embd_key]
        L = emb.shape[0]
        batch_data["mhc_emb"][i, :L] = emb
        mhc_mask = np.ones((max_mhc_len,), dtype=np.float32)
        mhc_mask[np.all(emb == PAD_VALUE, axis=-1)] = PAD_TOKEN  # Set padding to PAD_TOKEN
        batch_data["mhc_mask"][i] = mhc_mask
        # locate the row in seq_map that has the same key_norm
        key_norm = get_embed_key(r["mhc_embedding_key"], seq_map)
        mhc_seq = seq_map[key_norm]
        batch_data["mhc_onehot"][i] = seq_to_onehot(mhc_seq, max_seq_len=max_mhc_len)

    return {k: tf.convert_to_tensor(v) for k, v in batch_data.items()}


def train(parquet_path: str, embed_npz: str, seq_csv: str, embd_key_path: str,
          out_dir: str, epochs: int = 3, batch_size: int = 32, lr: float = 1e-4, embed_dim: int = 32):
    global EMB_DB
    EMB_DB = load_embedding_db(embed_npz)

    seq_map = pd.read_csv(seq_csv, index_col="allele")["mhc_sequence"].to_dict()
    embed_map = pd.read_csv(embd_key_path, index_col="key")["mhc_sequence"].to_dict()

    # harmonize seq_map keys using clean_keys function, update the keys in seq_map
    seq_map = {clean_key(k): v for k, v in seq_map.items()}

    df_all = pq.ParquetFile(parquet_path).read().to_pandas()

    max_pep_len = df_all["long_mer"].str.len().max()
    max_mhc_len = next(iter(EMB_DB.values())).shape[0]

    enc_dec = pmclust_subtract(max_pep_len,
                       max_mhc_len,
                       emb_dim= embed_dim,
                       heads= 8,
                       mask_token= MASK_TOKEN,
                       pad_token= PAD_TOKEN)

    opt = keras.optimizers.Adam(lr)

    loss_fn = masked_categorical_crossentropy
    indices = np.arange(len(df_all))

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        print(f"\nEpoch {epoch}/{epochs}")

        for step in range(0, len(indices), batch_size):
            batch_idx = indices[step:step + batch_size]
            batch_df = df_all.iloc[batch_idx]
            batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)

            # Separate the model inputs from the target labels
            model_inputs = {
                "pep_onehot": batch_data["pep_onehot"],
                "pep_mask": batch_data["pep_mask"],
                "mhc_emb": batch_data["mhc_emb"],
                "mhc_mask": batch_data["mhc_mask"],
                "mhc_onehot": batch_data["mhc_onehot"],  # This is the target for reconstruction
            }

            # print shapes
            # print(f"Batch {step // batch_size + 1}:")
            # for k, v in model_inputs.items():
            #     print(f"  {k}: {v.shape}")

            with tf.GradientTape() as tape:
                true_and_preds = enc_dec(model_inputs, training=True)

                pep_loss_per_sample = loss_fn(
                    true_and_preds["pep_ytrue_ypred"],
                    batch_data["pep_mask"]
                )
                mhc_loss_per_sample = loss_fn(
                    true_and_preds["mhc_ytrue_ypred"],
                    batch_data["mhc_mask"]
                )

                pep_loss = tf.reduce_mean(pep_loss_per_sample)
                mhc_loss = tf.reduce_mean(mhc_loss_per_sample)

                loss = pep_loss + mhc_loss

            grads = tape.gradient(loss, enc_dec.trainable_variables)
            opt.apply_gradients(zip(grads, enc_dec.trainable_variables))

            if step % (batch_size * 2) == 0:
                print(
                    f"step {step // batch_size:4d}  loss={loss.numpy():.4f} (pep={pep_loss.numpy():.4f}, mhc={mhc_loss.numpy():.4f})")

    weights_path = os.path.join(out_dir, "enc_dec.weights.h5")
    enc_dec.save_weights(weights_path)
    print(f"✓ Training finished. Weights saved to {weights_path}")

    # run again and predict
    latents = np.zeros((len(df_all), max_mhc_len, embed_dim), np.float32)
    print("\nRunning inference on the training data...")
    for step in range(0, len(indices), batch_size):
        batch_idx = indices[step:step + batch_size]
        batch_df = df_all.iloc[batch_idx]
        batch_data = rows_to_tensors(batch_df, max_pep_len, max_mhc_len, seq_map, embed_map)

        model_inputs = {
            "pep_onehot": batch_data["pep_onehot"],
            "pep_mask": batch_data["pep_mask"],
            "mhc_emb": batch_data["mhc_emb"],
            "mhc_mask": batch_data["mhc_mask"],
            "mhc_onehot": batch_data["mhc_onehot"],  # This is the target for reconstruction
        }

        true_preds = enc_dec(model_inputs, training=False)
        # split the predictions
        preds = {
            "pep_ytrue_ypred": true_preds["pep_ytrue_ypred"],
            "mhc_ytrue_ypred": true_preds["mhc_ytrue_ypred"],
            "cross_latent": true_preds["cross_latent"],
        }
        # print shapes
        # print(f"Batch {step // batch_size + 1}:")
        # for k, v in preds.items():
        #     print(f"  {k}: {v.shape}")

        pep_true, pep_pred = split_y_true_y_pred(preds["pep_ytrue_ypred"].numpy())
        mhc_true, mhc_pred = split_y_true_y_pred(preds["mhc_ytrue_ypred"].numpy())

        mhc_seq_pred = OHE_to_seq(mhc_pred, max_mhc_len)
        mhc_seq_orig = OHE_to_seq(mhc_true, max_mhc_len)
        print(f"Batch {step // batch_size + 1}: Predicted MHC sequences: {mhc_seq_pred}, Original: {mhc_seq_orig}")

        pep_pred = OHE_to_seq(pep_pred, max_pep_len)
        pep_true = OHE_to_seq(pep_true, max_pep_len)
        print(f"Batch {step // batch_size + 1}: Predicted peptide sequences: {pep_pred}, Original: {pep_true}")

        latents[batch_idx] = preds["cross_latent"].numpy()

    latents_path = os.path.join(out_dir, "mhc_latents.npy")
    np.save(latents_path, latents)


if __name__ == "__main__":
    # Suppress verbose TensorFlow logging, but keep errors.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    parquet_path = "../data/binding_affinity_data/positives_class1_subset.parquet"
    embed_npz_path = "/media/amirreza/lasse/trash/mhc1_encodings.npz"
    embd_key_path = "/media/amirreza/lasse/trash/mhc1_encodings.csv"
    seq_csv_path = "../data/alleles/aligned_PMGen_class_1.csv"
    out_dir = "run_PMClust"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train(
        parquet_path=parquet_path,
        embed_npz=embed_npz_path,
        seq_csv=seq_csv_path,
        embd_key_path=embd_key_path,
        out_dir=out_dir,
        epochs=2,
        batch_size=128,
    )
