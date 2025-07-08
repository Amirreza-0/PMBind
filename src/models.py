#!/usr/bin/env python
"""
pepmhc_cross_attention.py
-------------------------

Minimal end-to-end peptide × MHC cross-attention reconstruction model
with explainable attention visualisation.

Author: Amirreza 2025-07-08
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import seq_to_onehot, AttentionLayer, PositionalEncoding, MaskedEmbedding, ConcatMask, ConcatBarcode, SplitLayer



MASK_TOKEN = -1.0
PAD_TOKEN = -2.0


# Model builder function
def bicross_recon(max_pep_len: int,
                  max_mhc_len: int,
                  mask_token: float = MASK_TOKEN,
                  pad_token: float = PAD_TOKEN,
                  pep_emb_dim: int = 128,
                  mhc_emb_dim: int = 128,
                  heads: int = 4):
    """
    Builds peptide-MHC cross-attention models with reconstruction heads.

    Returns:
        recon_model: Model for reconstructing inputs.
        att_model: Model for attention weights.
        cross_latent_model: Model for latent representation.
    """
    # Inputs
    pep_OHE = keras.Input(shape=(max_pep_len, 21), name="pep_onehot")
    mask_pep = keras.Input(shape=(max_pep_len,), name="pep_mask")
    mhc_EMB = keras.Input(shape=(max_mhc_len, 1152), name="mhc_latent")
    mask_mhc = keras.Input(shape=(max_mhc_len,), name="mhc_mask")
    mhc_ids = keras.Input(shape=(max_mhc_len,), name="mhc_ids", dtype=tf.int32)
    mhc_OHE = keras.Input(shape=(max_mhc_len,), name="mhc_seq")

    # Embedding layer ------------------------------------------------
    # zero out the mask positions in the embeddings
    pep_OHE = MaskedEmbedding(mask_token=mask_token, pad_token=pad_token, name="pep_masked_OHE")(pep_OHE, mask_pep)
    mhc_EMB = MaskedEmbedding(mask_token=mask_token, pad_token=pad_token, name="mhc_masked_embedding")(mhc_EMB, mask_mhc)

    # add positional encoding and project pep_OHE to embedding dimension
    pep_OHE = PositionalEncoding(int(max_pep_len * 1.5) , 21, name="pep_pos_OHE")(pep_OHE)
    mhc_EMB = PositionalEncoding(int(max_mhc_len * 1.5), 1152, name="mhc_pos_encoding")(mhc_EMB)

    # Project to higher dimensions
    pep_OHE = layers.Dense(pep_emb_dim, activation=None, name="pep_embedding")(pep_OHE)
    mhc_EMB = layers.Dense(mhc_emb_dim, activation="relu",  name="mhc_embedding")(mhc_EMB)

    # Encoder ------------------------------------------------
    # Self-attention
    pep_self_attn, pep_self_attn_scores = AttentionLayer(  query_dim=pep_emb_dim, context_dim=pep_emb_dim, output_dim=pep_emb_dim,
                                                           type="self", heads=heads, resnet=True,
                                                           return_att_weights=True, name="pep_self_attn",
                                                           mask_token=mask_token,
                                                           pad_token=pad_token)(pep_OHE, mask_pep)
    mhc_self_attn, mhc_self_attn_scores = AttentionLayer(  query_dim=mhc_emb_dim, context_dim=mhc_emb_dim, output_dim=mhc_emb_dim,
                                                           type="self", heads=heads, resnet=True,
                                                           return_att_weights=True, name="mhc_self_attn",
                                                           mask_token=mask_token,
                                                           pad_token=pad_token)(mhc_EMB, mask_mhc)

    # Z Latent Representation ---------------------------------------------
    # Cross-attention
    # pep_cross_attn, pep_cross_attn_score  = AttentionLayer( query_dim=pep_emb_dim, context_dim=mhc_emb_dim, output_dim=pep_emb_dim,
    #                                                         type="cross", heads=heads, resnet=False,
    #                                                         return_att_weights=True, name="pep_cross_attn",
    #                                                         mask_token=mask_token,
    #                                                         pad_token=pad_token)(pep_self_attn, mask_pep,
    #                                                                              mhc_self_attn, mask_mhc)
    mhc_cross_attn, pep_cross_attn_scores = AttentionLayer( query_dim=mhc_emb_dim, context_dim=pep_emb_dim, output_dim=mhc_emb_dim,
                                                            type="cross", heads=heads, resnet=False,
                                                            return_att_weights=True, name="mhc_cross_attn",
                                                            mask_token=mask_token,
                                                            pad_token=pad_token)(mhc_self_attn, mask_mhc,
                                                                                 pep_self_attn, mask_pep)
    # create pmhc_mask
    pmhc_mask = ConcatMask(name="pmhc_mask")([mask_pep, mask_mhc])
    # create pmhc_barcode
    barcoded_mhc = ConcatBarcode(max_pep_len, mhc_emb_dim, name="pmhc_barcode")([mhc_cross_attn])

    # self-attention on the barcoded MHC
    barcoded_mhc_attn, barcoded_mhc_attn_score = AttentionLayer(  query_dim=mhc_emb_dim, context_dim=mhc_emb_dim, output_dim=mhc_emb_dim,
                                                                    type="self", heads=heads, resnet=False,
                                                                    return_att_weights=True, name="barcoded_mhc_attn",
                                                                    mask_token=mask_token,
                                                                    pad_token=pad_token)(barcoded_mhc, pmhc_mask)
    # Split the barcoded MHC into two parts: barcode and MHC
    barcode, mhc_splitted = SplitLayer((max_pep_len, max_mhc_len),name="barcode_mhc_split")([barcoded_mhc_attn])

    # Decoder ------------------------------------------------
    # Reconstruction heads
    pep_recon_dense = layers.Dense(64, activation='relu', name='pep_recon_dense')(barcode)
    pep_recon_out = layers.Dense(21, activation='softmax', name='pep_reconstruction')(pep_recon_dense)
    mhc_recon_dense = layers.Dense(64, activation='relu', name='mhc_recon_dense')(mhc_splitted)
    mhc_recon_out = layers.Dense(21, activation='softmax', name='mhc_reconstruction')(mhc_recon_dense)


    # TODO: add losses and metrics


    encoder = keras.Model(inputs=[pep_OHE, mask_pep, mhc_EMB, mask_mhc],
                            outputs=[mhc_cross_attn, pep_cross_attn_scores],
                            name='encoder')
    decoder = keras.Model(inputs=[pep_OHE, mask_pep, mhc_EMB, mask_mhc],
                                     outputs=[pep_recon_out, mhc_recon_out],
                                     name='decoder')

    return recon_model, att_model, cross_latent_model

# Test run with synthetic data
if __name__ == "__main__":
    tf.random.set_seed(0)
    np.random.seed(0)

    # Parameters
    max_pep_len = 15
    max_mhc_len = 34
    batch_size = 32
    pep_emb_dim = 64
    mhc_emb_dim = 64
    heads = 8

    # Generate synthetic data
    pep_lengths = np.random.randint(8, 16, size=batch_size)
    pep_seqs = [''.join(np.random.choice(list(AA), size=length)) for length in pep_lengths]
    pep_OHE = np.array([onehot(seq, max_pep_len) for seq in pep_seqs], dtype=np.float32)
    mask_pep = np.full((batch_size, max_pep_len), PAD_TOKEN, dtype=np.float32)
    for i, length in enumerate(pep_lengths):
        mask_pep[i, :length] = 1.0

    mhc_lengths = np.random.randint(30, 35, size=batch_size)
    mhc_EMB = np.random.randn(batch_size, max_mhc_len, 1152).astype(np.float32)
    for i, length in enumerate(mhc_lengths):
        mhc_EMB[i, length:, :] = 0.0
    mask_mhc = np.full((batch_size, max_mhc_len), PAD_TOKEN, dtype=np.float32)
    for i, length in enumerate(mhc_lengths):
        mask_mhc[i, :length] = 1.0

    # Build models
    recon_model, att_model, cross_latent_model = bicross_recon(
        max_pep_len=max_pep_len,
        max_mhc_len=max_mhc_len,
        mask_token=MASK_TOKEN,
        pad_token=PAD_TOKEN,
        pep_emb_dim=pep_emb_dim,
        mhc_emb_dim=mhc_emb_dim,
        heads=heads
    )

    # Run predictions
    pep_recon, mhc_recon = recon_model.predict([pep_OHE, mask_pep, mhc_EMB, mask_mhc])
    pep_att_weights, mhc_att_weights = att_model.predict([pep_OHE, mask_pep, mhc_EMB, mask_mhc])

    # Verify shapes
    print(f"pep_recon shape: {pep_recon.shape}")  # Expected: (32, 15, 21)
    print(f"mhc_recon shape: {mhc_recon.shape}")  # Expected: (32, 34, 1152)
    print(f"pep_att_weights shape: {pep_att_weights.shape}")  # Expected: (32, 8, 15, 34)
    print(f"mhc_att_weights shape: {mhc_att_weights.shape}")  # Expected: (32, 8, 34, 15)

    # Visualize attention weights for the first sample
    pep_att_avg = pep_att_weights[0].mean(axis=0)  # Shape: (15, 34)
    mhc_att_avg = mhc_att_weights[0].mean(axis=0)  # Shape: (34, 15)

    plt.figure(figsize=(6, 4))
    sns.heatmap(pep_att_avg, cmap='viridis')
    plt.title('Peptide-MHC Cross-Attention')
    plt.xlabel('MHC Positions')
    plt.ylabel('Peptide Positions')
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.heatmap(mhc_att_avg, cmap='viridis')
    plt.title('MHC-Peptide Cross-Attention')
    plt.xlabel('Peptide Positions')
    plt.ylabel('MHC Positions')
    plt.show()