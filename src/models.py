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
from keras.layers import Conv1D
from src.utils import (seq_to_onehot, AttentionLayer, PositionalEncoding, MaskedEmbedding,
                       ConcatMask, ConcatBarcode, SplitLayer, OHE_to_seq, determine_ks_dict)



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
    pep_OHE_inp = keras.Input(shape=(max_pep_len, 21), name="pep_onehot", dtype=tf.float32)
    mask_pep_inp = keras.Input(shape=(max_pep_len,), name="pep_mask", dtype=tf.float32)
    mhc_EMB_inp = keras.Input(shape=(max_mhc_len, 1152), name="mhc_latent", dtype=tf.float32)
    mask_mhc_inp = keras.Input(shape=(max_mhc_len,), name="mhc_mask", dtype=tf.float32)
    # mhc_ids = keras.Input(shape=(max_mhc_len,), name="mhc_ids", dtype=tf.int32)
    # mhc_OHE_inp = keras.Input(shape=(max_mhc_len,), name="mhc_seq", dtype=tf.float32)

    # Embedding layer ------------------------------------------------
    # zero out the mask positions in the embeddings
    pep_OHE = MaskedEmbedding(mask_token=mask_token, pad_token=pad_token, name="pep_masked_OHE")(pep_OHE_inp, mask_pep_inp)
    mhc_EMB = MaskedEmbedding(mask_token=mask_token, pad_token=pad_token, name="mhc_masked_embedding")(mhc_EMB_inp, mask_mhc_inp)

    # add positional encoding and project pep_OHE to embedding dimension also zero out the mask positions in the embeddings
    pep_OHE_m = PositionalEncoding(embed_dim=21 ,pos_range=int(max_pep_len * 3) , name="pep_pos_OHE")(pep_OHE, mask_pep_inp)
    mhc_EMB_m = PositionalEncoding(embed_dim=1152, pos_range=int(max_mhc_len * 3), name="mhc_pos_encoding")(mhc_EMB, mask_mhc_inp)

    # Project to higher dimensions
    pep_OHE = layers.Dense(pep_emb_dim, activation=None, name="pep_embedding")(pep_OHE_m)
    mhc_EMB = layers.Dense(mhc_emb_dim, activation="relu",  name="mhc_embedding")(mhc_EMB_m)

    # Encoder ------------------------------------------------
    # Self-attention
    pep_self_attn, pep_self_attn_scores = AttentionLayer(  query_dim=pep_emb_dim, context_dim=pep_emb_dim, output_dim=pep_emb_dim,
                                                           type="self", heads=heads, resnet=True,
                                                           return_att_weights=True, name="pep_self_attn",
                                                           mask_token=mask_token,
                                                           pad_token=pad_token)(pep_OHE, mask_pep_inp)
    mhc_self_attn, mhc_self_attn_scores = AttentionLayer(  query_dim=mhc_emb_dim, context_dim=mhc_emb_dim, output_dim=mhc_emb_dim,
                                                           type="self", heads=heads, resnet=True,
                                                           return_att_weights=True, name="mhc_self_attn",
                                                           mask_token=mask_token,
                                                           pad_token=pad_token)(mhc_EMB, mask_mhc_inp)

    # Latent Representation ---------------------------------------------
    # Cross-attention
    mhc_cross_attn, mhc_cross_attn_scores = AttentionLayer( query_dim=mhc_emb_dim, context_dim=pep_emb_dim, output_dim=mhc_emb_dim,
                                                            type="cross", heads=heads, resnet=False,
                                                            return_att_weights=True, name="mhc_cross_attn",
                                                            mask_token=mask_token,
                                                            pad_token=pad_token)(mhc_self_attn, mask_mhc_inp,
                                                                                 pep_self_attn, mask_pep_inp)
    #ks_dict = {"k1": 3, "s1": 2, "k2": 3, "s2": 1, "k3": 3, "s3": 1, "k4": 2, "s4": 1}
    ks_dict = determine_ks_dict(initial_input_dim=max_mhc_len, output_dims=[max_mhc_len,max_mhc_len//2,max_mhc_len//2,max_pep_len], max_strides=20, max_kernel_size=60)
    # Convolutional layers
    pepconv = Conv1D(filters=64, activation='relu', kernel_size=ks_dict["k1"], strides=ks_dict["s1"], padding='valid', name="pepconv1")(mhc_cross_attn)
    pepconv = Conv1D(filters=64, activation='relu', kernel_size=ks_dict["k2"], strides=ks_dict["s2"], padding='valid', name="pepconv2")(pepconv)
    pepconv = Conv1D(filters=32, activation='relu', kernel_size=ks_dict["k3"], strides=ks_dict["s3"], padding='valid', name="pepconv3")(pepconv)
    pepconv = Conv1D(filters=32, activation='relu', kernel_size=ks_dict["k4"], strides=ks_dict["s4"], padding='valid', name="pepconv4")(pepconv)
    pepconv, mhc_to_pep_attn_score = AttentionLayer(query_dim=32, context_dim=32, output_dim=32,
                                                            type="self", heads=heads, resnet=True,
                                                            return_att_weights=True, name="mhc_to_pep_attn",
                                                            mask_token=mask_token,
                                                            pad_token=pad_token)(pepconv, mask_pep_inp)
    # pep_cross_attn, pep_cross_attn_score  = AttentionLayer( query_dim=pep_emb_dim, context_dim=mhc_emb_dim, output_dim=pep_emb_dim,
    #                                                         type="cross", heads=heads, resnet=False,
    #                                                         return_att_weights=True, name="pep_cross_attn",
    #                                                         mask_token=mask_token,
    #                                                         pad_token=pad_token)(pep_self_attn, mask_pep,
    #                                                                              mhc_cross_attn, mask_mhc_inp)

    # the mhc_cross_attn shape is (batch_size, max_mhc_len, mhc_emb_dim)
    # verify the shape
    # assert mhc_cross_attn.shape == (None, max_mhc_len, mhc_emb_dim), f"Expected shape {(None, max_mhc_len, mhc_emb_dim)}, got {mhc_cross_attn.shape}"
    tf.print("mhc_cross_attn shape:", mhc_to_pep_attn_score.shape)

    # Decoder ------------------------------------------------
    # Reconstruction heads
    pep_recon_dense = layers.Dense(64, activation='relu', name='pep_recon_dense')(pepconv)
    pep_recon_out = layers.Dense(21, activation='softmax', name='pep_reconstruction')(pep_recon_dense)
    mhc_recon_dense = layers.Dense(64, activation='relu', name='mhc_recon_dense')(mhc_cross_attn)
    mhc_recon_out = layers.Dense(21, activation='softmax', name='mhc_reconstruction')(mhc_recon_dense)

    encoder = keras.Model(inputs=[pep_OHE_inp, mask_pep_inp, mhc_EMB_inp, mask_mhc_inp],
                            outputs={'cross_latent': mhc_cross_attn, 'attention_scores_CA': mhc_cross_attn_scores},
                            name='encoder')
    encoder_decoder = keras.Model(inputs=[pep_OHE_inp, mask_pep_inp, mhc_EMB_inp, mask_mhc_inp],
                                  outputs={'pep_reconstruction': pep_recon_out, 'mhc_reconstruction': mhc_recon_out,
                                           "mhc_to_pep_attn_score": mhc_to_pep_attn_score,
                                           "attention_scores_CA":mhc_cross_attn_scores,
                                           "pep_OHE_m":pep_OHE_m,
                                           "mhc_EMB_m":mhc_EMB_m},
                                  name='encoder_decoder')

    # Define losses
    # 1. reconstruction loss for barcode and MHC separately normalized by sequence length
    # 2. reconstruction loss of masked peptide and MHC positions
    # 3. (optional) reward function for attention weights with respect to anchor rules (eg. attention hotspots must be at least 2 positions apart)


    # Compile the model
    encoder_decoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'pep_reconstruction': 'categorical_crossentropy',
            'mhc_reconstruction': 'categorical_crossentropy'
        },
        loss_weights={
            'pep_reconstruction': 1.0,
            'mhc_reconstruction': 1.0
        },
        metrics={'pep_reconstruction': 'accuracy', 'mhc_reconstruction': 'accuracy'}
    )

    return encoder, encoder_decoder


# Test run with synthetic data
if __name__ == "__main__":
    tf.random.set_seed(0)
    np.random.seed(0)

    # Parameters
    max_pep_len = 14
    max_mhc_len = 342
    batch_size = 32
    pep_emb_dim = 128
    mhc_emb_dim = 128
    heads = 4

    # Generate synthetic data
    # Define amino acids
    # Define amino acids
    AA = "ACDEFGHIKLMNPQRSTVWY"

    # Position-specific amino acid frequencies for peptides
    # Simplified frequencies where certain positions prefer specific amino acids
    pep_pos_freq = {
        0: {"A": 0.3, "G": 0.2, "M": 0.2},  # Position 1 prefers A, G, M
        1: {"L": 0.3, "V": 0.3, "I": 0.2},  # Position 2 prefers hydrophobic
        2: {"D": 0.2, "E": 0.2, "N": 0.2},  # Position 3 prefers charged/polar
    }
    # Default distribution for other positions
    default_aa_freq = {aa: 1/len(AA) for aa in AA}

    # Generate peptides with position-specific preferences
    pep_lengths = np.random.choice([8, 9, 10, 11], size=batch_size, p=[0.1, 0.6, 0.2, 0.1])  # More realistic length distribution
    pep_seqs = []
    for length in pep_lengths:
        seq = []
        for pos in range(length):
            # Use position-specific frequencies if available, otherwise default
            freq = pep_pos_freq.get(pos, default_aa_freq)
            # Convert frequencies to probability array
            aa_list = list(AA)
            probs = [freq.get(aa, 0.01) for aa in aa_list]
            probs = np.array(probs) / sum(probs)  # Normalize
            seq.append(np.random.choice(aa_list, p=probs))
        pep_seqs.append(''.join(seq))

    # Convert peptide sequences to one-hot encoding
    pep_OHE = np.array([seq_to_onehot(seq, max_pep_len) for seq in pep_seqs], dtype=np.float32)
    mask_pep = np.full((batch_size, max_pep_len), PAD_TOKEN, dtype=np.float32)
    for i, length in enumerate(pep_lengths):
        mask_pep[i, :length] = 1.0

    # MHC alleles typically have conserved regions
    mhc_pos_freq = {
        0: {"G": 0.5, "D": 0.3},  # First position often G or D
        1: {"S": 0.4, "H": 0.3, "F": 0.2},
        # Add more positions as needed
    }

    # Generate MHC sequences with more realistic properties
    mhc_lengths = np.random.randint(340,342, size=batch_size)  # Less variation in length
    mhc_seqs = []
    for length in mhc_lengths:
        seq = []
        for pos in range(length):
            freq = mhc_pos_freq.get(pos, default_aa_freq)
            aa_list = list(AA)
            probs = [freq.get(aa, 0.01) for aa in aa_list]
            probs = np.array(probs) / sum(probs)
            seq.append(np.random.choice(aa_list, p=probs))
        mhc_seqs.append(''.join(seq))

    # Generate MHC embeddings (simulating ESM or similar)
    mhc_EMB = np.random.randn(batch_size, max_mhc_len, 1152).astype(np.float32)
    mhc_OHE = np.array([seq_to_onehot(seq, max_mhc_len) for seq in mhc_seqs], dtype=np.float32)

    # Create masks for MHC sequences
    mask_mhc = np.full((batch_size, max_mhc_len), PAD_TOKEN, dtype=np.float32)
    for i, length in enumerate(mhc_lengths):
        mask_mhc[i, :length] = 1.0
        mhc_EMB[i, length:, :] = 0.0  # Zero out padding positions

    # Generate MHC IDs (could represent allele types)
    mhc_ids = np.random.randint(0, 100, size=(batch_size, max_mhc_len), dtype=np.int32)

    # # mask 0.15 of the peptide positions update the mask with MASK_TOKEN and zero out the corresponding positions in the OHE
    # mask_pep[np.random.rand(batch_size, max_pep_len) < 0.15] = MASK_TOKEN
    # pep_OHE[mask_pep == MASK_TOKEN] = 0.0  # Zero out masked positions
    # # mask 0.15 of the MHC positions update the mask with MASK_TOKEN and zero out the corresponding positions in the EMB
    # mask_mhc[np.random.rand(batch_size, max_mhc_len) < 0.15] = MASK_TOKEN
    # mhc_EMB[mask_mhc == MASK_TOKEN] = 0.0  # Zero out masked positions

    # convert all inputs tensors
    # pep_OHE = tf.convert_to_tensor(pep_OHE, dtype=tf.float32)
    # mask_pep = tf.convert_to_tensor(mask_pep, dtype=tf.float32)
    # mhc_EMB = tf.convert_to_tensor(mhc_EMB, dtype=tf.float32)
    # mask_mhc = tf.convert_to_tensor(mask_mhc, dtype=tf.float32)
    # mhc_OHE = tf.convert_to_tensor(mhc_OHE, dtype=tf.float32)

    # Cov layers
    ks_dict = determine_ks_dict(initial_input_dim=max_mhc_len, output_dims=[16, 14, 12, 11], max_strides=20, max_kernel_size=60)

    # Build models
    encoder, encoder_decoder = bicross_recon(
        max_pep_len=max_pep_len,
        max_mhc_len=max_mhc_len,
        mask_token=MASK_TOKEN,
        pad_token=PAD_TOKEN,
        pep_emb_dim=pep_emb_dim,
        mhc_emb_dim=mhc_emb_dim,
        heads=heads,
    )

    # Print model summaries
    encoder.summary()
    encoder_decoder.summary()

    # Train
    encoder_decoder.fit(x=[pep_OHE, mask_pep, mhc_EMB, mask_mhc],
                        y={'pep_reconstruction': pep_OHE, 'mhc_reconstruction': mhc_OHE},
                        epochs=200, batch_size=batch_size )

    # save the model
    encoder_decoder.save("h5/bicross_encoder_decoder.h5")

    outputs = encoder_decoder.predict([pep_OHE, mask_pep, mhc_EMB, mask_mhc])
    pep_recon = outputs['pep_reconstruction']  # Shape: (batch_size, max_pep_len, 21)
    mhc_recon = outputs['mhc_reconstruction']  # Shape: (batch_size, max_mhc_len, 21)
    # barcoded_mhc_attn = outputs['barcode_mhc_attn']  # Shape: (batch_size, max_pep_len + max_mhc_len, mhc_emb_dim)
    conv_mhc_attn_scores = outputs['mhc_to_pep_attn_score']  # Shape: (batch_size, heads, max_pep_len + max_mhc_len, mhc_emb_dim)
    pep_OHE_m = outputs['pep_OHE_m']  # Masked peptide OHE
    mhc_EMB_m = outputs['mhc_EMB_m']  # Masked MHC embeddings

    # OHE to sequences
    pep_pred = OHE_to_seq(pep_recon) # (B, max_pep_len, 21) -> (B,)
    mhc_pred = OHE_to_seq(mhc_recon)
    pep_orig = OHE_to_seq(pep_OHE)
    mhc_orig = OHE_to_seq(mhc_OHE)

    print("Peptide Reconstruction:")
    print(pep_pred[0], " (original:", pep_orig[0], ")")
    print("MHC Reconstruction:")
    print(mhc_pred[0], " (original:", mhc_orig[0], ")")



    att_outputs = encoder.predict([pep_OHE, mask_pep, mhc_EMB, mask_mhc])
    mhc_att_weights = att_outputs['attention_scores_CA']  # MHC cross-attention weights

    # Verify shapes
    print(f"pep_recon shape: {pep_recon.shape}")  # Expected: (32, 15, 21)
    print(f"mhc_recon shape: {mhc_recon.shape}")  # Expected: (32, 34, 21)
    print(f"mhc_att_weights shape: {mhc_att_weights.shape}")  # Expected: (32, 8, 34, 15)

    # Visualize attention weights for the first sample
    mhc_att_avg = mhc_att_weights[0].mean(axis=0)  # Shape: (34, 15)

    plt.figure(figsize=(6, 4))
    sns.heatmap(mhc_att_avg, cmap='viridis')
    plt.title('Peptide-MHC Cross-Attention')
    plt.xlabel('Peptide Positions')
    plt.ylabel('MHC Positions')
    plt.show()

    # Visualize barcoded MHC attention
    conv_mhc_attn_avg = conv_mhc_attn_scores[0].mean(axis=0)  # Shape: (max_pep_len + max_mhc_len, mhc_emb_dim)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conv_mhc_attn_avg, cmap='viridis')
    plt.title('Convolution pMHC Attention')
    plt.xlabel('Conv Positions')
    plt.ylabel('Conv Positions')
    plt.show()

    # Visualize masked peptide OHE
    plt.figure(figsize=(6, 4))
    sns.heatmap(pep_OHE_m[0], cmap='viridis')
    plt.title('Masked Peptide OHE')
    plt.xlabel('Amino Acid Types')
    plt.ylabel('Peptide Positions')
    plt.show()

    # Visualize masked MHC embeddings
    plt.figure(figsize=(6, 4))
    sns.heatmap(mhc_EMB_m[0], cmap='viridis')
    plt.title('Masked MHC Embeddings')
    plt.xlabel('Embedding Dimensions')
    plt.ylabel('MHC Positions')
    plt.show()
