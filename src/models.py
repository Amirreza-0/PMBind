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

from utils import (seq_to_onehot, AttentionLayer, PositionalEncoding, MaskedEmbedding,
                       ConcatMask, ConcatBarcode, SplitLayer, OHE_to_seq, determine_ks_dict,
                       SelfAttentionWith2DMask, AddGaussianNoise, RotaryPositionalEncoding,
                       SubtractLayer, SubtractAttentionLayer, MaskedCategoricalCrossentropy,
                       masked_categorical_crossentropy, RotaryPositionalEncoding,
                        GlobalSTDPooling1D,GlobalMeanPooling1D, GlobalSTDPooling1D, GumbelSoftmax,
                   AnchorPositionExtractor)


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
    mhc_EMB = MaskedEmbedding(mask_token=mask_token, pad_token=pad_token, name="mhc_masked_embedding1")(mhc_EMB_inp, mask_mhc_inp)

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


#################################### Minimal BiCross Model ###################################
def bicross_recon_mini(max_pep_len: int,
                       max_mhc_len: int,
                       emb_dim: int = 96,
                       heads: int = 4,
                       mask_token: float = MASK_TOKEN,
                       pad_token: float = PAD_TOKEN):

    # -------------------------------------------------------------------
    # INPUTS
    # -------------------------------------------------------------------
    pep_OHE_in = keras.Input((max_pep_len, 21), name="pep_onehot")
    pep_mask_in = keras.Input((max_pep_len,), name="pep_mask")

    mhc_emb_in = keras.Input((max_mhc_len, 1152), name="mhc_latent")
    mhc_mask_in = keras.Input((max_mhc_len,), name="mhc_mask")

    # -------------------------------------------------------------------
    # MASKED  EMBEDDING  +  PE
    # -------------------------------------------------------------------
    pep = MaskedEmbedding(mask_token, pad_token, name="pep_mask2")(pep_OHE_in, pep_mask_in)
    pep = PositionalEncoding(21, int(max_pep_len * 3), name="pep_pos1")(pep, pep_mask_in)
    pep = layers.Dense(emb_dim, name="pep_Dense1")(pep)
    pep = layers.Dropout(0.1, name="pep_Dropout1")(pep)

    mhc = MaskedEmbedding(mask_token, pad_token, name="mhc_mask2")(mhc_emb_in, mhc_mask_in)
    mhc = PositionalEncoding(1152, int(max_mhc_len * 3), name="mhc_pos1")(mhc, mhc_mask_in)
    mhc = layers.Dense(emb_dim, name="mhc_dense1")(mhc)
    mhc = layers.Dropout(0.1, name="mhc_Dropout1")(mhc)

    # -------------------------------------------------------------------
    # Add Gaussian Noise
    # -------------------------------------------------------------------
    pep = AddGaussianNoise(0.1, name="pep_gaussian_noise")(pep)
    mhc = AddGaussianNoise(0.1, name="mhc_gaussian_noise")(mhc)

    # # -------------------------------------------------------------------
    # # Custom Masked Self-Attention Layer
    # # -------------------------------------------------------------------
    pmhc = layers.Concatenate(name="pmhc_concat", axis=-2)([pep, mhc])
    latent_pmhc, att_pmhc = SelfAttentionWith2DMask(query_dim=emb_dim,
                                          context_dim=emb_dim,
                                          output_dim=emb_dim, heads=heads,
                                          return_att_weights=True,
                                          name='SelfAttentionWith2DMask',
                                          epsilon=1e-6,
                                          mask_token=mask_token,
                                          pad_token=-pad_token)(
        pmhc, pep_mask_in, mhc_mask_in
    )

    # -------------------------------------------------------------------
    # Custom Latent reduction Layer
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # RECONSTRUCTION  HEADS
    # -------------------------------------------------------------------
    # Encoder down convolution
    latent_pmhc2 = layers.Dense(32, activation='relu', name='latent_pmhc2')(latent_pmhc)
    latent_pmhc = layers.Dense(emb_dim, activation='relu', name='latent_pmhc3')(latent_pmhc2)
    pmhc_recon = layers.Dense(emb_dim, activation='relu', name='pmhc_recon')(latent_pmhc)

    # Split the pmhc_recon into peptide and MHC reconstructions
    tf.print(max_mhc_len, mhc_mask_in)
    pep_split, mhc_split = SplitLayer([max_pep_len, max_mhc_len], name="pmhc_split")(pmhc_recon)
    pep_softmax = layers.Dense(21, activation='softmax', name="pep_reconstruction")(pep_split)
    mhc_softmax = layers.Dense(21, activation='softmax', name="mhc_reconstruction")(mhc_split)

    # -------------------------------------------------------------------
    # MODELS
    # -------------------------------------------------------------------
    # The encoder for clustering uses the MHC-queried latent space
    encoder = keras.Model([pep_OHE_in, pep_mask_in, mhc_emb_in, mhc_mask_in],
                          {"cross_latent": latent_pmhc2, "attention_pmhc": att_pmhc},
                          name="encoder1")

    enc_dec = keras.Model([pep_OHE_in, pep_mask_in, mhc_emb_in, mhc_mask_in],
                          {"pep_reconstruction": pep_softmax,
                           "mhc_reconstruction": mhc_softmax},
                          name="encoder_decoder")

    enc_dec.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss={"pep_reconstruction": "categorical_crossentropy",
              "mhc_reconstruction": "categorical_crossentropy"}
    )
    return encoder, enc_dec


#################################################################################################
def pmclust_subtract(max_pep_len: int,
                       max_mhc_len: int,
                       emb_dim: int = 21,
                       heads: int = 4,
                       noise_std: float = 0.1,
                       mask_token: float = MASK_TOKEN,
                       pad_token: float = PAD_TOKEN):

    # -------------------------------------------------------------------
    # INPUTS
    # -------------------------------------------------------------------
    pep_OHE_in = keras.Input((max_pep_len, 21), name="pep_onehot")
    pep_mask_in = keras.Input((max_pep_len,), name="pep_mask")

    mhc_emb_in = keras.Input((max_mhc_len, 1152), name="mhc_emb")
    mhc_mask_in = keras.Input((max_mhc_len,), name="mhc_mask")

    mhc_OHE_in = keras.Input((max_mhc_len, 21), name="mhc_onehot")  # Optional input for MHC one-hot encoding

    # -------------------------------------------------------------------
    # MASKED  EMBEDDING  +  PE
    # -------------------------------------------------------------------
    pep = MaskedEmbedding(mask_token, pad_token, name="pep_mask2")(pep_OHE_in, pep_mask_in)
    pep = PositionalEncoding(21, int(max_pep_len * 3), name="pep_pos1")(pep, pep_mask_in)
    pep = layers.Dense(emb_dim, name="pep_Dense1")(pep)
    pep = layers.LayerNormalization(name="pep_norm1")(pep)
    pep = layers.Dropout(0.2, name="pep_Dropout1")(pep)

    mhc = MaskedEmbedding(mask_token, pad_token, name="mhc_mask2")(mhc_emb_in, mhc_mask_in)
    mhc = PositionalEncoding(1152, int(max_mhc_len * 3), name="mhc_pos1")(mhc, mhc_mask_in)
    mhc = layers.Dense(emb_dim, name="mhc_dense1")(mhc)
    mhc = layers.LayerNormalization(name="mhc_norm1")(mhc)
    mhc = layers.Dropout(0.2, name="mhc_Dropout1")(mhc)

    # -------------------------------------------------------------------
    # Subtract Layer
    # -------------------------------------------------------------------
    mhc_subtracted_p = SubtractLayer(name="pmhc_subtract")(pep, pep_mask_in, mhc, mhc_mask_in) # (B, M, P*D) = mhc_expanded – peptide_expanded
    tf.print("mhc_subtracted_p shape:", mhc_subtracted_p.shape)

    # -------------------------------------------------------------------
    # Add Gaussian Noise
    # -------------------------------------------------------------------
    mhc_subtracted_p = AddGaussianNoise(noise_std, name="pmhc_gaussian_noise")(mhc_subtracted_p)

    query_dim = int(emb_dim*max_pep_len)

    # # -------------------------------------------------------------------
    # Normal Self-Attention Layer
    # # -------------------------------------------------------------------
    mhc_subtracted_p_attn, mhc_subtracted_p_attn_scores = AttentionLayer(
        query_dim=query_dim, context_dim=query_dim, output_dim=query_dim,
        type="self", heads=heads, resnet=True,
        return_att_weights=True, name='mhc_subtracted_p_attn',
        mask_token=mask_token,
        pad_token=pad_token
    )(mhc_subtracted_p, mhc_mask_in)
    peptide_cross_att, peptide_cross_attn_scores = AttentionLayer(
        query_dim=int(emb_dim), context_dim=query_dim, output_dim=int(emb_dim),
        type="cross", heads=heads, resnet=False,
        return_att_weights=True, name='peptide_cross_att',
        mask_token=mask_token,
        pad_token=pad_token
    )(pep, pep_mask_in, mhc_subtracted_p_attn, mhc_mask_in)

    # -------------------------------------------------------------------
    # RECONSTRUCTION  HEADS
    # -------------------------------------------------------------------
    latent_sequence = layers.Dense(emb_dim * max_pep_len * 2, activation='relu', name='latent_mhc_dense1')(
        mhc_subtracted_p_attn)
    latent_sequence = layers.LayerNormalization(name="latent_mhc_norm1")(latent_sequence)
    latent_sequence = layers.Dropout(0.2, name='latent_mhc_dropout1')(latent_sequence)
    latent_sequence = layers.Dense(emb_dim, activation='relu', name='cross_latent')(latent_sequence)  # Shape: (B, M, D)

    # --- Latent Vector for Clustering (pooled) ---
    avg_pooled = GlobalMeanPooling1D(name="avg_pooled")(latent_sequence)  # Shape: (B, D)
    std_pooled = GlobalSTDPooling1D(name="std_pooled")(latent_sequence)
    latent_vector = layers.Concatenate(name="latent_vector_concat", axis=-1)([avg_pooled, std_pooled])  # Shape: (B, D*2)

    # --- Reconstruction Heads ---
    mhc_recon_head = layers.Dropout(0.2, name='latent_mhc_dropout2')(latent_sequence)
    mhc_recon = layers.Dense(21, activation='softmax', name='mhc_reconstruction_pred')(mhc_recon_head)
    pep_recon = layers.Dense(emb_dim, activation='relu', name='pep_latent')(peptide_cross_att)
    pep_recon = layers.Dense(21, activation='softmax', name='pep_reconstruction_pred')(pep_recon)

    pep_out = layers.Concatenate(name='pep_ytrue_ypred', axis=-1)([pep_OHE_in, pep_recon])  # (B,P,42)
    mhc_out = layers.Concatenate(name='mhc_ytrue_ypred', axis=-1)([mhc_OHE_in, mhc_recon])  # (B,M,42)

    # -------------------------------------------------------------------
    # MODELS
    # -------------------------------------------------------------------
    enc_dec = keras.Model([pep_OHE_in, pep_mask_in, mhc_emb_in, mhc_mask_in, mhc_OHE_in],
                          {"pep_ytrue_ypred": pep_out,
                           "mhc_ytrue_ypred": mhc_out,
                          "cross_latent": latent_sequence,
                            "latent_vector": latent_vector,
                            "attention_scores_CA": peptide_cross_attn_scores},
                          name="encoder_decoder")

    return enc_dec


def pmclust_cross_attn(max_pep_len: int,
                            max_mhc_len: int,
                            emb_dim: int = 96,
                            heads: int = 4,
                            noise_std: float = 0.1,
                            mask_token: float = MASK_TOKEN,
                            pad_token: float = PAD_TOKEN):
    """
    Builds a pMHC interaction model using self-attention followed by cross-attention.

    This architecture first creates rich, context-aware representations of the peptide
    and MHC sequences independently using self-attention. It then models their
    interaction using two cross-attention layers, allowing the model to learn which
    parts of each sequence are most relevant to the other.
    """
    # -------------------------------------------------------------------
    # INPUTS
    # -------------------------------------------------------------------
    pep_OHE_in = keras.Input((max_pep_len, 21), name="pep_onehot")
    pep_mask_in = keras.Input((max_pep_len,), dtype=tf.float32, name="pep_mask")

    mhc_emb_in = keras.Input((max_mhc_len, 1152), name="mhc_emb")
    mhc_mask_in = keras.Input((max_mhc_len,), dtype=tf.float32, name="mhc_mask")

    # This input is required for the loss function and final output concatenation
    mhc_OHE_in = keras.Input((max_mhc_len, 21), name="mhc_onehot")

    # -------------------------------------------------------------------
    # 1. INITIAL EMBEDDING + POSITIONAL ENCODING
    # -------------------------------------------------------------------
    # Process peptide input
    pep = MaskedEmbedding(mask_token, pad_token, name="pep_mask_embed")(pep_OHE_in, pep_mask_in)
    pep = PositionalEncoding(21, int(max_pep_len * 3), name="pep_pos_enc")(pep, pep_mask_in)
    pep = layers.Dense(emb_dim, name="pep_dense_embed")(pep)
    pep = layers.Dropout(0.1, name="pep_dropout_embed")(pep)

    # Process MHC input
    mhc = MaskedEmbedding(mask_token, pad_token, name="mhc_mask_embed")(mhc_emb_in, mhc_mask_in)
    mhc = PositionalEncoding(1152, int(max_mhc_len * 3), name="mhc_pos_enc")(mhc, mhc_mask_in)
    mhc = layers.Dense(emb_dim, name="mhc_dense_embed")(mhc)
    mhc = layers.Dropout(0.1, name="mhc_dropout_embed")(mhc)

    # -------------------------------------------------------------------
    # 2. INDEPENDENT PROCESSING (SELF-ATTENTION)
    # -------------------------------------------------------------------
    # Create rich representations of each sequence independently
    pep_self_attn = AttentionLayer(
        query_dim=emb_dim, context_dim=emb_dim, output_dim=emb_dim,
        type="self", heads=heads, resnet=True, return_att_weights=False,
        name='pep_self_attention', mask_token=mask_token, pad_token=pad_token
    )(pep, pep_mask_in)

    mhc_self_attn = AttentionLayer(
        query_dim=emb_dim, context_dim=emb_dim, output_dim=emb_dim,
        type="self", heads=heads, resnet=True, return_att_weights=False,
        name='mhc_self_attention', mask_token=mask_token, pad_token=pad_token
    )(mhc, mhc_mask_in)

    # -------------------------------------------------------------------
    # 3. INTERACTION MODELING (CROSS-ATTENTION)
    # -------------------------------------------------------------------
    # Peptide queries the MHC to get a contextualized representation
    contextual_peptide, pep_x_mhc_scores = AttentionLayer(
        query_dim=emb_dim, context_dim=emb_dim, output_dim=emb_dim,
        type="cross", heads=heads, resnet=False, return_att_weights=True,
        name='peptide_cross_attention', mask_token=mask_token, pad_token=pad_token
    )(pep_self_attn, pep_mask_in, mhc_self_attn, mhc_mask_in)

    # MHC queries the peptide to get its own contextualized representation
    contextual_mhc, mhc_x_pep_scores = AttentionLayer(
        query_dim=emb_dim, context_dim=emb_dim, output_dim=emb_dim,
        type="cross", heads=heads, resnet=False, return_att_weights=True,
        name='mhc_cross_attention', mask_token=mask_token, pad_token=pad_token
    )(mhc_self_attn, mhc_mask_in, pep_self_attn, pep_mask_in)

    # -------------------------------------------------------------------
    # 4. LATENT SPACE & RECONSTRUCTION HEADS
    # -------------------------------------------------------------------
    # Add optional noise for robustness, similar to a VAE
    contextual_peptide_noisy = AddGaussianNoise(noise_std, name="pep_gaussian_noise")(contextual_peptide)
    contextual_mhc_noisy = AddGaussianNoise(noise_std, name="mhc_gaussian_noise")(contextual_mhc)

    # --- Latent Space for Visualization (keeps sequence dimension) ---
    # This will be the main latent output for heatmaps and detailed analysis.
    latent_sequence = layers.Dense(emb_dim, activation='relu', name='latent_sequence_output')(contextual_mhc_noisy)

    # --- Latent Vector for Clustering (pooled) ---
    # Create a single vector per sample by pooling the sequence latent.
    latent_vector_pooled = layers.GlobalAveragePooling1D(name="gap_latent")(latent_sequence)
    latent_vector_pooled = layers.Dense(emb_dim * 2, activation='relu', name='latent_dense1')(latent_vector_pooled)
    latent_vector_pooled = layers.Dropout(0.2, name='latent_dropout')(latent_vector_pooled)
    latent_vector_pooled = layers.Dense(emb_dim, activation='relu', name='latent_vector_output')(latent_vector_pooled)

    # --- Reconstruction Heads ---
    # Reconstruct the peptide from its contextual representation
    pep_recon = layers.Dense(emb_dim, activation='relu', name='pep_recon_dense')(contextual_peptide_noisy)
    pep_recon = layers.Dense(21, activation='softmax', name='pep_reconstruction_pred')(pep_recon)

    # Reconstruct the MHC from its contextual representation
    mhc_recon = layers.Dense(emb_dim, activation='relu', name='mhc_recon_dense')(contextual_mhc_noisy)
    mhc_recon = layers.Dense(21, activation='softmax', name='mhc_reconstruction_pred')(mhc_recon)

    # --- Prepare Outputs for Loss Calculation ---
    # Concatenate true and predicted values for the custom loss function
    pep_out = layers.Concatenate(name='pep_ytrue_ypred', axis=-1)([pep_OHE_in, pep_recon])
    mhc_out = layers.Concatenate(name='mhc_ytrue_ypred', axis=-1)([mhc_OHE_in, mhc_recon])

    # -------------------------------------------------------------------
    # 5. MODEL DEFINITION
    # -------------------------------------------------------------------
    # Define the full encoder-decoder model
    enc_dec = keras.Model(
        inputs=[pep_OHE_in, pep_mask_in, mhc_emb_in, mhc_mask_in, mhc_OHE_in],
        outputs={
            "pep_ytrue_ypred": pep_out,
            "mhc_ytrue_ypred": mhc_out,
            "cross_latent": latent_sequence,          # Use the pre-pooled latent for visualization
            "latent_vector": latent_vector_pooled,    # Use the pooled vector for UMAP
            "pep_x_mhc_scores": pep_x_mhc_scores,
            "mhc_x_pep_scores": mhc_x_pep_scores
        },
        name="pmhc_cross_attention_model"
    )

    # Compile the model
    enc_dec.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss={
            "pep_ytrue_ypred": masked_categorical_crossentropy,
            "mhc_ytrue_ypred": masked_categorical_crossentropy
        },
        # You can define loss weights if needed
        # loss_weights={"pep_ytrue_ypred": 1.0, "mhc_ytrue_ypred": 1.0}
    )
    return enc_dec


########################################## MoE ###############################################
class ExpertLayer(keras.layers.Layer):
    """
    Expert layer that applies a dense transformation to the input.

    Args:
        input_dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        output_dim (int): Output dimension.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(ExpertLayer, self).__init__()
        self.w_init = tf.random_normal_initializer(stddev=0.1)
        self.z_init = tf.zeros_initializer()
        self.input_dim = input_dim
        self.dense1 = keras.layers.Dense(
            hidden_dim,  # Changed from output_dim to hidden_dim
            kernel_initializer=self.w_init,
            bias_initializer=self.z_init,
            name='expert_dense1'
        )
        self.dropout1 = keras.layers.Dropout(dropout_rate, name='expert_dropout1')
        self.dense2 = keras.layers.Dense(
            output_dim,  # Keep as output_dim
            kernel_initializer=self.w_init,
            bias_initializer=self.z_init,
            name='expert_dense2'
        )
        self.dropout = keras.layers.Dropout(dropout_rate, name='expert_dropout2')

    def call(self, x, training=False):
        x = self.dense1(x)
        x = tf.nn.relu(x)  # Apply ReLU activation after first layer
        if training:
            x = self.dropout1(x, training=training)
        x = self.dense2(x)
        if training:
            x = self.dropout(x, training=training)
        # Removed final ReLU to allow negative outputs before sigmoid
        return x


class EnhancedMixtureOfExperts(keras.layers.Layer):
    """
    Enhanced Mixture of Experts layer that uses cluster assignments.

    - During training: use hard clustering (vector) to route to a specific expert per sample
    - During inference: use soft clustering to mix experts' weights per sample
    """

    def __init__(self, input_dim, hidden_dim=32, num_experts=16, output_dim=1, dropout_rate=0.2):
        super(EnhancedMixtureOfExperts, self).__init__()
        self.input_dim = input_dim
        #self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.output_dim = output_dim

        #self.experts = [ExpertLayer(self.input_dim, self.hidden_dim, self.output_dim, dropout_rate) for _ in range(self.num_experts)]

    def build(self, inputs):
        self.gate_weight = self.add_weight(shape=(self.input_dim, self.num_experts),
                                           initializer='random_uniform',
                                            trainable=True,
                                            name='gate_weight')
        self.dense_weight = self.add_weight(shape=(self.num_experts, self.input_dim, self.output_dim),
                                            initializer='random_normal',
                                            trainable=True,
                                            name='dense_weight')
        self.bias = self.add_weight(shape=(self.num_experts, self.output_dim),
                                    initializer='zeros',
                                    trainable=True,
                                    name='bias')


    # def hardmax(self, soft_clusters):
    #     """
    #     Convert soft clustering to hard clustering by selecting the max index.
    #     Args:
    #         soft_clusters: Tensor of shape (B, num_experts).
    #     Returns:
    #         Hard clustering tensor of shape (B, num_experts).
    #     """
    #     hard_indices = tf.argmax(soft_clusters, axis=-1)
    #     hard_clusters = tf.one_hot(hard_indices, depth=self.num_experts, dtype=tf.float32)
    #     return hard_clusters

    def call(self, x_batch, training=False):
        """
        We have to generate a one-hot encoding that defines which expert is activated for each sample. (B, num_experts, ohe_num_experts).

        Args:
            x: Input tensor of shape (B, input_dim).
            training: Whether the layer is in training mode.
        Returns:
            Output tensor of shape (B, output_dim).
        """
        gate = tf.matmul(x_batch, self.gate_weight) # (B, num_experts)
        if training:
            gate_probs = tf.nn.softmax(gate / 0.2, axis=-1)
        else:
            gate_probs = tf.nn.softmax(gate, axis=-1)
        gate_probs = tf.expand_dims(gate_probs, axis=-1) #(B, num_experts, 1)
        # tf.print("Gate probabilities shape:", gate_probs.shape, "Gate probabilities:", gate_probs)
        # gate_probs dim = (B, input_dim)
        out = tf.einsum('bi,eio->beo', x_batch, self.dense_weight) #(B, num_experts, output_dim)
        out = out + self.bias #(B, num_experts, output_dim)
        out = tf.multiply(out, gate_probs) #(B, num_experts, output_dim)
        out = tf.reduce_sum(out, axis=1) #(B, output_dim)
        return out, gate_probs



####
def pmbind_subtract_moe_auto(max_pep_len: int,
                               max_mhc_len: int,
                               emb_dim: int = 96,
                               heads: int = 4,
                               noise_std: float = 0.1,
                               num_experts: int = 30,
                               mask_token: float = MASK_TOKEN,
                               pad_token: float = PAD_TOKEN):
    """
    Builds a pMHC autoencoder model with a Mixture-of-Experts (MoE) classifier head.

    This model performs two tasks:
    1. Autoencoding: Reconstructs peptide and MHC sequences from a latent representation.
    2. Classification: Predicts a binary label using an MoE head, where experts are
       selected based on an internally generated clustering of the latent space.
    """
    # -------------------------------------------------------------------
    # INPUTS
    # -------------------------------------------------------------------
    pep_OHE_in = keras.Input((max_pep_len, 21), name="pep_onehot")
    pep_mask_in = keras.Input((max_pep_len,), name="pep_mask")

    mhc_emb_in = keras.Input((max_mhc_len, 1152), name="mhc_emb")
    mhc_mask_in = keras.Input((max_mhc_len,), name="mhc_mask")

    mhc_OHE_in = keras.Input((max_mhc_len, 21), name="mhc_onehot")  # Optional input for MHC one-hot encoding

    # -------------------------------------------------------------------
    # MASKED  EMBEDDING  +  PE
    # -------------------------------------------------------------------
    pep = MaskedEmbedding(mask_token, pad_token, name="pep_mask2")(pep_OHE_in, pep_mask_in)
    pep = PositionalEncoding(21, int(max_pep_len * 3), name="pep_pos1")(pep, pep_mask_in)
    pep = layers.Dense(emb_dim, name="pep_Dense1")(pep)
    pep = layers.LayerNormalization(name="pep_norm1")(pep)
    pep = layers.Dropout(0.2, name="pep_Dropout1")(pep)

    mhc = MaskedEmbedding(mask_token, pad_token, name="mhc_mask2")(mhc_emb_in, mhc_mask_in)
    mhc = PositionalEncoding(1152, int(max_mhc_len * 3), name="mhc_pos1")(mhc, mhc_mask_in)
    mhc = layers.Dense(emb_dim, name="mhc_dense1")(mhc)
    mhc = layers.LayerNormalization(name="mhc_norm1")(mhc)
    mhc = layers.Dropout(0.2, name="mhc_Dropout1")(mhc)

    # -------------------------------------------------------------------
    # Subtract Layer
    # -------------------------------------------------------------------
    mhc_subtracted_p = SubtractLayer(name="pmhc_subtract")(pep, pep_mask_in, mhc,
                                                           mhc_mask_in)  # (B, M, P*D) = mhc_expanded – peptide_expanded
    tf.print("mhc_subtracted_p shape:", mhc_subtracted_p.shape)

    # -------------------------------------------------------------------
    # Add Gaussian Noise
    # -------------------------------------------------------------------
    mhc_subtracted_p = AddGaussianNoise(noise_std, name="pmhc_gaussian_noise")(mhc_subtracted_p)

    query_dim = int(emb_dim * max_pep_len)

    # # -------------------------------------------------------------------
    # Normal Self-Attention Layer
    # # -------------------------------------------------------------------
    mhc_subtracted_p_attn, mhc_subtracted_p_attn_scores = AttentionLayer(
        query_dim=query_dim, context_dim=query_dim, output_dim=query_dim,
        type="self", heads=heads, resnet=True,
        return_att_weights=True, name='mhc_subtracted_p_attn',
        mask_token=mask_token,
        pad_token=pad_token
    )(mhc_subtracted_p, mhc_mask_in)
    peptide_cross_att, peptide_cross_attn_scores = AttentionLayer(
        query_dim=int(emb_dim), context_dim=query_dim, output_dim=int(emb_dim),
        type="cross", heads=heads, resnet=False,
        return_att_weights=True, name='peptide_cross_att',
        mask_token=mask_token,
        pad_token=pad_token
    )(pep, pep_mask_in, mhc_subtracted_p_attn, mhc_mask_in)

    # -------------------------------------------------------------------
    # RECONSTRUCTION  HEADS
    # -------------------------------------------------------------------
    latent_sequence = layers.Dense(emb_dim * max_pep_len * 2, activation='relu', name='latent_mhc_dense1')(
        mhc_subtracted_p_attn)
    latent_sequence = layers.LayerNormalization(name="latent_mhc_norm1")(latent_sequence)
    latent_sequence = layers.Dropout(0.2, name='latent_mhc_dropout1')(latent_sequence)
    latent_sequence = layers.Dense(emb_dim, activation='relu', name='cross_latent')(latent_sequence)  # Shape: (B, M, D)

    # --- Latent Vector for Clustering (pooled) ---
    avg_pooled = GlobalMeanPooling1D(name="avg_pooled")(latent_sequence)  # Shape: (B, D)
    std_pooled = GlobalSTDPooling1D(name="std_pooled")(latent_sequence)
    latent_vector = layers.Concatenate(name="latent_vector_concat", axis=-1)(
        [avg_pooled, std_pooled])  # Shape: (B, max_mhc_len+D)

    # --- Reconstruction Heads ---
    mhc_recon_head = layers.Dropout(0.2, name='latent_mhc_dropout2')(latent_sequence)
    mhc_recon = layers.Dense(21, activation='softmax', name='mhc_reconstruction_pred')(mhc_recon_head)
    pep_recon = layers.Dense(emb_dim, activation='relu', name='pep_latent')(peptide_cross_att)
    pep_recon = layers.Dense(21, activation='softmax', name='pep_reconstruction_pred')(pep_recon)

    pep_out = layers.Concatenate(name='pep_ytrue_ypred', axis=-1)([pep_OHE_in, pep_recon]) #(B,P,42)
    mhc_out = layers.Concatenate(name='mhc_ytrue_ypred', axis=-1)([mhc_OHE_in, mhc_recon]) #(B,M,42)

    # -------------------------------------------------------------------
    # CLASSIFIER HEAD (MIXTURE OF EXPERTS)
    # -------------------------------------------------------------------
    # 1. Gating network: Generate soft cluster assignments from the latent vector
    bigger_probs = layers.Dense(num_experts * 2, activation='relu', name='gating_network_dense1')(latent_vector)
    bigger_probs = layers.Dropout(0.2, name='gating_network_dropout1')(bigger_probs)
    logits = layers.Dense(num_experts, name='gating_network_logits')(bigger_probs)

    #soft_cluster_probs = GumbelSoftmax(name="gumble_softmax")(logits) # Shape: (B, num_experts)
    gate_probs = layers.Activation('softmax', name="soft_cluster_probs")(logits)  # Shape: (B, num_experts)

    # MLP head
    x = layers.Dense(emb_dim, activation='relu', name='cls_dense1')(latent_vector)
    x = layers.Dropout(0.2, name='cls_dropout1')(x)
    x = layers.Dense(emb_dim//2, activation='relu', name='cls_dense2')(x)
    x = layers.Dropout(0.2, name='cls_dropout2')(x)
    y_pred = layers.Dense(1, activation='sigmoid', name='cls_ypred')(x)  # Final logit output

    # 2. MoE layer: Get weighted prediction from experts
    # moe_layer = EnhancedMixtureOfExperts(
    #     input_dim=max_mhc_len+emb_dim,
    #     hidden_dim=emb_dim * 2,
    #     num_experts=num_experts,
    #     output_dim=1,
    #     dropout_rate=0.2
    # )
    # pred_logits, gate_probs = moe_layer(latent_vector) # (B, 1)
    # y_pred = layers.Dense(1, activation='sigmoid', name='cls_ypred')(pred_logits)  # Final logit output

    # -------------------------------------------------------------------
    # MODEL DEFINITION
    # -------------------------------------------------------------------
    model = keras.Model(
        inputs=[pep_OHE_in, pep_mask_in, mhc_emb_in, mhc_mask_in, mhc_OHE_in],
        outputs={
            "pep_ytrue_ypred": pep_out,
            "mhc_ytrue_ypred": mhc_out,
            "latent_vector": latent_vector,
            "soft_cluster_probs": gate_probs,
            "cls_ypred": y_pred,
        },
        name="pmbind_subtract_moe_autoencoder"
    )

    return model


def pmbind_anchor_extractor(max_pep_len: int,
                               max_mhc_len: int,
                               emb_dim: int = 4,
                               heads: int = 4,
                               noise_std: float = 0.1,
                               num_anchors: int = 2, # 2 for MHC1 and 4 for MHC2
                               mask_token: float = MASK_TOKEN,
                               pad_token: float = PAD_TOKEN):
    """
    Builds a pMHC autoencoder model with a Mixture-of-Experts (MoE) classifier head.

    This model performs two tasks:
    1. Autoencoding: Reconstructs peptide and MHC sequences from a latent representation.
    2. Classification: Predicts a binary label using an MoE head, where experts are
       selected based on an internally generated clustering of the latent space.
    """
    # -------------------------------------------------------------------
    # INPUTS
    # -------------------------------------------------------------------
    pep_OHE_in = keras.Input((max_pep_len, 21), name="pep_onehot")
    pep_mask_in = keras.Input((max_pep_len,), name="pep_mask")

    mhc_emb_in = keras.Input((max_mhc_len, 1152), name="mhc_emb")
    mhc_mask_in = keras.Input((max_mhc_len,), name="mhc_mask")

    # -------------------------------------------------------------------
    # MASKED  EMBEDDING  +  PE
    # -------------------------------------------------------------------
    pep = MaskedEmbedding(mask_token, pad_token, name="pep_mask2")(pep_OHE_in, pep_mask_in)
    pep = PositionalEncoding(21, int(max_pep_len * 3), name="pep_pos1")(pep, pep_mask_in)
    pep = layers.Dense(emb_dim, name="pep_Dense1")(pep)
    pep = layers.LayerNormalization(name="pep_norm1")(pep)
    pep = layers.Dropout(0.2, name="pep_Dropout1")(pep) # (B,P,E)

    mhc = MaskedEmbedding(mask_token, pad_token, name="mhc_mask2")(mhc_emb_in, mhc_mask_in)
    mhc = PositionalEncoding(1152, int(max_mhc_len * 3), name="mhc_pos1")(mhc, mhc_mask_in)
    mhc = layers.Dense(emb_dim, name="mhc_dense1")(mhc)
    mhc = layers.LayerNormalization(name="mhc_norm1")(mhc)
    mhc = layers.Dropout(0.2, name="mhc_Dropout1")(mhc) # (B,M,E)

    # -------------------------------------------------------------------
    # Add Gaussian Noise
    # -------------------------------------------------------------------
    mhc = AddGaussianNoise(noise_std, name="pmhc_gaussian_noise")(mhc)

    # self attns
    pep_attn = AttentionLayer(
        query_dim=int(emb_dim), context_dim=int(emb_dim), output_dim=int(emb_dim),
        type="self", heads=heads, resnet=True,
        return_att_weights=False, name='pep_self_att',
        mask_token=mask_token,
        pad_token=pad_token
    )(pep, pep_mask_in)  # (B,P,E)

    mhc_attn = AttentionLayer(
        query_dim=int(emb_dim), context_dim=int(emb_dim), output_dim=int(emb_dim),
        type="self", heads=heads, resnet=True,
        return_att_weights=False, name='mhc_self_att',
        mask_token=mask_token,
        pad_token=pad_token
    )(mhc, mhc_mask_in)  # (B,M,E)

    peptide_cross_att, peptide_cross_attn_scores = AttentionLayer(
        query_dim=int(emb_dim), context_dim=int(emb_dim), output_dim=int(emb_dim),
        type="cross", heads=heads, resnet=False,
        return_att_weights=True, name='peptide_cross_att',
        mask_token=mask_token,
        pad_token=pad_token
    )(pep_attn, pep_mask_in, mhc_attn, mhc_mask_in) # (B,P,E), (B, P, M)

    # dense to reduce dim to 4
    peptide_cross_att = layers.Dense(4, activation="relu", name='peptide_cross_att_dense')(peptide_cross_att)  # (B,P,E)

    # anchor extractor
    outs, inds, weights, barcode_out, barcode_att = AnchorPositionExtractor(
        num_anchors=num_anchors,
        dist_thr=[7, max_pep_len],
        name='anchor_extractor',
        project=False,
        mask_token=mask_token,
        pad_token=pad_token,
        return_att_weights=True
    )(peptide_cross_att, pep_mask_in)  # (B,num_anchors,E), (B,num_anchors), (B,num_anchors), (B,E), (B,P,M)

    # Flatten the outs
    mhc_anchor_flat = layers.Flatten(name='mhc_anchor_flat')(outs)  # (B, num_anchors*E)

    # CLASSIFIER HEAD
    flat_dense = layers.Dense(32, activation='relu', name='latent_vector')(mhc_anchor_flat)  # (B, 32)
    y_pred = layers.Dense(1, activation='sigmoid', name='cls_ypred')(flat_dense)  # (B, 1)

    # -------------------------------------------------------------------
    # MODEL DEFINITION
    # -------------------------------------------------------------------
    model = keras.Model(
        inputs=[pep_OHE_in, pep_mask_in, mhc_emb_in, mhc_mask_in],
        outputs={
            "cls_ypred": y_pred,
            "anchor_positions": inds,
            "anchor_weights": weights,
            "anchor_embeddings": outs,
            "peptide_cross_attn_scores": peptide_cross_attn_scores,
            "barcode_out": barcode_out,
            "barcode_att": barcode_att
        },
        name="pmbind_anchor_extractor"
    )

    return model

####


# Test run with synthetic data
# if __name__ == "__main__":
#     tf.random.set_seed(0)
#     np.random.seed(0)
#
#     # Parameters
#     max_pep_len = 14
#     max_mhc_len = 342
#     batch_size = 32
#     pep_emb_dim = 128
#     mhc_emb_dim = 128
#     heads = 4
#
#     # Generate synthetic data
#     # Define amino acids
#     # Define amino acids
#     AA = "ACDEFGHIKLMNPQRSTVWY"
#
#     # Position-specific amino acid frequencies for peptides
#     # Simplified frequencies where certain positions prefer specific amino acids
#     pep_pos_freq = {
#         0: {"A": 0.3, "G": 0.2, "M": 0.2},  # Position 1 prefers A, G, M
#         1: {"L": 0.3, "V": 0.3, "I": 0.2},  # Position 2 prefers hydrophobic
#         2: {"D": 0.2, "E": 0.2, "N": 0.2},  # Position 3 prefers charged/polar
#     }
#     # Default distribution for other positions
#     default_aa_freq = {aa: 1/len(AA) for aa in AA}
#
#     # Generate peptides with position-specific preferences
#     pep_lengths = np.random.choice([8, 9, 10, 11], size=batch_size, p=[0.1, 0.6, 0.2, 0.1])  # More realistic length distribution
#     pep_seqs = []
#     for length in pep_lengths:
#         seq = []
#         for pos in range(length):
#             # Use position-specific frequencies if available, otherwise default
#             freq = pep_pos_freq.get(pos, default_aa_freq)
#             # Convert frequencies to probability array
#             aa_list = list(AA)
#             probs = [freq.get(aa, 0.01) for aa in aa_list]
#             probs = np.array(probs) / sum(probs)  # Normalize
#             seq.append(np.random.choice(aa_list, p=probs))
#         pep_seqs.append(''.join(seq))
#
#     # Convert peptide sequences to one-hot encoding
#     pep_OHE = np.array([seq_to_onehot(seq, max_pep_len) for seq in pep_seqs], dtype=np.float32)
#     mask_pep = np.full((batch_size, max_pep_len), PAD_TOKEN, dtype=np.float32)
#     for i, length in enumerate(pep_lengths):
#         mask_pep[i, :length] = 1.0
#
#     # MHC alleles typically have conserved regions
#     mhc_pos_freq = {
#         0: {"G": 0.5, "D": 0.3},  # First position often G or D
#         1: {"S": 0.4, "H": 0.3, "F": 0.2},
#         # Add more positions as needed
#     }
#
#     # Generate MHC sequences with more realistic properties
#     mhc_lengths = np.random.randint(340,342, size=batch_size)  # Less variation in length
#     mhc_seqs = []
#     for length in mhc_lengths:
#         seq = []
#         for pos in range(length):
#             freq = mhc_pos_freq.get(pos, default_aa_freq)
#             aa_list = list(AA)
#             probs = [freq.get(aa, 0.01) for aa in aa_list]
#             probs = np.array(probs) / sum(probs)
#             seq.append(np.random.choice(aa_list, p=probs))
#         mhc_seqs.append(''.join(seq))
#
#     # Generate MHC embeddings (simulating ESM or similar)
#     mhc_EMB = np.random.randn(batch_size, max_mhc_len, 1152).astype(np.float32)
#     mhc_OHE = np.array([seq_to_onehot(seq, max_mhc_len) for seq in mhc_seqs], dtype=np.float32)
#
#     # Create masks for MHC sequences
#     mask_mhc = np.full((batch_size, max_mhc_len), PAD_TOKEN, dtype=np.float32)
#     for i, length in enumerate(mhc_lengths):
#         mask_mhc[i, :length] = 1.0
#         mhc_EMB[i, length:, :] = 0.0  # Zero out padding positions
#
#     # Generate MHC IDs (could represent allele types)
#     mhc_ids = np.random.randint(0, 100, size=(batch_size, max_mhc_len), dtype=np.int32)
#
#     # # mask 0.15 of the peptide positions update the mask with MASK_TOKEN and zero out the corresponding positions in the OHE
#     # mask_pep[np.random.rand(batch_size, max_pep_len) < 0.15] = MASK_TOKEN
#     # pep_OHE[mask_pep == MASK_TOKEN] = 0.0  # Zero out masked positions
#     # # mask 0.15 of the MHC positions update the mask with MASK_TOKEN and zero out the corresponding positions in the EMB
#     # mask_mhc[np.random.rand(batch_size, max_mhc_len) < 0.15] = MASK_TOKEN
#     # mhc_EMB[mask_mhc == MASK_TOKEN] = 0.0  # Zero out masked positions
#
#     # convert all inputs tensors
#     # pep_OHE = tf.convert_to_tensor(pep_OHE, dtype=tf.float32)
#     # mask_pep = tf.convert_to_tensor(mask_pep, dtype=tf.float32)
#     # mhc_EMB = tf.convert_to_tensor(mhc_EMB, dtype=tf.float32)
#     # mask_mhc = tf.convert_to_tensor(mask_mhc, dtype=tf.float32)
#     # mhc_OHE = tf.convert_to_tensor(mhc_OHE, dtype=tf.float32)
#
#     # Cov layers
#     ks_dict = determine_ks_dict(initial_input_dim=max_mhc_len, output_dims=[16, 14, 12, 11], max_strides=20, max_kernel_size=60)
#
#     # Build models
#     encoder, encoder_decoder = bicross_recon(
#         max_pep_len=max_pep_len,
#         max_mhc_len=max_mhc_len,
#         mask_token=MASK_TOKEN,
#         pad_token=PAD_TOKEN,
#         pep_emb_dim=pep_emb_dim,
#         mhc_emb_dim=mhc_emb_dim,
#         heads=heads,
#     )
#
#     # Print model summaries
#     encoder.summary()
#     encoder_decoder.summary()
#
#     # Train
#     encoder_decoder.fit(x=[pep_OHE, mask_pep, mhc_EMB, mask_mhc],
#                         y={'pep_reconstruction': pep_OHE, 'mhc_reconstruction': mhc_OHE},
#                         epochs=200, batch_size=batch_size )
#
#     # save the model
#     encoder_decoder.save("h5/bicross_encoder_decoder.h5")
#
#     outputs = encoder_decoder.predict([pep_OHE, mask_pep, mhc_EMB, mask_mhc])
#     pep_recon = outputs['pep_reconstruction']  # Shape: (batch_size, max_pep_len, 21)
#     mhc_recon = outputs['mhc_reconstruction']  # Shape: (batch_size, max_mhc_len, 21)
#     # barcoded_mhc_attn = outputs['barcode_mhc_attn']  # Shape: (batch_size, max_pep_len + max_mhc_len, mhc_emb_dim)
#     conv_mhc_attn_scores = outputs['mhc_to_pep_attn_score']  # Shape: (batch_size, heads, max_pep_len + max_mhc_len, mhc_emb_dim)
#     pep_OHE_m = outputs['pep_OHE_m']  # Masked peptide OHE
#     mhc_EMB_m = outputs['mhc_EMB_m']  # Masked MHC embeddings
#
#     # OHE to sequences
#     pep_pred = OHE_to_seq(pep_recon) # (B, max_pep_len, 21) -> (B,)
#     mhc_pred = OHE_to_seq(mhc_recon)
#     pep_orig = OHE_to_seq(pep_OHE)
#     mhc_orig = OHE_to_seq(mhc_OHE)
#
#     print("Peptide Reconstruction:")
#     print(pep_pred[0], " (original:", pep_orig[0], ")")
#     print("MHC Reconstruction:")
#     print(mhc_pred[0], " (original:", mhc_orig[0], ")")
#
#
#
#     att_outputs = encoder.predict([pep_OHE, mask_pep, mhc_EMB, mask_mhc])
#     mhc_att_weights = att_outputs['attention_scores_CA']  # MHC cross-attention weights
#
#     # Verify shapes
#     print(f"pep_recon shape: {pep_recon.shape}")  # Expected: (32, 15, 21)
#     print(f"mhc_recon shape: {mhc_recon.shape}")  # Expected: (32, 34, 21)
#     print(f"mhc_att_weights shape: {mhc_att_weights.shape}")  # Expected: (32, 8, 34, 15)
#
#     # Visualize attention weights for the first sample
#     mhc_att_avg = mhc_att_weights[0].mean(axis=0)  # Shape: (34, 15)
#
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(mhc_att_avg, cmap='viridis')
#     plt.title('Peptide-MHC Cross-Attention')
#     plt.xlabel('Peptide Positions')
#     plt.ylabel('MHC Positions')
#     plt.show()
#
#     # Visualize barcoded MHC attention
#     conv_mhc_attn_avg = conv_mhc_attn_scores[0].mean(axis=0)  # Shape: (max_pep_len + max_mhc_len, mhc_emb_dim)
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(conv_mhc_attn_avg, cmap='viridis')
#     plt.title('Convolution pMHC Attention')
#     plt.xlabel('Conv Positions')
#     plt.ylabel('Conv Positions')
#     plt.show()
#
#     # Visualize masked peptide OHE
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(pep_OHE_m[0], cmap='viridis')
#     plt.title('Masked Peptide OHE')
#     plt.xlabel('Amino Acid Types')
#     plt.ylabel('Peptide Positions')
#     plt.show()
#
#     # Visualize masked MHC embeddings
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(mhc_EMB_m[0], cmap='viridis')
#     plt.title('Masked MHC Embeddings')
#     plt.xlabel('Embedding Dimensions')
#     plt.ylabel('MHC Positions')
#     plt.show()
