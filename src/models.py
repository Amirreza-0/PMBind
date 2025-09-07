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
                   AnchorPositionExtractor, OHEKmerWindows, Sampling)


MASK_TOKEN = -1.0
PAD_TOKEN = -2.0

############################## pMHC Clustering Model with Subtract Layer ##############################

def pmclust_subtract(max_pep_len: int,
                       max_mhc_len: int,
                       emb_dim: int = 21,
                       heads: int = 4,
                       noise_std: float = 0.1,
                       mask_token: float = MASK_TOKEN,
                       pad_token: float = PAD_TOKEN,
                        esm_dim: int = 1536):

    # -------------------------------------------------------------------
    # INPUTS
    # -------------------------------------------------------------------
    pep_OHE_in = keras.Input((max_pep_len, 21), name="pep_onehot")
    pep_mask_in = keras.Input((max_pep_len,), name="pep_mask")

    mhc_emb_in = keras.Input((max_mhc_len, esm_dim), name="mhc_emb")
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
    mhc = PositionalEncoding(esm_dim, int(max_mhc_len * 3), name="mhc_pos1")(mhc, mhc_mask_in)
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
    latent_vector = layers.Concatenate(name="latent_vector_concat", axis=-1)([avg_pooled, std_pooled])  # Shape: (B, D + max_mhc_len)

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

##################################### Multi-task pMHC Model ###################################

def pmbind_multitask_subtract(max_pep_len: int,
                             max_mhc_len: int,
                             emb_dim: int = 64,
                             heads: int = 4,
                             transformer_layers: int = 2,
                             mask_token: float = MASK_TOKEN,
                             pad_token: float = PAD_TOKEN,
                             noise_std: float = 0.1,
                             latent_dim: int = 128,
                             ESM_dim: int = 1536):
    """
    A multi-task model for semi-supervised pMHC analysis, updated to use
    the custom SelfAttentionWith2DMask layer.
    """
    # -------------------------------------------------------------------
    # INPUTS
    # -------------------------------------------------------------------
    pep_blossom62_in = layers.Input((max_pep_len, 23), name="pep_blossom62")
    pep_mask_in = layers.Input((max_pep_len,), name="pep_mask")
    pep_ohe_target_in = layers.Input((max_pep_len, 21), name="pep_ohe_target")

    mhc_emb_in = layers.Input((max_mhc_len, ESM_dim), name="mhc_emb")
    mhc_mask_in = layers.Input((max_mhc_len,), name="mhc_mask")
    mhc_ohe_target_in = layers.Input((max_mhc_len, 21), name="mhc_ohe_target")

    # -------------------------------------------------------------------
    # SHARED ENCODER
    # -------------------------------------------------------------------
    # 1. Embed Peptide and MHC to the same dimension
    pep = MaskedEmbedding(mask_token, pad_token, name="pep_mask_embed")(pep_blossom62_in, pep_mask_in)
    pep = PositionalEncoding(23, int(max_pep_len * 2), name="pep_pos_enc")(pep, pep_mask_in)
    pep = layers.Dense(emb_dim, name="pep_dense_embed")(pep)
    pep = layers.Dropout(0.2, name="pep_dropout")(pep)

    mhc = MaskedEmbedding(mask_token, pad_token, name="mhc_mask_embed")(mhc_emb_in, mhc_mask_in)
    mhc = PositionalEncoding(ESM_dim, int(max_mhc_len * 2), name="mhc_pos_enc")(mhc, mhc_mask_in)
    mhc = layers.Dense(emb_dim, name="mhc_dense_embed")(mhc)
    mhc = layers.Dropout(0.2, name="mhc_dropout")(mhc)
    mhc = layers.LayerNormalization()(mhc)

    # add Gaussian noise
    mhc = AddGaussianNoise(noise_std, name="mhc_gaussian_noise")(mhc)

    pmhc_concat = layers.Concatenate(axis=1, name="pmhc_concat")([pep, mhc])  # (B, P+M, D)

    # pmhc self-attention with 2D mask
    pmhc_interaction, pmhc_attn_weights = SelfAttentionWith2DMask(
        query_dim=emb_dim,
        context_dim=emb_dim,
        output_dim=emb_dim,
        heads=heads,
        return_att_weights=True,
        self_attn_mhc=False,  # Prevent both peptide and MHC self-attention in this layer
        apply_rope=True,
        name="pmhc_2d_masked_attention"
    )(pmhc_concat, pep_mask_in, mhc_mask_in)


    # The final output of the encoder is our shared latent sequence
    latent_sequence = pmhc_interaction  # Shape: (B, P+M, D)
    latent_sequence = layers.BatchNormalization(name="latent_sequence_norm")(latent_sequence)

    # Subtract Layer to produce clustering-friendly latent space
    # Split back into peptide and MHC parts
    pep_interacted = layers.Lambda(
        lambda x: x[:, :max_pep_len, :],
        name="extract_pep_interacted"
    )(latent_sequence) # Shape: (B, P, D)

    mhc_interacted = layers.Lambda(
        lambda x: x[:, max_pep_len:, :],
        name="extract_mhc_interacted"
    )(latent_sequence) # Shape: (B, M, D)

    # --- Interaction Module 3: Difference-based Compatibility ---
    # Using SubtractLayer to compute element-wise differences
    pmhc_subtracted = SubtractLayer(name="compatibility_difference")(
        pep_interacted, pep_mask_in, mhc_interacted, mhc_mask_in
    )  # Shape: (B, M, P*latent_dim)

    # make 4d (B, M, P, D)
    pmhc_subtracted = keras.layers.Reshape((max_mhc_len, max_pep_len, emb_dim), name="pmhc_subtracted_reshape")(pmhc_subtracted) # (B, M, P, D)
    # permute to (B, P, M, D)
    pmhc_subtracted = keras.layers.Permute((2, 1, 3), name="pmhc_subtracted_permute")(pmhc_subtracted) # (B, P, M, D)

    # join back to 2d (B, P+M, D)
    # Reduce interaction tensor to per-position embeddings before concatenation.
    # (B, P, M, D) -> (B, P, D) and (B, M, D), then concat -> (B, P+M, D)
    pep_from_interactions = layers.Lambda(lambda x: tf.reduce_mean(x, axis=2),
                                          name="pep_from_interactions")(pmhc_subtracted)  # (B, P, D)
    mhc_from_interactions = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1),
                                          name="mhc_from_interactions")(pmhc_subtracted)  # (B, M, D)
    pmhc_subtracted = layers.Concatenate(axis=1, name="pmhc_subtracted_to_2d")(
        [pep_from_interactions, mhc_from_interactions])  # (B, P+M, D)

    # pool on axis 1 # TODO this part directly affects clustering
    pooled_latent = GlobalMeanPooling1D(name="latent_sequence_pooling", axis=-1)(pmhc_subtracted) # (B, P+M)

    # pooled latent vector for classification
    binding_head = layers.Dense(emb_dim, activation="relu", name="binding_dense1")(pooled_latent) # (B, D)
    binding_head = layers.GaussianDropout(0.2, name="binding_gaussian_dropout")(binding_head)
    binding_pred = layers.Dense(1, activation="sigmoid", name="binding_pred")(binding_head) # (B, 1)

    # -------------------------------------------------------------------
    # TASK 2: RECONSTRUCTION HEAD
    # -------------------------------------------------------------------
    pep_latent_seq = latent_sequence[:, :max_pep_len, :] # (B, P, D)
    mhc_latent_seq = latent_sequence[:, max_pep_len:, :] # (B, M, D)

    pep_recon_head = layers.Dense(emb_dim * 2, activation='relu')(pep_latent_seq)
    pep_recon_head = layers.Dropout(0.2, name='pep_recon_dropout')(pep_recon_head)
    pep_recon_pred = layers.Dense(21, activation='softmax', name='pep_reconstruction_pred')(pep_recon_head)

    mhc_recon_head = layers.Dense(emb_dim * 2, activation='relu')(mhc_latent_seq)
    mhc_recon_head = layers.Dropout(0.2, name='mhc_recon_dropout')(mhc_recon_head)
    mhc_recon_pred = layers.Dense(21, activation='softmax', name='mhc_reconstruction_pred')(mhc_recon_head)

    pep_out = layers.Concatenate(name='pep_ytrue_ypred')([pep_ohe_target_in, pep_recon_pred])
    mhc_out = layers.Concatenate(name='mhc_ytrue_ypred')([mhc_ohe_target_in, mhc_recon_pred])

    # -------------------------------------------------------------------
    # MODEL DEFINITION
    # -------------------------------------------------------------------
    pMHC_multitask_model = keras.Model(
        inputs=[pep_blossom62_in, pep_mask_in, mhc_emb_in, mhc_mask_in, pep_ohe_target_in, mhc_ohe_target_in],
        outputs={
            "pep_ytrue_ypred": pep_out,
            "mhc_ytrue_ypred": mhc_out,
            "cls_ypred": binding_pred, # (B, 1)
            "attn_weights": pmhc_attn_weights, # (B, heads, P+M, P+M)
            "latent_vector": pooled_latent, # (B, P+M+D)
            "latent_seq": latent_sequence # (B, P+M, D)
        },
        name="pMHC_Multitask_Transformer_Custom_Attention"
    )

    return pMHC_multitask_model


def pmbind_multitask(max_pep_len: int,
                     max_mhc_len: int,
                     emb_dim: int = 64,
                     heads: int = 4,
                     transformer_layers: int = 2,
                     mask_token: float = MASK_TOKEN,
                     pad_token: float = PAD_TOKEN,
                     noise_std: float = 0.1,
                     latent_dim: int = 128,
                     drop_out_rate: float = 0.2,
                     ESM_dim: int = 1536):
    """
    A multi-task model for semi-supervised pMHC analysis, updated to use
    the custom SelfAttentionWith2DMask layer.
    """
    # -------------------------------------------------------------------
    # INPUTS
    # -------------------------------------------------------------------
    pep_blossom62_in = layers.Input((max_pep_len, 23), name="pep_blossom62")
    pep_mask_in = layers.Input((max_pep_len,), name="pep_mask")
    pep_ohe_target_in = layers.Input((max_pep_len, 21), name="pep_ohe_target")

    mhc_emb_in = layers.Input((max_mhc_len, ESM_dim), name="mhc_emb")
    mhc_mask_in = layers.Input((max_mhc_len,), name="mhc_mask")
    mhc_ohe_target_in = layers.Input((max_mhc_len, 21), name="mhc_ohe_target")

    # -------------------------------------------------------------------
    # SHARED ENCODER
    # -------------------------------------------------------------------
    # 1. Embed Peptide and MHC to the same dimension
    pep = MaskedEmbedding(mask_token, pad_token, name="pep_mask_embed")(pep_blossom62_in, pep_mask_in)
    pep = PositionalEncoding(23, int(max_pep_len * 3), name="pep_pos_enc")(pep, pep_mask_in)
    pep = layers.Dense(emb_dim, name="pep_dense_embed")(pep)
    pep = layers.Dropout(drop_out_rate, name="pep_dropout")(pep)

    mhc = MaskedEmbedding(mask_token, pad_token, name="mhc_mask_embed")(mhc_emb_in, mhc_mask_in)
    mhc = PositionalEncoding(ESM_dim, int(max_mhc_len * 3), name="mhc_pos_enc")(mhc, mhc_mask_in)
    mhc = layers.Dense(emb_dim, name="mhc_dense_embed")(mhc)
    mhc = layers.Dropout(drop_out_rate, name="mhc_dropout")(mhc)
    mhc = layers.LayerNormalization(name="mhc_layer_norm")(mhc)

    # add Gaussian noise
    pep = AddGaussianNoise(noise_std, name="pep_gaussian_noise")(pep)
    mhc = AddGaussianNoise(noise_std, name="mhc_gaussian_noise")(mhc)

    pmhc_concat = layers.Concatenate(axis=1, name="pmhc_concat")([pep, mhc])  # (B, P+M, D)

    # pmhc self-attention with 2D mask
    pmhc_interaction, pmhc_attn_weights = SelfAttentionWith2DMask(
        query_dim=emb_dim,
        context_dim=emb_dim,
        output_dim=emb_dim,
        heads=heads,
        return_att_weights=True,
        self_attn_mhc=False,  # Prevent both peptide and MHC self-attention in this layer
        apply_rope=True,
        name="pmhc_2d_masked_attention"
    )(pmhc_concat, pep_mask_in, mhc_mask_in)


    # The final output of the encoder is our shared latent sequence
    latent_sequence = pmhc_interaction  # Shape: (B, P+M, D)
    latent_sequence = layers.BatchNormalization(name="latent_sequence_norm")(latent_sequence)

    # -------------------------------------------------------------------
    # TASK 1: BINDING PREDICTION HEAD # TODO this part directly affects clustering
    # -------------------------------------------------------------------
    # pooled_mean2 = GlobalMeanPooling1D(name="latent_vector_pool-2", axis=-2)(latent_sequence) # (B, D)
    pooled_std2 = GlobalSTDPooling1D(name="latent_vector_std-2", axis=-2)(latent_sequence) # (B, D)
    pooled_mean1 = GlobalMeanPooling1D(name="latent_vector_pool-1", axis=-1)(latent_sequence)  # (B, P+M)
    # pooled_std1 = GlobalSTDPooling1D(name="latent_vector_std-1", axis=-1)(latent_sequence)  # (B, P+M)

    # concatenate mean and std pooled vectors
    pooled_latent = layers.Concatenate(name="pooled_latent_concat", axis=-1)([pooled_std2, pooled_mean1]) # (B, 1*(P+M+D))

    binding_head = layers.Dense(emb_dim, activation="relu", name="binding_dense1")(pooled_latent)
    binding_head = layers.GaussianDropout(drop_out_rate, name="binding_gaussian_dropout")(binding_head)
    binding_pred = layers.Dense(1, activation="sigmoid", name="binding_pred")(binding_head) # (B, 1)

    # -------------------------------------------------------------------
    # TASK 2: RECONSTRUCTION HEAD
    # -------------------------------------------------------------------
    pep_latent_seq = latent_sequence[:, :max_pep_len, :] # (B, P, D)
    mhc_latent_seq = latent_sequence[:, max_pep_len:, :] # (B, M, D)

    pep_recon_head = layers.Dense(emb_dim * 2, activation='relu')(pep_latent_seq)
    pep_recon_head = layers.Dropout(drop_out_rate, name='pep_recon_dropout')(pep_recon_head)
    pep_recon_pred = layers.Dense(21, activation='softmax', name='pep_reconstruction_pred')(pep_recon_head)

    mhc_recon_head = layers.Dense(emb_dim * 2, activation='relu')(mhc_latent_seq)
    mhc_recon_head = layers.Dropout(drop_out_rate, name='mhc_recon_dropout')(mhc_recon_head)
    mhc_recon_pred = layers.Dense(21, activation='softmax', name='mhc_reconstruction_pred')(mhc_recon_head)

    pep_out = layers.Concatenate(name='pep_ytrue_ypred')([pep_ohe_target_in, pep_recon_pred])
    mhc_out = layers.Concatenate(name='mhc_ytrue_ypred')([mhc_ohe_target_in, mhc_recon_pred])

    # -------------------------------------------------------------------
    # MODEL DEFINITION
    # -------------------------------------------------------------------
    pMHC_multitask_model = keras.Model(
        inputs=[pep_blossom62_in, pep_mask_in, mhc_emb_in, mhc_mask_in, pep_ohe_target_in, mhc_ohe_target_in],
        outputs={
            "pep_ytrue_ypred": pep_out,
            "mhc_ytrue_ypred": mhc_out,
            "cls_ypred": binding_pred,
            "attn_weights": pmhc_attn_weights, # (B, heads, P+M, P+M)
            "latent_vector": pooled_latent, # (B, P+M+D)
            "latent_seq": latent_sequence # (B, P+M, D)
        },
        name="pMHC_Multitask_Transformer_Custom_Attention"
    )

    return pMHC_multitask_model


# def pmbind_multitask_co_attn(max_pep_len: int,
#                             max_mhc_len: int,
#                             emb_dim: int = 64,
#                             heads: int = 4,
#                             transformer_layers: int = 2,
#                             mask_token: float = MASK_TOKEN,
#                             pad_token: float = PAD_TOKEN,
#                             noise_std: float = 0.1,
#                             latent_dim: int = 128,
#                             drop_out_rate: float = 0.2,
#                             ESM_dim: int = 1536):
#     """
#     A multi-task model for semi-supervised pMHC analysis, updated to use
#     the custom SelfAttentionWith2DMask layer.
#     """
#     # -------------------------------------------------------------------
#     # INPUTS
#     # -------------------------------------------------------------------
#     pep_blossom62_in = layers.Input((max_pep_len, 23), name="pep_blossom62")
#     pep_mask_in = layers.Input((max_pep_len,), name="pep_mask")
#     pep_ohe_target_in = layers.Input((max_pep_len, 21), name="pep_ohe_target")
#
#     mhc_emb_in = layers.Input((max_mhc_len, ESM_dim), name="mhc_emb")
#     mhc_mask_in = layers.Input((max_mhc_len,), name="mhc_mask")
#     mhc_ohe_target_in = layers.Input((max_mhc_len, 21), name="mhc_ohe_target")
#
#     # -------------------------------------------------------------------
#     # SHARED ENCODER
#     # -------------------------------------------------------------------
#     # 1. Embed Peptide and MHC to the same dimension
#     pep = MaskedEmbedding(mask_token, pad_token, name="pep_mask_embed")(pep_blossom62_in, pep_mask_in)
#     pep = PositionalEncoding(23, int(max_pep_len * 2), name="pep_pos_enc")(pep, pep_mask_in)
#     pep = layers.Dense(emb_dim, name="pep_dense_embed")(pep)
#     pep = layers.Dropout(drop_out_rate, name="pep_dropout")(pep)
#
#     mhc = MaskedEmbedding(mask_token, pad_token, name="mhc_mask_embed")(mhc_emb_in, mhc_mask_in)
#     mhc = PositionalEncoding(ESM_dim, int(max_mhc_len * 2), name="mhc_pos_enc")(mhc, mhc_mask_in)
#     mhc = layers.Dense(emb_dim, name="mhc_dense_embed")(mhc)
#     mhc = layers.Dropout(drop_out_rate, name="mhc_dropout")(mhc)
#     mhc = layers.LayerNormalization()(mhc)
#
#     # add Gaussian noise
#     pep = AddGaussianNoise(noise_std, name="pep_gaussian_noise")(pep)
#     mhc = AddGaussianNoise(noise_std, name="mhc_gaussian_noise")(mhc)
#
#     pmhc_concat = layers.Concatenate(axis=1, name="pmhc_concat")([pep, mhc])  # (B, P+M, D)
#
#     # pmhc self-attention with 2D mask
#     pmhc_interaction, pmhc_attn_weights = SelfAttentionWith2DMask(
#         query_dim=emb_dim,
#         context_dim=emb_dim,
#         output_dim=emb_dim,
#         heads=heads,
#         return_att_weights=True,
#         self_attn_mhc=False,  # Prevent both peptide and MHC self-attention in this layer
#         apply_rope=True,
#         name="pmhc_2d_masked_attention"
#     )(pmhc_concat, pep_mask_in, mhc_mask_in)
#
#
#     # The final output of the encoder is our shared latent sequence
#     latent_sequence = pmhc_interaction  # Shape: (B, P+M, D)
#     latent_sequence = layers.BatchNormalization(name="latent_sequence_norm")(latent_sequence)
#
#     # -------------------------------------------------------------------
#     # TASK 1: BINDING PREDICTION HEAD # TODO this part directly affects clustering
#     # -------------------------------------------------------------------
#     pooled_mean2 = GlobalMeanPooling1D(name="latent_vector_pool-2", axis=-2)(latent_sequence) # (B, P+M)
#     # pooled_std2 = GlobalSTDPooling1D(name="latent_vector_std-2", axis=-2)(latent_sequence) # (B, P+M)
#     # pooled_mean1 = GlobalMeanPooling1D(name="latent_vector_pool-1", axis=-1)(latent_sequence)  # (B, D)
#     pooled_std1 = GlobalSTDPooling1D(name="latent_vector_std-1", axis=-1)(latent_sequence)  # (B, D)
#
#     # concatenate mean and std pooled vectors
#     pooled_latent = layers.Concatenate(name="pooled_latent_concat", axis=-1)([pooled_std1, pooled_mean2]) # (B, 1*(P+M+D))
#
#     binding_head = layers.Dense(emb_dim, activation="relu", name="binding_dense1")(pooled_latent)
#     binding_head = layers.GaussianDropout(drop_out_rate, name="binding_gaussian_dropout")(binding_head)
#     binding_pred = layers.Dense(1, activation="sigmoid", name="binding_pred")(binding_head) # (B, 1)
#
#     # -------------------------------------------------------------------
#     # TASK 2: RECONSTRUCTION HEAD
#     # -------------------------------------------------------------------
#     pep_latent_seq = latent_sequence[:, :max_pep_len, :] # (B, P, D)
#     mhc_latent_seq = latent_sequence[:, max_pep_len:, :] # (B, M, D)
#
#     pep_recon_head = layers.Dense(emb_dim * 2, activation='relu')(pep_latent_seq)
#     pep_recon_head = layers.Dropout(drop_out_rate, name='pep_recon_dropout')(pep_recon_head)
#     pep_recon_pred = layers.Dense(21, activation='softmax', name='pep_reconstruction_pred')(pep_recon_head)
#
#     mhc_recon_head = layers.Dense(emb_dim * 2, activation='relu')(mhc_latent_seq)
#     mhc_recon_head = layers.Dropout(drop_out_rate, name='mhc_recon_dropout')(mhc_recon_head)
#     mhc_recon_pred = layers.Dense(21, activation='softmax', name='mhc_reconstruction_pred')(mhc_recon_head)
#
#     pep_out = layers.Concatenate(name='pep_ytrue_ypred')([pep_ohe_target_in, pep_recon_pred])
#     mhc_out = layers.Concatenate(name='mhc_ytrue_ypred')([mhc_ohe_target_in, mhc_recon_pred])
#
#     # -------------------------------------------------------------------
#     # MODEL DEFINITION
#     # -------------------------------------------------------------------
#     pMHC_multitask_model = keras.Model(
#         inputs=[pep_blossom62_in, pep_mask_in, mhc_emb_in, mhc_mask_in, pep_ohe_target_in, mhc_ohe_target_in],
#         outputs={
#             "pep_ytrue_ypred": pep_out,
#             "mhc_ytrue_ypred": mhc_out,
#             "cls_ypred": binding_pred,
#             "attn_weights": pmhc_attn_weights, # (B, heads, P+M, P+M)
#             "latent_vector": pooled_latent, # (B, P+M+D)
#             "latent_seq": latent_sequence # (B, P+M, D)
#         },
#         name="pMHC_Multitask_Transformer_Custom_Attention"
#     )
#
#     return pMHC_multitask_model
