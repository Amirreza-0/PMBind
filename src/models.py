#!/usr/bin/env python
"""
pepmhc_cross_attention.py
-------------------------

Minimal end-to-end peptide × MHC cross-attention reconstruction model
with explainable attention visualisation.

Author: Amirreza 2025-07-08
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import (
    MASK_TOKEN,
    PAD_TOKEN,
    PositionalEncoding,
    MaskedEmbedding,
    GlobalMeanPooling1D,
    GlobalSTDPooling1D,
    SelfAttentionWith2DMask
)



def pmbind_multitask(max_pep_len: int,
                              max_mhc_len: int,
                              emb_dim: int = 32,
                              heads: int = 2,
                              transformer_layers: int = 2,
                              mask_token: float = MASK_TOKEN,
                              pad_token: float = PAD_TOKEN,
                              noise_std: float = 0.1,
                              latent_dim: int = 128,
                              drop_out_rate: float = 0.2,
                              l2_reg: float = 0.01,
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
    # pep = layers.LayerNormalization(name="pep_norm_input1")(pep)
    pep = PositionalEncoding(23, int(max_pep_len * 3), name="pep_pos_enc")(pep, pep_mask_in)
    pep = layers.GaussianNoise(noise_std, name="pep_gaussian_noise")(pep)
    pep = layers.SpatialDropout1D(drop_out_rate, name="pep_dropout")(
        pep)  # SpatialDropout1D for sequence data to drop entire feature maps
    pep = layers.Dense(emb_dim, name="pep_dense_embed")(pep)
    pep = layers.Dropout(drop_out_rate)(pep)

    mhc = MaskedEmbedding(mask_token, pad_token, name="mhc_mask_embed")(mhc_emb_in, mhc_mask_in)
    # mhc = layers.LayerNormalization(name="mhc_norm_layer_inp")(mhc)
    mhc = layers.GaussianNoise(noise_std, name="mhc_gaussian_noise")(mhc)
    mhc = PositionalEncoding(ESM_dim, int(max_mhc_len * 3), name="mhc_pos_enc")(mhc, mhc_mask_in)
    mhc = layers.SpatialDropout1D(drop_out_rate, name="mhc_dropout")(
        mhc)  # SpatialDropout1D for sequence data to drop entire feature maps
    mhc = layers.Dense(emb_dim, name="mhc_dense_embed")(mhc)
    mhc = layers.Dropout(drop_out_rate)(mhc)
    #mhc = layers.LayerNormalization(name="mhc_layer_norm")(mhc)

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
    #latent_sequence = layers.BatchNormalization(name="latent_sequence_norm")(latent_sequence)

    # -------------------------------------------------------------------
    # TASK 1: BINDING PREDICTION HEAD # this part directly affects clustering
    # -------------------------------------------------------------------
    # pooled_latent = GlobalMaxPooling1D(name="latent_vector_pool-2", axis=-2)(latent_sequence) # (B, D) # Tells how strong the overall signal is across the sequence (using this forces the model to pu one feature as the distance/binder information)
    pooled_std2 = GlobalSTDPooling1D(name="latent_vector_std-2", axis=-2)(
       latent_sequence)  # (B, D) # Tells how much variation there is in the signal across the sequence
    pooled_mean1 = GlobalMeanPooling1D(name="latent_vector_pool-1", axis=-1)(
       latent_sequence)  # (B, P+M) # Tells how strong the overall signal is across the features
    # pooled_std1 = GlobalSTDPooling1D(name="latent_vector_std-1", axis=-1)(latent_sequence)  # (B, P+M) # Tells how much variation there is in the signal across the features

    # concatenate mean and std pooled vectors
    pooled_latent = layers.Concatenate(name="pooled_latent_concat", axis=-1)(
       [pooled_mean1, pooled_std2])  # (B, 1*(D+P+M))

    binding_head = layers.Dropout(drop_out_rate * 1.5, name='dropout_pooled_latent')(pooled_latent)
    binding_head = layers.Dense(emb_dim // 2, activation="gelu", name="binding_dense1",
                                kernel_regularizer=keras.regularizers.l2(l2_reg),
                                bias_regularizer=keras.regularizers.l2(l2_reg))(binding_head)
    binding_head = layers.Dropout(drop_out_rate, name="binding_dropout2")(binding_head)
    binding_pred = layers.Dense(1, activation="sigmoid", name="binding_pred", dtype="float32")(binding_head)  # (B, 1)

    # -------------------------------------------------------------------
    # TASK 2: RECONSTRUCTION HEAD
    # -------------------------------------------------------------------
    pep_latent_seq = latent_sequence[:, :max_pep_len, :]  # (B, P, D)
    mhc_latent_seq = latent_sequence[:, max_pep_len:, :]  # (B, M, D)

    pep_recon_head = layers.Dropout(drop_out_rate, name='pep_recon_dropout_1')(pep_latent_seq)
    pep_recon_head = layers.Dense(emb_dim * 2, activation='relu')(pep_recon_head)
    pep_recon_head = layers.Dropout(drop_out_rate, name='pep_recon_dropout_2')(pep_recon_head)
    pep_recon_pred = layers.Dense(21, activation='softmax', name='pep_reconstruction_pred')(pep_recon_head)

    mhc_recon_head = layers.Dropout(drop_out_rate, name='mhc_recon_dropout_1')(mhc_latent_seq)
    mhc_recon_head = layers.Dense(emb_dim * 2, activation='relu')(mhc_recon_head)
    mhc_recon_head = layers.Dropout(drop_out_rate, name='mhc_recon_dropout_2')(mhc_recon_head)
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
            "attn_weights": pmhc_attn_weights,  # (B, heads, P+M, P+M)
            "latent_vector": pooled_latent,  # (B, P+M+D)
            "latent_seq": latent_sequence  # (B, P+M, D)
        },
        name="pMHC_Multitask_Transformer_Custom_Attention"
    )

    return pMHC_multitask_model