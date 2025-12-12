#!/usr/bin/env python
"""
models.py
-------------------------

End-to-end peptide × MHC cross-attention reconstruction model.
This multi-task model architecture jointly learns a binary binding label (classification) and per-residue reconstruction for peptide. A shared transformer-style encoder with a custom 2D-masked cross-attention produces a latent sequence that supports clustering and downstream analyses; the model exposes attention weights and pooled latent vectors for explainability, visualization, and unsupervised / classification guided clustering of pMHCs..

Author: Amirreza Aleyasin 2025-12-12
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
    GlobalMaxPooling1D,
    AttentionLayer,
    SelfAttentionWith2DMaskRMSNorm,
    SpatialTemporalDropout1D,
    SubtractLayer
)


def pmbind_multitask_v12(max_pep_len: int,
                         max_mhc_len: int,
                         pep_dim: int = 14,
                         emb_dim: int = 32,
                         heads: int = 8,
                         transformer_layers: int = 2,
                         mask_token: float = MASK_TOKEN,
                         pad_token: float = PAD_TOKEN,
                         noise_std: float = 0.1,
                         latent_dim: int = 32,
                         drop_out_rate: float = 0.18,
                         l2_reg: float = 0.01,
                         ESM_dim: int = 14,
                         return_logits: bool = False):
    """
    A multi-task model for pMHC reconstruction and binding prediction.
    """
    # -------------------------------------------------------------------
    # INPUTS
    # -------------------------------------------------------------------
    pep_in = layers.Input((max_pep_len, pep_dim), name="pep_emb")  # (B, P, D)
    pep_mask_in = layers.Input((max_pep_len,), name="pep_mask")  # (B, P)
    pep_ohe_target_in = layers.Input((max_pep_len, 21),
                                     name="pep_ohe_target")  # (B, P, 21) one-hot encoded target for reconstruction

    mhc_emb_in = layers.Input((max_mhc_len, ESM_dim), name="mhc_emb")  # (B, M, ESM_dim)
    mhc_mask_in = layers.Input((max_mhc_len,), name="mhc_mask")  # (B, M)

    # -------------------------------------------------------------------
    # input processing
    # -------------------------------------------------------------------
    # 1. Embed Peptide and MHC to the same dimension
    pep = MaskedEmbedding(mask_token, pad_token, name="pep_mask_embed")(pep_in, pep_mask_in)  # (B, P, D)
    pep = PositionalEncoding(pep_dim, int(max_pep_len * 3), name="pep_pos_enc")(pep, pep_mask_in)  # (B, P, D)
    pep = layers.GaussianNoise(noise_std, name="pep_gaussian_noise")(pep)  # (B, P, D)
    pep = layers.SpatialDropout1D(drop_out_rate, name="pep_dropout")(
        pep)  # SpatialDropout1D for sequence data to drop entire feature maps

    mhc = MaskedEmbedding(mask_token, pad_token, name="mhc_mask_embed")(mhc_emb_in, mhc_mask_in)  # (B, M, ESM_dim)
    mhc = layers.GaussianNoise(noise_std, name="mhc_gaussian_noise")(mhc)  # (B, M, ESM_dim)
    mhc = PositionalEncoding(ESM_dim, int(max_mhc_len * 3), name="mhc_pos_enc")(mhc, mhc_mask_in)  # (B, M, ESM_dim)
    mhc = layers.SpatialDropout1D(drop_out_rate, name="mhc_dropout")(
        mhc)  # SpatialDropout1D for sequence data to drop entire feature maps

    # -------------------------------------------------------------------
    # Subtract pep from MHC embeddings
    # -------------------------------------------------------------------
    mhc_subtracted = SubtractLayer(name="mhc_subtract_pep")(pep, pep_mask_in, mhc, mhc_mask_in)  # (B, M, P*D)
    # P*D is now our new MHC embedding dimension after subtraction

    # layer norm
    mhc_subtracted = layers.LayerNormalization(epsilon=1e-6, name="mhc_subtract_layernorm")(mhc_subtracted)

    mhc_subtracted = layers.Dense(latent_dim, activation="gelu",
                                  kernel_regularizer=keras.regularizers.l2(l2_reg),
                                  bias_regularizer=keras.regularizers.l2(l2_reg),
                                  name="mhc_subtract_dense")(mhc_subtracted)  # (B, M, L)

    # Normal self attention
    pmhc_interaction, pmhc_attn_weights = AttentionLayer(
        query_dim=latent_dim,
        context_dim=latent_dim,
        output_dim=latent_dim,
        type='self',
        heads=heads,
        resnet=True,
        return_att_weights=True,
        name='pmhc_self_attention',
        epsilon=1e-6,
        gate=True,
        mask_token=mask_token,
        pad_token=pad_token,
        apply_rope=True,
    )(mhc_subtracted, mhc_mask_in)  # (B, M, L), (B, heads, M, M)

    pmhc_pool = GlobalMaxPooling1D(name="mhc_pool_after_subtract", axis=-2)(pmhc_interaction)  # (B, L)

    pmhc_pool = layers.Dense(
        pep_dim * max_pep_len,
        activation=None,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        bias_regularizer=keras.regularizers.l2(l2_reg),
        name="mhc_project_after_subtract"
    )(pmhc_pool)  # (B, P*D)

    # # (B, P*D) to (B, P, D) - reshape without learnable layers
    pmhc = layers.Reshape((max_pep_len, pep_dim), name="mhc_reshape")(pmhc_pool)  # (B, P, D)

    latent_sequence = pmhc

    # -------------------------------------------------------------------
    # TASK 1: BINDING PREDICTION HEAD # this part directly affects clustering
    # -------------------------------------------------------------------
    # pooled_latent = GlobalMaxPooling1D(name="latent_vector_pool-2", axis=-2)(latent_sequence) # (B, D) # Tells how strong the overall signal is across the sequence (using this forces the model to pu one feature as the distance/binder information)
    # pooled_std2 = GlobalSTDPooling1D(name="latent_vector_std-2", axis=-2)(
    #    latent_sequence)  # (B, D) # Tells how much variation there is in the signal across the sequence
    # pooled_mean1 = GlobalMeanPooling1D(name="latent_vector_pool-1", axis=-1)(
    #    latent_sequence)  # (B, P+M) # Tells how strong the overall signal is across the features
    # pooled_std1 = GlobalSTDPooling1D(name="latent_vector_std-1", axis=-1)(latent_sequence)  # (B, P+M) # Tells how much variation there is in the signal across the features

    # concatenate mean and std pooled vectors
    # pooled_latent = layers.Concatenate(name="pooled_latent_concat", axis=-1)(
    #    [pooled_mean1, pooled_std2])  # (B, 1*(D+P+M))

    pooled_latent = GlobalMaxPooling1D(name="latent_vector_pool", axis=-2)(latent_sequence)  # (B, D)
    binding_head = layers.Dropout(drop_out_rate * 1.5, name='dropout_pooled_latent')(pooled_latent)
    binding_head = layers.Dense(emb_dim // 2, activation="gelu", name="binding_dense1",
                                kernel_regularizer=keras.regularizers.l2(l2_reg),
                                bias_regularizer=keras.regularizers.l2(l2_reg))(binding_head)
    binding_head = layers.Dropout(drop_out_rate, name="binding_dropout2")(binding_head)
    binding_activation = None if return_logits else "sigmoid"
    binding_pred = layers.Dense(1, activation=binding_activation, name="binding_pred", dtype="float32")(
        binding_head)  # (B, 1)

    # -------------------------------------------------------------------
    # TASK 2: RECONSTRUCTION HEAD
    # -------------------------------------------------------------------
    pep_recon_head = layers.Dropout(drop_out_rate, name='pep_recon_dropout_1')(latent_sequence)
    pep_recon_head = layers.Dense(emb_dim * 2, activation='gelu')(pep_recon_head)
    pep_recon_head = layers.Dropout(drop_out_rate, name='pep_recon_dropout_2')(pep_recon_head)
    pep_recon_pred = layers.Dense(21, activation='softmax', name='pep_reconstruction_pred')(pep_recon_head)

    pep_out = layers.Concatenate(name='pep_ytrue_ypred')([pep_ohe_target_in, pep_recon_pred])

    # -------------------------------------------------------------------
    # MODEL DEFINITION
    # -------------------------------------------------------------------
    pMHC_multitask_model = keras.Model(
        inputs=[pep_in, pep_mask_in, mhc_emb_in, mhc_mask_in, pep_ohe_target_in],
        outputs={
            "pep_ytrue_ypred": pep_out,
            "cls_ypred": binding_pred,
            "attn_weights": pmhc_attn_weights,  # (B, heads, M, M)
            "latent_vector": pooled_latent,  # (B, D+P+M)
            "latent_seq": latent_sequence  # (B, P, D)
        },
        name="pMHC_Multitask_Transformer_Subtract_CrossAttention_Model_V12"
    )

    return pMHC_multitask_model


def pmbind_multitask_v17(max_pep_len: int,
                         max_mhc_len: int,
                         pep_dim: int = 14,
                         mhc_dim: int = 14,
                         emb_dim: int = 32,
                         heads: int = 8,
                         transformer_layers: int = 2,
                         mask_token: float = MASK_TOKEN,
                         pad_token: float = PAD_TOKEN,
                         noise_std: float = 0.1,
                         latent_dim: int = 32,
                         drop_out_rate: float = 0.18,
                         l2_reg: float = 0.02,
                         return_logits: bool = False):
    """
    A multi-task model for pMHC reconstruction and binding prediction.
    """
    # -------------------------------------------------------------------
    # INPUTS
    # -------------------------------------------------------------------
    pep_in = layers.Input((max_pep_len, pep_dim), name="pep_emb")  # (B, P, D)
    pep_mask_in = layers.Input((max_pep_len,), name="pep_mask")  # (B, P)
    pep_ohe_target_in = layers.Input((max_pep_len, 21),
                                     name="pep_ohe_target")  # (B, P, 21) one-hot encoded target for reconstruction

    mhc_emb_in = layers.Input((max_mhc_len, mhc_dim), name="mhc_emb")  # (B, M, mhc_dim)
    mhc_mask_in = layers.Input((max_mhc_len,), name="mhc_mask")  # (B, M)

    # -------------------------------------------------------------------
    # input processing
    # -------------------------------------------------------------------
    # 1. Embed Peptide and MHC to the same dimension
    pep = MaskedEmbedding(mask_token, pad_token, name="pep_mask_embed")(pep_in, pep_mask_in)  # (B, P, D)
    pep = PositionalEncoding(pep_dim, int(max_pep_len * 3), name="pep_pos_enc")(pep, pep_mask_in)  # (B, P, D)
    pep = layers.GaussianNoise(noise_std, name="pep_gaussian_noise")(pep)  # (B, P, D)
    pep = SpatialTemporalDropout1D(drop_out_rate, name="pep_dropout")(
        pep)  # SpatialDropout1D for sequence data to drop entire feature maps

    mhc = MaskedEmbedding(mask_token, pad_token, name="mhc_mask_embed")(mhc_emb_in, mhc_mask_in)  # (B, M, mhc_dim)
    mhc = layers.GaussianNoise(noise_std, name="mhc_gaussian_noise")(mhc)  # (B, M, mhc_dim)
    mhc = PositionalEncoding(mhc_dim, int(max_mhc_len * 3), name="mhc_pos_enc")(mhc, mhc_mask_in)  # (B, M, mhc_dim)
    mhc = SpatialTemporalDropout1D(drop_out_rate, name="mhc_dropout")(
        mhc)  # SpatialDropout1D for sequence data to drop entire feature maps

    # print shapes
    print(f"pep shape: {pep.shape}, mhc shape: {mhc.shape}")

    pmhc_concat = layers.Concatenate(axis=1, name="pmhc_concat")([pep, mhc])  # (B, P+M, D)

    # -------------------------------------------------------------------
    # TRANSFORMER CROSS-ATTENTION ENCODER
    # -------------------------------------------------------------------
    # pmhc self-attention with 2D mask
    pmhc_inter = SelfAttentionWith2DMaskRMSNorm(
        query_dim=pep_dim,
        context_dim=mhc_dim,
        output_dim=pep_dim,
        heads=heads,
        return_att_weights=False,
        self_attn_mhc=False,  # Prevent both peptide and MHC self-attention in this layer
        apply_rope=True,
        resnet=True,
        name="pmhc_2d_masked_attention"
    )(pmhc_concat, pep_mask_in, mhc_mask_in)

    pmhc_interaction, pmhc_attn_weights = SelfAttentionWith2DMaskRMSNorm(
        query_dim=pep_dim,
        context_dim=mhc_dim,
        output_dim=pep_dim,
        heads=heads,
        return_att_weights=True,
        self_attn_mhc=False,  # Prevent both peptide and MHC self-attention in this layer
        apply_rope=True,
        resnet=False,
        name="pmhc_2d_masked_attention_2"
    )(pmhc_inter, pep_mask_in, mhc_mask_in)  # (B, P+M, D)

    #
    latent_sequence = pmhc_interaction

    # -------------------------------------------------------------------
    # SHARED LATENT REPRESENTATION FOR BOTH HEADS
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

    # -------------------------------------------------------------------
    # TASK 1a: CORE SELECTION HEAD - g(c, S_i, MHC)
    # -------------------------------------------------------------------
    core_head = layers.Dropout(drop_out_rate * 1.5, name='dropout_core_head')(pooled_latent)
    core_head = layers.Dense(emb_dim // 2, activation="gelu", name="core_dense1",
                             kernel_regularizer=keras.regularizers.l2(l2_reg),
                             bias_regularizer=keras.regularizers.l2(l2_reg))(core_head)
    core_head = layers.Dropout(drop_out_rate, name="core_dropout2")(core_head)
    core_logits = layers.Dense(1, activation=None, name="core_logits", dtype="float32")(core_head)

    # -------------------------------------------------------------------
    # TASK 1b: BINDING PREDICTION HEAD - f(c, MHC)
    # -------------------------------------------------------------------
    binding_head = layers.Dropout(drop_out_rate * 1.5, name='dropout_binding_head')(pooled_latent)
    binding_head = layers.Dense(emb_dim // 2, activation="gelu", name="binding_dense1",
                                kernel_regularizer=keras.regularizers.l2(l2_reg),
                                bias_regularizer=keras.regularizers.l2(l2_reg))(binding_head)
    binding_head = layers.Dropout(drop_out_rate, name="binding_dropout2")(binding_head)
    binding_logits = layers.Dense(1, activation=None, name="binding_logits", dtype="float32")(binding_head)

    # Optional sigmoid for direct prediction (backwards compatibility)
    binding_pred = layers.Activation("sigmoid", name="binding_pred", dtype="float32")(
        binding_logits) if not return_logits else binding_logits

    # -------------------------------------------------------------------
    # TASK 3: RECONSTRUCTION HEAD
    # -------------------------------------------------------------------
    # Extract only the peptide portion from the latent sequence (first P positions)
    pep_latent = latent_sequence[:, :max_pep_len, :]  # (B, P, D)

    pep_recon_head = layers.Dropout(drop_out_rate, name='pep_recon_dropout_1')(pep_latent)
    pep_recon_head = layers.Dense(emb_dim * 2, activation='gelu')(pep_recon_head)
    pep_recon_head = layers.Dropout(drop_out_rate, name='pep_recon_dropout_2')(pep_recon_head)
    pep_recon_pred = layers.Dense(21, activation='softmax', name='pep_reconstruction_pred', dtype="float32")(
        pep_recon_head)

    pep_out = layers.Concatenate(name='pep_ytrue_ypred')([pep_ohe_target_in, pep_recon_pred])

    # -------------------------------------------------------------------
    # MODEL DEFINITION
    # -------------------------------------------------------------------
    pMHC_multitask_model = keras.Model(
        inputs=[pep_in, pep_mask_in, mhc_emb_in, mhc_mask_in, pep_ohe_target_in],
        outputs={
            "pep_ytrue_ypred": pep_out,
            "cls_ypred": binding_pred,
            "core_logits": core_logits,  # g(c, S_i, MHC) for core selection
            "binding_logits": binding_logits,  # f(c, MHC) for binding prediction
            "attn_weights": pmhc_attn_weights,
            "latent_vector": pooled_latent,
            "latent_seq": latent_sequence
        },
        name="pMHC_Multitask_Transformer_Subtract_Model_V17"
    )

    return pMHC_multitask_model