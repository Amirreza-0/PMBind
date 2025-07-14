#!/usr/bin/env python
"""
pepmhc_cross_attention.py
-------------------------

End-to-end peptide × MHC cross-attention reconstruction model with
explainable attention visualisation.

Author: Amirreza 2024-07-08
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Conv1D                    # avoid tf-keras / keras mix

# ---------- project-specific utilities -----------------------------
from src.utils import (
        seq_to_onehot, AttentionLayer, PositionalEncoding, MaskedEmbedding,
        determine_ks_dict, CustomDense,                       # layers
        generate_synthetic_pMHC_data,                         # data generator
        OHE_to_seq, MaskedCategoricalCELossLayer,
    # decoding
)

from src.visualizations import visualize_cross_attention_weights, plot_1d_heatmap

# -------------------------------------------------------------------
MASK_TOKEN = -1.0    # will be written into mask tensor for "masked" positions
PAD_TOKEN  = -2.0    # will be written into mask tensor for padding positions
# -------------------------------------------------------------------


# ===================================================================
#  masked categorical cross-entropy helper
# ===================================================================
def masked_cat_ce(y_true, y_pred, mask, pad_token=-2, mask_token=-1):
    """
    y_true … (B,L,21)  one-hot
    y_pred … (B,L,21)  soft-max
    mask    … (B,L)    1 = valid, 0 = PAD or MASK
    """
    loss_per_position = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1)
    mask = tf.where(mask == pad_token, 0., 1.)
    masked_loss = loss_per_position * mask
    total_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)

    return total_loss


# ===================================================================
#  functional graph (encoder-decoder)
# ===================================================================
def build_bicross_net(max_pep_len: int,
                      max_mhc_len: int,
                      mask_token: float = MASK_TOKEN,
                      pad_token: float = PAD_TOKEN,
                      pep_emb_dim: int = 128,
                      mhc_emb_dim: int = 128,
                      heads: int = 4,
                      cross_attention: bool = True) -> keras.Model:
    """
    Returns a tf.keras.Model that produces
        • peptide reconstruction (soft-max)
        • MHC reconstruction (soft-max)
        • attention maps & masked inputs   (for visualisation)
    """
    # ---------------------- inputs ----------------------------------
    pep_OHE_inp  = keras.Input((max_pep_len, 21),  name="pep_onehot")
    mask_pep_inp = keras.Input((max_pep_len,),     name="pep_mask")
    mhc_EMB_inp  = keras.Input((max_mhc_len,1152), name="mhc_latent")
    mask_mhc_inp = keras.Input((max_mhc_len,),     name="mhc_mask")
    mhc_OHE_inp  = keras.Input((max_mhc_len, 21),  name="mhc_onehot")

    # ---------------------- embedding + positional ------------------
    pep_OHE = MaskedEmbedding(mask_token, pad_token,
                              name="pep_masked_OHE")(pep_OHE_inp, mask_pep_inp)
    mhc_EMB = MaskedEmbedding(mask_token, pad_token,
                              name="mhc_masked_EMB")(mhc_EMB_inp, mask_mhc_inp)

    pep_OHE_m = PositionalEncoding(21,  int(max_pep_len*3),
                                   name="pep_pos_enc")(pep_OHE, mask_pep_inp)
    mhc_EMB_m = PositionalEncoding(1152, int(max_mhc_len*3),
                                   name="mhc_pos_enc")(mhc_EMB, mask_mhc_inp)

    pep_emb = CustomDense(pep_emb_dim, activation=None,
                          mask_token=mask_token, pad_token=pad_token,
                          name="pep_proj")(pep_OHE_m, mask_pep_inp)
    mhc_emb = CustomDense(mhc_emb_dim, activation='relu',
                          mask_token=mask_token, pad_token=pad_token,
                          name="mhc_proj")(mhc_EMB_m, mask_mhc_inp)

    # ---------------------- self-attention --------------------------
    pep_sa, _  = AttentionLayer(pep_emb_dim, pep_emb_dim, pep_emb_dim,
                                type="self", heads=heads, resnet=True,
                                return_att_weights=True,
                                name="pep_self_attn")(pep_emb, mask_pep_inp)

    mhc_sa, _  = AttentionLayer(mhc_emb_dim, mhc_emb_dim, mhc_emb_dim,
                                type="self", heads=heads, resnet=True,
                                return_att_weights=True,
                                name="mhc_self_attn")(mhc_emb, mask_mhc_inp)

    # ---------------------- cross-attention -------------------------
    if cross_attention:
        mhc_ca, mhc_ca_scores = AttentionLayer(
                                mhc_emb_dim, pep_emb_dim, mhc_emb_dim,
                                type="cross", heads=heads, resnet=False,
                                return_att_weights=True,
                                name="mhc_cross_attn")(
                                    mhc_sa, mask_mhc_inp,
                                    pep_sa, mask_pep_inp)

        # ---------------------- conv + extra self-attn on CA ------------
        ks_dict = determine_ks_dict(max_mhc_len,
                                    [max_mhc_len, max_mhc_len // 2,
                                     max_mhc_len // 2, max_pep_len],
                                    max_strides=30, max_kernel_size=80)

        pepconv = Conv1D(64, ks_dict["k1"], ks_dict["s1"], padding='valid',
                         activation='relu', name="pepconv1")(mhc_ca)
        pepconv = Conv1D(64, ks_dict["k2"], ks_dict["s2"], padding='valid',
                         activation='relu', name="pepconv2")(pepconv)
        pepconv = Conv1D(32, ks_dict["k3"], ks_dict["s3"], padding='valid',
                         activation='relu', name="pepconv3")(pepconv)
        pepconv = Conv1D(32, ks_dict["k4"], ks_dict["s4"], padding='valid',
                         activation='relu', name="pepconv4")(pepconv)

        pepconv, mhc_to_pep_scores = AttentionLayer(
            32, 32, 32,
            type="self", heads=heads, resnet=True,
            return_att_weights=True,
            name="mhc_to_pep_attn")(
            pepconv, mask_pep_inp)

        pep_lat = pepconv  # latent for peptide decoder
        mhc_lat = mhc_ca  # latent for MHC   decoder
    else:                       # fallback: simply concatenate # TODO
        mhc_ca = layers.Concatenate(axis=-2)([mhc_sa, pep_sa])
        mhc_ca_scores = tf.zeros((1,))          # dummy
        mhc_to_pep_scores = tf.zeros((1,))      # dummy
        pep_lat = pep_sa
        mhc_lat = mhc_ca

    # ---------------------- decoder heads ---------------------------
    pep_recon = CustomDense(64, activation=None,
                            mask_token=mask_token, pad_token=pad_token,
                            name="pep_dec_dense")(pep_lat, mask_pep_inp)
    pep_recon = layers.Dense(21, activation='softmax',
                            name="pep_recon")(pep_recon)

    mhc_recon = CustomDense(64, activation=None,
                            mask_token=mask_token, pad_token=pad_token,
                            name="mhc_dec_dense")(mhc_lat, mask_mhc_inp)
    mhc_recon = layers.Dense(21, activation='softmax',
                            name="mhc_recon")(mhc_recon)
    loss_pep = MaskedCategoricalCELossLayer(pad_token=pad_token, mask_token=mask_token, name="masked_ce_loss_pep")(
        [pep_OHE_inp, pep_recon, mask_pep_inp])
    loss_mhc = MaskedCategoricalCELossLayer(pad_token=pad_token, mask_token=mask_token, name="masked_ce_loss_mhc")(
        [mhc_OHE_inp, mhc_recon, mask_mhc_inp]
    )

    # ---------------------- assemble model --------------------------
    return keras.Model(
        inputs =[pep_OHE_inp, mask_pep_inp, mhc_EMB_inp, mask_mhc_inp, mhc_OHE_inp],
        outputs={'pep_recon'            : loss_pep,
                 'mhc_recon'            : loss_mhc,
                 'mhc_to_pep_attn_score': mhc_to_pep_scores,
                 'attention_scores_CA'  : mhc_ca_scores},
        name="bicross_encoder_decoder")


# ===================================================================
#  custom training wrapper
# ===================================================================
class BiCrossModel(keras.Model):
    """Wraps the functional graph and computes the masked CE losses."""

    def __init__(self, graph: keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.graph = graph

    # --------------------------------------------------------------
    def train_step(self, data):
        """
        data = ([pep_OHE, mask_pep, mhc_EMB, mask_mhc,
                 y_true_pep, y_true_mhc], _dummy_label)
        -> label is ignored, everything is in `x`
        """
        (pep_OHE, mask_pep,
         mhc_EMB, mask_mhc,
         y_true_pep, y_true_mhc) = data[0]

        with tf.GradientTape() as tape:
            outs   = self.graph([pep_OHE, mask_pep, mhc_EMB, mask_mhc],
                                training=True)
            pep_ce = masked_cat_ce(y_true_pep, outs['pep_recon'], mask_pep)
            mhc_ce = masked_cat_ce(y_true_mhc, outs['mhc_recon'], mask_mhc)
            loss   = pep_ce + mhc_ce

        grads = tape.gradient(loss, self.graph.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.graph.trainable_variables))

        return {"loss": loss, "pep_ce": pep_ce, "mhc_ce": mhc_ce}

    # --------------------------------------------------------------
    def call(self, inputs, training=False):
        return self.graph(inputs, training=training)


# ===================================================================
#  main (synthetic test)
# ===================================================================
if __name__ == "__main__":
    tf.random.set_seed(0)
    np.random.seed(0)

    # ---------------- parameters -----------------------------------
    max_pep_len = 14
    max_mhc_len = 60
    batch_size  = 20
    pep_dim     = 16
    mhc_dim     = 16
    heads       = 4
    epochs      = 100

    # ---------------- synthetic data -------------------------------
    (pep_OHE, mask_pep,
     mhc_EMB, mask_mhc,
     mhc_OHE, mhc_ids, ks_dict) = generate_synthetic_pMHC_data(
                                    batch_size, max_pep_len, max_mhc_len)

    y_true_pep = pep_OHE.copy()
    y_true_mhc = mhc_OHE.copy()

    # dataset for model.fit : give everything via `x`, `y=None`
    x_all = [pep_OHE, mask_pep, mhc_EMB, mask_mhc,
             y_true_pep, y_true_mhc]

    # ---------------- build & train --------------------------------
    model = build_bicross_net(max_pep_len, max_mhc_len,
                                 mask_token=MASK_TOKEN, pad_token=PAD_TOKEN,
                                 pep_emb_dim=pep_dim, mhc_emb_dim=mhc_dim,
                                 heads=heads)

#pep_OHE_inp, mask_pep_inp, mhc_EMB_inp, mask_mhc_inp
    #model = BiCrossModel(base_net)
    model.compile(optimizer=keras.optimizers.Adam(1e-3))
    model.fit(x=[pep_OHE, mask_pep, mhc_EMB, mask_mhc, mhc_OHE],
              epochs=epochs,
              batch_size=batch_size,
              verbose=2)

    # ---------------- inference ------------------------------------
    outs = model.predict([pep_OHE, mask_pep, mhc_EMB, mask_mhc, mhc_OHE])
    pep_recon = outs['pep_recon']
    mhc_recon = outs['mhc_recon']
    mhc_att   = outs['attention_scores_CA']
    mhc_to_pep_attn_score = outs['mhc_to_pep_attn_score']

    # print shapes
    print("pep_recon shape:", pep_recon.shape)
    print("mhc_recon shape:", mhc_recon.shape)
    print("mhc_att shape:", mhc_att.shape)
    print("mhc_to_pep_attn_score shape:", mhc_to_pep_attn_score.shape)


    # OHE → sequences
    pep_pred = OHE_to_seq(pep_recon)
    mhc_pred = OHE_to_seq(mhc_recon)
    pep_orig = OHE_to_seq(pep_OHE)
    mhc_orig = OHE_to_seq(mhc_OHE)
    pep_mask_vis = mask_pep
    mhc_mask_vis = mask_mhc


    print("\nPeptide reconstruction example")
    print("pred :", pep_pred[0])
    print("true :", pep_orig[0])

    print("\nMHC reconstruction example")
    print("pred :", mhc_pred[0][:60], "...")
    print("true :", mhc_orig[0][:60], "...")
    b = -1
    # ---------------- visualise attention --------------------------
    visualize_cross_attention_weights(mhc_att[b], peptide_seq=pep_orig[b], mhc_seq=mhc_orig[b], top_n=5)
    visualize_cross_attention_weights(mhc_to_pep_attn_score[b], peptide_seq=mhc_orig[b], mhc_seq=pep_orig[b], top_n=5)
    plot_1d_heatmap(pep_mask_vis[b])
    plot_1d_heatmap(mhc_mask_vis[b])
    # ---------------- save model -----------------------------------
    model.save("bicross_encoder_decoder.h5")