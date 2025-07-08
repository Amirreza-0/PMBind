#!/usr/bin/env python
"""
pepmhc_cross_attention.py
-------------------------

Minimal end-to-end demo of a peptide × MHC cross-attention classifier
with explainable attention visualisation.

Author: 2025-05-22
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {a: i for i, a in enumerate(AA)}
UNK = 20  # Index for "unknown"
MASK_TOKEN = -1.0
PAD_TOKEN = -2.0

# Helper function to one-hot encode peptide sequences
def onehot(seq: str, max_len: int) -> np.ndarray:
    """Return (max_len, 21) one-hot matrix."""
    mat = np.zeros((max_len, 21), dtype=np.float32)
    for i, aa in enumerate(seq[:max_len]):
        mat[i, AA_TO_INT.get(aa, UNK)] = 1.0
    return mat

# Custom Attention Layer
class AttentionLayer(keras.layers.Layer):
    """
    Custom multi-head attention layer supporting self- and cross-attention.

    Args:
        query_dim (int): Input feature dimension for query.
        context_dim (int): Input feature dimension for context (key and value).
        output_dim (int): Output feature dimension.
        type (str): 'self' or 'cross'.
        heads (int): Number of attention heads.
        resnet (bool): Whether to use residual connection.
        return_att_weights (bool): Whether to return attention weights.
        name (str): Layer name.
        epsilon (float): Epsilon for layer normalization.
        gate (bool): Whether to use gating mechanism.
        mask_token (float): Value for masked tokens.
        pad_token (float): Value for padded tokens.
    """
    def __init__(self, query_dim, context_dim, output_dim, type, heads=4,
                 resnet=True, return_att_weights=False, name='attention',
                 epsilon=1e-6, gate=True, mask_token=-1., pad_token=-2.):
        super().__init__(name=name)
        assert isinstance(query_dim, int) and isinstance(context_dim, int) and isinstance(output_dim, int)
        assert type in ['self', 'cross']
        if resnet:
            assert query_dim == output_dim
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.type = type
        self.heads = heads
        self.resnet = resnet
        self.return_att_weights = return_att_weights
        self.epsilon = epsilon
        self.gate = gate
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.att_dim = output_dim // heads  # Attention dimension per head

    def build(self, x):
        # Projection weights
        self.q_proj = self.add_weight(shape=(self.heads, self.query_dim, self.att_dim),
                                      initializer='random_normal', trainable=True, name=f'q_proj_{self.name}')
        self.k_proj = self.add_weight(shape=(self.heads, self.context_dim, self.att_dim),
                                      initializer='random_normal', trainable=True, name=f'k_proj_{self.name}')
        self.v_proj = self.add_weight(shape=(self.heads, self.context_dim, self.att_dim),
                                      initializer='random_normal', trainable=True, name=f'v_proj_{self.name}')
        if self.gate:
            self.g = self.add_weight(shape=(self.heads, self.query_dim, self.att_dim),
                                     initializer='random_uniform', trainable=True, name=f'gate_{self.name}')
        self.norm = layers.LayerNormalization(epsilon=self.epsilon, name=f'ln_{self.name}')
        if self.type == 'cross':
            self.norm_context = layers.LayerNormalization(epsilon=self.epsilon, name=f'ln_context_{self.name}')
        self.norm_out = layers.LayerNormalization(epsilon=self.epsilon, name=f'ln_out_{self.name}')
        if self.resnet:
            self.norm_resnet = layers.LayerNormalization(epsilon=self.epsilon, name=f'ln_resnet_{self.name}')
        self.out_w = self.add_weight(shape=(self.heads * self.att_dim, self.output_dim),
                                     initializer='random_normal', trainable=True, name=f'outw_{self.name}')
        self.out_b = self.add_weight(shape=(self.output_dim,), initializer='zeros',
                                     trainable=True, name=f'outb_{self.name}')
        self.scale = 1.0 / tf.math.sqrt(tf.cast(self.att_dim, tf.float32))

    def call(self, x, mask, context=None, context_mask=None):
        """
        Args:
            x: Tensor of shape (B, N, query_dim) for query.
            mask: Tensor of shape (B, N).
            context: Tensor of shape (B, M, context_dim) for key/value in cross-attention.
            context_mask: Tensor of shape (B, M) for context.
        """
        mask = tf.cast(mask, tf.float32)
        if self.type == 'self':
            context = x
            context_mask = mask
            q_input = k_input = v_input = self.norm(x)
            mask_q = mask_k = tf.where(mask == self.pad_token, 0., 1.)
        else:
            assert context is not None and context_mask is not None
            q_input = self.norm(x)
            k_input = v_input = self.norm_context(context)
            mask_q = tf.where(mask == self.pad_token, 0., 1.)
            mask_k = tf.where(context_mask == self.pad_token, 0., 1.)

        # Project query, key, value
        q = tf.einsum('bnd,hde->bhne', q_input, self.q_proj)
        k = tf.einsum('bmd,hde->bhme', k_input, self.k_proj)
        v = tf.einsum('bmd,hde->bhme', v_input, self.v_proj)

        # Compute attention scores
        att = tf.einsum('bhne,bhme->bhnm', q, k) * self.scale
        mask_q_exp = tf.expand_dims(mask_q, axis=1)
        mask_k_exp = tf.expand_dims(mask_k, axis=1)
        attention_mask = tf.einsum('bqn,bkm->bqnm', mask_q_exp, mask_k_exp)
        attention_mask = tf.broadcast_to(attention_mask, tf.shape(att))
        att += (1.0 - attention_mask) * -1e9
        att = tf.nn.softmax(att, axis=-1) * attention_mask

        # Compute output
        out = tf.einsum('bhnm,bhme->bhne', att, v)
        if self.gate:
            g = tf.einsum('bnd,hde->bhne', q_input, self.g)
            g = tf.nn.sigmoid(g)
            out *= g

        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [tf.shape(x)[0], tf.shape(x)[1], self.heads * self.att_dim])
        out = tf.matmul(out, self.out_w) + self.out_b

        if self.resnet:
            out += x
            out = self.norm_resnet(out)
        out = self.norm_out(out)
        mask_exp = tf.expand_dims(mask_q, axis=-1)
        out *= mask_exp

        return (out, att) if self.return_att_weights else out

# Model builder function
def bicross_recon(max_pep_len: int,
                  max_mhc_len: int,
                  mask_token: float = MASK_TOKEN,
                  pad_token: float = PAD_TOKEN,
                  pep_emb_dim: int = 64,
                  mhc_emb_dim: int = 64,
                  heads: int = 8):
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

    # Self-attention
    pep_self_attn = AttentionLayer(query_dim=21, context_dim=21, output_dim=pep_emb_dim,
                                   type="self", heads=heads, resnet=False,
                                   return_att_weights=False, name="pep_self_attn",
                                   mask_token=mask_token, pad_token=pad_token)(pep_OHE, mask_pep)
    mhc_self_attn = AttentionLayer(query_dim=1152, context_dim=1152, output_dim=mhc_emb_dim,
                                   type="self", heads=heads, resnet=False,
                                   return_att_weights=False, name="mhc_self_attn",
                                   mask_token=mask_token, pad_token=pad_token)(mhc_EMB, mask_mhc)

    # Cross-attention
    pep_cross_attn = AttentionLayer(query_dim=pep_emb_dim, context_dim=mhc_emb_dim, output_dim=pep_emb_dim,
                                    type="cross", heads=heads, resnet=True,
                                    return_att_weights=False, name="pep_cross_attn",
                                    mask_token=mask_token, pad_token=pad_token)(pep_self_attn, mask_pep, mhc_self_attn, mask_mhc)
    mhc_cross_attn = AttentionLayer(query_dim=mhc_emb_dim, context_dim=pep_emb_dim, output_dim=mhc_emb_dim,
                                    type="cross", heads=heads, resnet=True,
                                    return_att_weights=False, name="mhc_cross_attn",
                                    mask_token=mask_token, pad_token=pad_token)(mhc_self_attn, mask_mhc, pep_self_attn, mask_pep)

    # Reconstruction heads
    pep_recon_dense = layers.Dense(128, activation='relu', name='pep_recon_dense')(pep_cross_attn)
    pep_recon_out = layers.Dense(21, activation='softmax', name='pep_reconstruction')(pep_recon_dense)
    mhc_recon_dense = layers.Dense(512, activation='relu', name='mhc_recon_dense')(mhc_cross_attn)
    mhc_recon_out = layers.Dense(1152, activation='linear', name='mhc_reconstruction')(mhc_recon_dense)

    # Latent representation
    pep_latent = layers.GlobalAveragePooling1D(name='pep_latent_pool')(pep_cross_attn)
    mhc_latent = layers.GlobalAveragePooling1D(name='mhc_latent_pool')(mhc_cross_attn)
    cross_latent = layers.Concatenate(name='cross_latent')([pep_latent, mhc_latent])

    # Define models
    recon_model = keras.Model(inputs=[pep_OHE, mask_pep, mhc_EMB, mask_mhc],
                              outputs=[pep_recon_out, mhc_recon_out],
                              name='reconstruction_model')
    pep_cross_attn_att = AttentionLayer(query_dim=pep_emb_dim, context_dim=mhc_emb_dim, output_dim=pep_emb_dim,
                                        type="cross", heads=heads, resnet=True,
                                        return_att_weights=True, name="pep_cross_attn_att",
                                        mask_token=mask_token, pad_token=pad_token)(pep_self_attn, mask_pep, mhc_self_attn, mask_mhc)
    mhc_cross_attn_att = AttentionLayer(query_dim=mhc_emb_dim, context_dim=pep_emb_dim, output_dim=mhc_emb_dim,
                                        type="cross", heads=heads, resnet=True,
                                        return_att_weights=True, name="mhc_cross_attn_att",
                                        mask_token=mask_token, pad_token=pad_token)(mhc_self_attn, mask_mhc, pep_self_attn, mask_pep)
    att_model = keras.Model(inputs=[pep_OHE, mask_pep, mhc_EMB, mask_mhc],
                            outputs=[pep_cross_attn_att[1], mhc_cross_attn_att[1]],
                            name='attention_model')
    cross_latent_model = keras.Model(inputs=[pep_OHE, mask_pep, mhc_EMB, mask_mhc],
                                     outputs=cross_latent,
                                     name='cross_latent_model')

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