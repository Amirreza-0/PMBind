import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.utils import resample
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List, Hashable
from Bio.PDB import PDBParser
from scipy.spatial.distance import pdist, squareform
from scipy.special import softmax
import os, tempfile
from io import StringIO
from typing import Literal, Union, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from Bio import AlignIO
from Bio.Align.Applications import MafftCommandline


# Constants
AA = "ACDEFGHIKLMNPQRSTVWY-"
AA_TO_INT = {a: i for i, a in enumerate(AA)}
UNK_IDX = 20  # Index for "unknown"
MASK_TOKEN = -1.0
NORM_TOKEN = 1.0
PAD_TOKEN = -2.0
PAD_VALUE = 0.0
MASK_VALUE = 0.0


# Helper function to one-hot encode peptide sequences
# def seq_to_onehot(seq: str, max_len: int) -> np.ndarray:
#     """Return (max_len, 21) one-hot matrix."""
#     mat = np.zeros((max_len, 21), dtype=np.float32)
#     for i, aa in enumerate(seq[:max_len]):
#         mat[i, AA_TO_INT.get(aa, UNK)] = 1.0
#     return mat

def seq_to_onehot(sequence: str, max_seq_len: int) -> np.ndarray:
    """Convert peptide sequence to one-hot encoding"""
    arr = np.full((max_seq_len, 21), PAD_VALUE, dtype=np.float32) # initialize padding with 0
    for j, aa in enumerate(sequence.upper()[:max_seq_len]):
        arr[j, AA_TO_INT.get(aa, UNK_IDX)] = 1.0
        # print number of UNKs in the sequence
    # num_unks = np.sum(arr[:, UNK_IDX])
    # zero out gaps
    arr[:, AA_TO_INT['-']] = PAD_VALUE  # Set gaps to PAD_VALUE
    # if num_unks > 0:
    #     print(f"Warning: {num_unks} unknown amino acids in sequence '{sequence}'")
    return arr


def OHE_to_seq(ohe: np.ndarray, gap: bool = False) -> list:
    """
    Convert a one-hot encoded matrix back to a peptide sequence.
    # (B, max_pep_len, 21) -> (B, max_pep_len)
    Args:
        ohe: One-hot encoded matrix of shape (B, N, 21).
    Returns:
        sequence: Peptide sequence as a string. (B,)
    """
    sequence = []
    for i in range(ohe.shape[0]):  # Iterate over batch dimension
        seq = []
        for j in range(ohe.shape[1]):  # Iterate over sequence length
            if gap and np.all(ohe[i, j] == 0):
                seq.append('-')
            else:
                aa_index = np.argmax(ohe[i, j])  # Get index of the max value in one-hot encoding
                if aa_index < len(AA):  # Check if it's a valid amino acid index
                    seq.append(AA[aa_index])
                else:
                    seq.append('X')  # Use 'X' for unknown amino acids
        sequence.append(''.join(seq))  # Join the list into a string
    return sequence  # Return list of sequences


def OHE_to_seq_single(ohe: np.ndarray, gap=False) -> str:
    """
    Convert a one-hot encoded matrix back to a peptide sequence.
    Args:
        ohe: One-hot encoded matrix of shape (N, 21).
    Returns:
        sequence: Peptide sequence as a string.
    """
    seq = []
    for j in range(ohe.shape[0]):  # Iterate over sequence length
        if gap and np.all(ohe[j] == 0):
            seq.append('-')
        else:
            aa_index = np.argmax(ohe[j])  # Get index of the max value in one-hot encoding
            seq.append(AA[aa_index])
    return ''.join(seq)  # Join the list into a string


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


class MaskedEmbedding(keras.layers.Layer):
    def __init__(self, mask_token=-1., pad_token=-2., name='masked_embedding'):
        super().__init__(name=name)
        self.mask_token = mask_token
        self.pad_token = pad_token

    def call(self, x, mask):
        """
        Args:
            x: Input tensor of shape (B, N, D)
            mask: Tensor of shape (B, N)
        Returns:
            Tensor with masked positions set to zero.
        """
        mask = tf.cast(mask, tf.float32)
        mask = tf.where((mask == self.pad_token) | (mask == self.mask_token), 0., 1.)
        return x * mask[:, :, tf.newaxis]  # Apply mask to zero out positions



class PositionalEncoding(keras.layers.Layer):
    """
    Sinusoidal Positional Encoding layer that applies encodings
    only to non-masked tokens.

    Args:
        embed_dim (int): Dimension of embeddings (must match input last dim).
        max_len (int): Maximum sequence length expected (used to precompute encodings).
    """

    def __init__(self, embed_dim, pos_range=100, mask_token=-1., pad_token=-2., name='positional_encoding'):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.pos_range = pos_range
        self.mask_token = mask_token
        self.pad_token = pad_token

    def build(self, x):
        # Create (1, pos_range, embed_dim) encoding matrix
        pos = tf.range(self.pos_range, dtype=tf.float32)[:, tf.newaxis]  # (pos_range, 1)
        i = tf.range(self.embed_dim, dtype=tf.float32)[tf.newaxis, :]  # (1, embed_dim)
        #angle_rates = 1 / tf.pow(300.0, (2 * (i // 2)) / tf.cast(self.embed_dim, tf.float32))
        angle_rates = tf.pow(300.0, -(2 * tf.floor(i / 2)) / tf.cast(self.embed_dim, tf.float32))
        angle_rads = pos * angle_rates  # (pos_range, embed_dim)

        # Apply sin to even indices, cos to odd indices
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)  # (max_len, embed_dim)
        pos_encoding = pos_encoding[tf.newaxis, ...]  # (1, max_len, embed_dim)
        self.pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x, mask):
        """
        Args:
            x: Input tensor of shape (B, N, D)
            mask: Tensor of shape (B,N)
        Returns:
            Tensor with positional encodings added for masked and non padded tokens.
        """
        seq_len = tf.shape(x)[1]
        pe = self.pos_encoding[:, :seq_len, :]  # (1, N, D)
        mask = tf.cast(mask[:, :, tf.newaxis], tf.float32)  # (B, N, 1)
        mask = tf.where(mask == self.pad_token, 0., 1.)
        pe = pe * mask  # zero out positions where mask is 0

        return x + pe


@tf.function
def select_indices(ind, n, m_range):
    """
    Select top-n indices from `ind` (descending sorted) such that:
    - First index is always selected.
    - Each subsequent index has a distance from all previously selected
      indices between m_range[0] and m_range[1], inclusive.
    Args:
        ind: Tensor of shape (B, N) with descending sorted indices.
        n: Number of indices to select.
        m_range: List or tuple [min_distance, max_distance]
    Returns:
        Tensor of shape (B, n) with selected indices per batch.
    """
    m_min = tf.constant(m_range[0], dtype=tf.int32)
    m_max = tf.constant(m_range[1], dtype=tf.int32)

    def per_batch_select(indices):
        top = indices[0]
        selected = tf.TensorArray(dtype=tf.int32, size=n)
        selected = selected.write(0, top)
        count = tf.constant(1)
        i = tf.constant(1)

        def cond(i, count, selected):
            return tf.logical_and(i < tf.shape(indices)[0], count < n)

        def body(i, count, selected):
            candidate = indices[i]
            selected_vals = selected.stack()[:count]
            distances = tf.abs(selected_vals - candidate)
            if_valid = tf.reduce_all(
                tf.logical_and(distances >= m_min, distances <= m_max)
            )
            selected = tf.cond(if_valid,
                               lambda: selected.write(count, candidate),
                               lambda: selected)
            count = tf.cond(if_valid, lambda: count + 1, lambda: count)
            return i + 1, count, selected

        _, _, selected = tf.while_loop(
            cond, body, [i, count, selected],
            shape_invariants=[i.get_shape(), count.get_shape(), tf.TensorShape(None)]
        )
        return selected.stack()

    return tf.map_fn(per_batch_select, ind, dtype=tf.int32)


class AnchorPositionExtractor(keras.layers.Layer):
    def __init__(self, num_anchors, dist_thr, name='anchor_extractor', project=True,
                 mask_token=-1., pad_token=-2., return_att_weights=False):
        super().__init__()
        assert isinstance(dist_thr, list) and len(dist_thr) == 2
        assert num_anchors > 0
        self.num_anchors = num_anchors
        self.dist_thr = dist_thr
        self.name = name
        self.project = project
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.return_att_weights = return_att_weights

    def build(self, input_shape):  # att_out (B,N,E)
        b, n, e = input_shape[0], input_shape[1], input_shape[2]
        self.barcode = tf.random.uniform(shape=(1, 1, e))  # add as a token to input
        self.q = self.add_weight(shape=(e, e),
                                 initializer='random_normal',
                                 trainable=True, name=f'query_{self.name}')
        self.k = self.add_weight(shape=(e, e),
                                 initializer='random_normal',
                                 trainable=True, name=f'key_{self.name}')
        self.v = self.add_weight(shape=(e, e),
                                 initializer='random_normal',
                                 trainable=True, name=f'value_{self.name}')
        self.ln = layers.LayerNormalization(name=f'ln_{self.name}')
        if self.project:
            self.g = self.add_weight(shape=(self.num_anchors, e, e),
                                     initializer='random_uniform',
                                     trainable=True, name=f'gate_{self.name}')
            self.w = self.add_weight(shape=(1, self.num_anchors, e, e),
                                     initializer='random_normal',
                                     trainable=True, name=f'w_{self.name}')

    def call(self, input, mask):  # (B,N,E) this is peptide embedding and (B,N) for mask

        mask = tf.cast(mask, tf.float32)  # (B, N)
        mask = tf.where(mask == self.pad_token, 0., 1.)

        barcode = self.barcode
        barcode = tf.broadcast_to(barcode, (tf.shape(input)[0], 1, tf.shape(input)[-1]))  # (B,N,E)
        q = tf.matmul(barcode, self.q)  # (B,1,E)*(E,E)->(B,1,E)
        k = tf.matmul(input, self.k)  # (B,N,E)*(E,E)->(B,N,E)
        v = tf.matmul(input, self.v)  # (B,N,E)*(E,E)->(B,N,E)
        scale = 1 / tf.math.sqrt(tf.cast(tf.shape(input)[-1], tf.float32))
        barcode_att = tf.matmul(q, k, transpose_b=True) * scale  # (B,1,E)*(B,E,N)->(B,1,N)
        # mask: (B,N) => (B,1,N)
        mask_exp = tf.expand_dims(mask, axis=1)
        additive_mask = (1.0 - mask_exp) * -1e9
        barcode_att += additive_mask
        barcode_att = tf.nn.softmax(barcode_att)
        barcode_att *= mask_exp  # to remove the impact of row wise attention of padded tokens. since all are 1e-9
        barcode_out = tf.matmul(barcode_att, v)  # (B,1,N)*(B,N,E)->(B,1,E)
        # barcode_out represents a vector for all information from peptide
        # barcode_att represents the anchor positions which are the tokens with highest weights
        inds, weights, outs = self.find_anchor(input,
                                               barcode_att)  # (B,num_anchors) (B,num_anchors) (B, num_anchors, E)
        if self.project:
            pos_encoding = tf.broadcast_to(
                tf.expand_dims(inds, axis=-1),
                (tf.shape(outs)[0], tf.shape(outs)[1], tf.shape(outs)[2])
            )
            pos_encoding = tf.cast(pos_encoding, tf.float32)
            dim = tf.cast(tf.shape(outs)[-1], tf.float32)
            ra = tf.range(dim, dtype=tf.float32) / dim
            pos_encoding = tf.sin(pos_encoding / tf.pow(40., ra))
            outs += pos_encoding

            weights_bc = tf.expand_dims(weights, axis=-1)
            weights_bc = tf.broadcast_to(weights_bc, (tf.shape(weights_bc)[0],
                                                      tf.shape(weights_bc)[1],
                                                      tf.shape(outs)[-1]
                                                      ))  # (B,num_anchors, E)
            outs = tf.expand_dims(outs, axis=-2)  # (B, num_anchors, 1, E)
            outs_w = tf.matmul(outs, self.w)  # (B,num_anchors,1,E)*(1,num_anchors,E,E)->(B,num_anchors,1,E)
            outs_g = tf.nn.sigmoid(tf.matmul(outs, self.g))
            outs_w = tf.squeeze(outs_w, axis=-2)  # (B,num_anchors,E)
            outs_g = tf.squeeze(outs_g, axis=-2)
            # multiply by attention weights from barcode_att to choose best anchors and additional feature gating
            outs = outs_w * outs_g * weights_bc  # (B, num_anchors, E)
        outs = self.ln(outs)
        # outs -> anchor info, inds -> anchor indeces, weights -> anchor att weights, barcode_out -> whole peptide features
        # (B,num_anchors,E), (B,num_anchors), (B,num_anchors), (B,E)
        if self.return_att_weights:
            return outs, inds, weights, tf.squeeze(barcode_out, axis=1), barcode_att
        else:
            return outs, inds, weights, tf.squeeze(barcode_out, axis=1)

    def find_anchor(self, input, barcode_att):  # (B,N,E), (B,1,N)
        inds = tf.argsort(barcode_att, axis=-1, direction='DESCENDING', stable=False)  # (B,1,N)
        inds = tf.squeeze(inds, axis=1)  # (B,N)
        selected_inds = select_indices(inds, n=self.num_anchors, m_range=self.dist_thr)  # (B,num_anchors)
        sorted_selected_inds = tf.sort(selected_inds)
        sorted_selected_weights = tf.gather(tf.squeeze(barcode_att, axis=1),
                                            sorted_selected_inds,
                                            axis=1,
                                            batch_dims=1)  # (B,num_anchors)
        sorted_selected_output = tf.gather(input, sorted_selected_inds, axis=1, batch_dims=1)  # (B,num_anchors,E)
        return sorted_selected_inds, sorted_selected_weights, sorted_selected_output


class ConcatMask(keras.layers.Layer):
    def __init__(self, name='concat_mask'):
        super().__init__(name=name)

    def call(self, mask1, mask2):
        """
        Args:
            mask1: Tensor of shape (B, N1)
            mask2: Tensor of shape (B, N2)
        Returns:
            Concatenated mask tensor of shape (B, N1 + N2)
        """
        mask1 = tf.cast(mask1, tf.float32)
        mask2 = tf.cast(mask2, tf.float32)
        return tf.concat([mask1, mask2], axis=1)


class ConcatBarcode(keras.layers.Layer):
    def __init__(self, name='barcode_layer'):
        super().__init__(name=name)

    def call(self, x, barcode):
        """ Args:
            x: Input tensor of shape (B, N, D)
            barcode: Input tenshor of shape (B,N2,1)
        Returns:
            x: Tensor with barcode concatenated at the beginning.
            mask: Mask tensor of shape (B, N + barcode_length)
        """
        tf.debugging.assert_rank(x, 3, message="Input tensor x must be 3D (B, N, D)")
        barcode = tf.broadcast_to(barcode, shape=(tf.shape(x)[0], tf.shape(barcode)[1], tf.shape(x)[-1]))
        # concat barcode to the input
        x = tf.concat([barcode, x], axis=1) #(B,N2+N,D)
        return x



class SplitLayer(keras.layers.Layer):
    def __init__(self, split_size, name='split_layer'):
        """
        :param split_size: a float of ints that sum up your input dimension
        :param name: str
        """
        super().__init__(name=name)
        self.split_size = split_size

    def call(self, x):
        """
        Args:
            x: Input tensor of shape (B, N, D)
        Returns:
            List of tensors split along the second dimension.
        """
        return tf.split(x, num_or_size_splits=self.split_size, axis=1)  # Split along the second dimension


def determine_conv_params(input_dim, output_dim, max_kernel_size=5, max_strides=2):
    """
    Determine kernel size and strides for a single Conv1D layer.

    Args:
        input_dim (int): Input sequence length.
        output_dim (int): Desired output sequence length.
        max_kernel_size (int): Maximum allowed kernel size.
        max_strides (int): Maximum allowed strides.

    Returns:
        tuple: (kernel_size, strides) if found, else None.
    """
    candidates = []
    for strides in range(1, max_strides + 1):
        for kernel_size in range(1, max_kernel_size + 1):
            if (input_dim - kernel_size) // strides + 1 == output_dim:
                candidates.append((kernel_size, strides))
    if candidates:
        candidates.sort(key=lambda x: (x[1], x[0]))  # Prefer smaller strides, then kernel size
        return candidates[0]
    return None

def determine_ks_dict(initial_input_dim, output_dims, max_kernel_size=50, max_strides=20):
    """
    Determine kernel sizes and strides for four sequential Conv1D layers.

    Args:
        initial_input_dim (int): Initial input sequence length.
        output_dims (list of int): List of four output sequence lengths after each layer.
        max_kernel_size (int): Maximum allowed kernel size.
        max_strides (int): Maximum allowed strides.

    Returns:
        dict: Dictionary with keys "k1", "s1", ..., "k4", "s4", or None if no valid parameters.
    """
    if len(output_dims) != 4:
        raise ValueError("output_dims must contain exactly four integers.")

    ks_dict = {}
    current_dim = initial_input_dim

    for i, output_dim in enumerate(output_dims, start=1):
        result = determine_conv_params(current_dim, output_dim, max_kernel_size, max_strides)
        if result is not None:
            kernel_size, strides = result
            ks_dict[f"k{i}"] = kernel_size
            ks_dict[f"s{i}"] = strides
            current_dim = output_dim  # Update input for next layer
        else:
            print(f"No valid parameters found for layer {i}: {current_dim} → {output_dim}")
            return None

    return ks_dict

# # Example usage
# if __name__ == "__main__":
#     initial_input = 180
#     output_dims = [79, 43, 19, 11]
#     result = determine_ks_dict(initial_input, output_dims)
#     print(result)  # Expected: {"k1": 3, "s1": 2, "k2": 3, "s2": 1, "k3": 3, "s3": 1, "k4": 2, "s4": 1}

def masked_categorical_crossentropy(y_true, y_pred, mask):
    """
    Compute masked categorical cross-entropy loss.

    Args:
        y_true: True labels (tensor).
        y_pred: Predicted probabilities (tensor).
        mask: Mask tensor indicating positions to include in the loss.

    Returns:
        Mean masked loss (tensor).
    """
    MASK_TOKEN = -1.0  # Replace with your actual mask token value
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    mask_binary = tf.cast(mask == MASK_TOKEN, tf.float32)
    masked_loss = loss * mask_binary
    sum_masked_loss = tf.reduce_sum(masked_loss)
    num_masked = tf.reduce_sum(mask_binary)
    return tf.math.divide_no_nan(sum_masked_loss, num_masked)


class CustomDense(keras.layers.Layer):
    """
    Custom dense layer that applies a linear transformation to the input.
    """
    def __init__(self, units, activation=None, name='custom_dense', mask_token=-1.0, pad_token=-2.0):
        super().__init__(name=name)
        self.units = units
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True, name=f'w_{self.name}')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True, name=f'b_{self.name}')

    def call(self, inputs, mask = None): # inputs: (B, N, D) mask: (B, N)
        output = tf.matmul(inputs, self.w) + self.b
        if self.activation is not None:
            output = self.activation(output)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            mask = tf.where(mask == self.pad_token, 0.0, 1.0)
            mask = tf.expand_dims(mask, axis=-1)  # (B, N, 1)
            output *= mask
        return output

# TODO
class SelfAttentionWith2DMask(keras.layers.Layer):
    """
    Custom self-attention layer that supports 2D masks.
    """
    def __init__(self, query_dim, context_dim, output_dim, heads=4,
                 return_att_weights=False, name='SelfAttentionWith2DMask',
                 epsilon=1e-6, mask_token=-1., pad_token=-2.):
        super().__init__(name=name)
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.heads = heads
        self.return_att_weights = return_att_weights
        self.epsilon = epsilon
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.att_dim = output_dim // heads  # Attention dimension per head

    def build(self, x):
        # Projection weights
        self.norm1 = layers.LayerNormalization(epsilon=self.epsilon, name=f'ln1_{self.name}')
        self.q_proj = self.add_weight(shape=(self.heads, self.query_dim, self.att_dim),
                                      initializer='random_normal', trainable=True, name=f'q_proj_{self.name}')
        self.k_proj = self.add_weight(shape=(self.heads, self.context_dim, self.att_dim),
                                      initializer='random_normal', trainable=True, name=f'k_proj_{self.name}')
        self.v_proj = self.add_weight(shape=(self.heads, self.context_dim, self.att_dim),
                                      initializer='random_normal', trainable=True, name=f'v_proj_{self.name}')
        self.g_proj = self.add_weight(shape=(self.heads, self.query_dim, self.att_dim),
                                      initializer='random_uniform', trainable=True, name=f'g_proj_{self.name}')
        self.norm2 = layers.LayerNormalization(epsilon=self.epsilon, name=f'ln2_{self.name}')


        self.out_w = self.add_weight(shape=(self.heads * self.att_dim, self.output_dim),
                                     initializer='random_normal', trainable=True, name=f'outw_{self.name}')
        self.out_b = self.add_weight(shape=(self.output_dim,), initializer='zeros',
                                     trainable=True, name=f'outb_{self.name}')
        self.scale = 1.0 / tf.math.sqrt(tf.cast(self.att_dim, tf.float32))

    def call(self, x_pmhc, p_mask, m_mask):
        """
        Args:
            x: Tensor of shape (B, N+M, query_dim) for query.
            mask: Tensor of shape (B, N, M) for 2D mask.
        :param x:
        :param mask:
        :return:
        """
        x_pmhc = self.norm1(x_pmhc)  # Normalize input
        p_mask = tf.cast(p_mask, tf.float32)
        m_mask = tf.cast(m_mask, tf.float32)
        p_mask = tf.where(p_mask==self.pad_token, x=0., y=1.)  # (B, N)
        m_mask = tf.where(m_mask==self.pad_token, x=0., y=1.)  # (B, M)

        q = tf.einsum('bxd,hde->bhxe', x_pmhc , self.q_proj)  # (B, N+M, E) * (H, E, D) -> (B, H, N+M, D)
        k = tf.einsum('bxd,hde->bhxe', x_pmhc, self.k_proj) # (B, N+M, E) * (H, E, D) -> (B, H, N+M, D)
        v = tf.einsum('bxd,hde->bhxe', x_pmhc, self.v_proj) # (B, N+M, E) * (H, E, D) -> (B, H, N+M, D)
        g = tf.einsum('bxd,hde->bhxe', x_pmhc, self.g_proj)  # (B, N+M, E) * (H, E, D) -> (B, H, N+M, D)

        att = tf.matmul(q, k, transpose_b=True) * self.scale  # (B, H, N+M, D) * (B, H, D, N+M) -> (B, H, N+M, N+M)
        # Create 2D mask
        mask_2d = self.mask_2d(p_mask, m_mask)
        mask_2d = tf.cast(mask_2d, tf.float32)  # (B, N+M, N+M)
        mask_2d_neg = (1.0 - mask_2d) * -1e9  # Apply mask to attention scores
        att = tf.nn.softmax(att + tf.expand_dims(mask_2d_neg, axis=1), axis=-1)  # Apply softmax to attention scores
        att *= tf.expand_dims(mask_2d, axis=1) # remove the impact of row wise attention of padded tokens. since all are 1e-9
        out = tf.matmul(att, v)  # (B, H, N+M, N+M) * (B, H, N+M, D) -> (B, H, N+M, D)
        out *= tf.nn.sigmoid(g)  # Apply gating mechanism
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [tf.shape(x_pmhc)[0], tf.shape(x_pmhc)[1], self.heads * self.att_dim])
        out = tf.matmul(out, self.out_w) + self.out_b
        out = self.norm2(out)
        if self.return_att_weights:
            return out, att
        else:
            return out

    def mask_2d(self, p_mask, m_mask):
        p_mask = tf.cast(p_mask, tf.float32)
        m_mask = tf.cast(m_mask, tf.float32)
        p_mask = tf.expand_dims(p_mask, axis=-1)
        m_mask = tf.expand_dims(m_mask, axis=-1) # (B, N, 1), (B, M, 1)
        # zero square masks
        self_peptide_mask = tf.zeros_like(p_mask, dtype=tf.float32) # (B, N, 1)
        self_peptide_mask_2d = tf.broadcast_to(self_peptide_mask, (
            tf.shape(p_mask)[0], tf.shape(p_mask)[1], tf.shape(p_mask)[1])) #(B, N, N)
        self_mhc_mask = tf.zeros_like(m_mask, dtype=tf.float32)
        self_mhc_mask_2d = tf.broadcast_to(self_mhc_mask, (
            tf.shape(m_mask)[0], tf.shape(m_mask)[1], tf.shape(m_mask)[1])) # (B, M, M)
        # one and zero masks
        pep_mhc_mask_secondpart = tf.broadcast_to(p_mask, (tf.shape(p_mask)[0], tf.shape(p_mask)[1], tf.shape(m_mask)[-1])) # (B, N, M)
        pep_mhc_mask_secondpart = pep_mhc_mask_secondpart * tf.transpose(m_mask, perm=[0, 2, 1]) # (B,N,M)*(B,1,M)=(B, N, M)
        mhc_pep_mask_secondpart = tf.broadcast_to(m_mask, (tf.shape(m_mask)[0], tf.shape(m_mask)[1], tf.shape(p_mask)[-1])) # (B, M, N)
        mhc_pep_mask_secondpart = mhc_pep_mask_secondpart * tf.transpose(p_mask, perm=[0, 2, 1]) # (B,M,N)*(B,1,N)=(B, M, N)
        # combined masks (B,N+M,N+M)
        combined_mask_1 = tf.concat([self_peptide_mask_2d, pep_mhc_mask_secondpart], axis=2) # (B, N, N+M)
        combined_mask_2 = tf.concat([mhc_pep_mask_secondpart, self_mhc_mask_2d], axis=2) # (B, M, N+M)
        final_mask = tf.concat([combined_mask_1, combined_mask_2], axis=1) # (B, N+M, N+M)
        return final_mask


class MaskedCategoricalCrossentropyLoss(layers.Layer):
    """    # Define losses
    # 1. reconstruction loss for barcode and MHC separately normalized by sequence length
    # 2. reconstruction loss of masked peptide and MHC positions
    # 3. (optional) reward function for attention weights with respect to anchor rules (eg. attention hotspots must be at least 2 positions apart)
    """
    def __init__(self, name=None, **kwargs):
        super(MaskedCategoricalCrossentropyLoss, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        y_pred, y_true, mask = inputs
        loss_per_position = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1)
        masked_loss = loss_per_position * mask
        total_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
        self.add_loss(total_loss)
        return y_pred


class AddGaussianNoise(layers.Layer):
    def __init__(self, std=0.1, **kw): super().__init__(**kw); self.std = std

    def call(self, x, training=None):
        if training: return x + tf.random.normal(tf.shape(x), stddev=self.std)
        return x


def generate_synthetic_pMHC_data(batch_size=100, max_pep_len=20, max_mhc_len=10):
    # Generate synthetic data
    # Position-specific amino acid frequencies for peptides
    # Simplified frequencies where certain positions prefer specific amino acids
    pep_pos_freq = {
        0: {"A": 0.3, "G": 0.2, "M": 0.2},  # Position 1 prefers A, G, M
        1: {"L": 0.3, "V": 0.3, "I": 0.2},  # Position 2 prefers hydrophobic
        2: {"D": 0.2, "E": 0.2, "N": 0.2},  # Position 3 prefers charged/polar
        3: {"S": 0.3, "T": 0.2, "Q": 0.2},  # Position 4 prefers polar
        4: {"R": 0.3, "K": 0.2, "H": 0.2},  # Position 5 prefers basic
        5: {"F": 0.3, "Y": 0.2, "W": 0.2},  # Position 6 prefers aromatic
        6: {"C": 0.3, "P": 0.2, "A": 0.2},  # Position 7 prefers small residues
        7: {"G": 0.3, "D": 0.2, "E": 0.2},  # Position 8 prefers small/charged
        8: {"L": 0.3, "V": 0.2, "I": 0.2},  # Position 9 prefers hydrophobic
    }
    # Default distribution for other positions
    default_aa_freq = {aa: 1/len(AA) for aa in AA}

    # Generate peptides with position-specific preferences
    pep_lengths = np.random.choice([8, 9, 10, 11, 12], size=batch_size, p=[0.1, 0.5, 0.2, 0.1, 0.1])  # More realistic length distribution
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
        # mask gaps with pad token
        for pos in range(length):
            if pep_seqs[i][pos] == '-':
                mask_pep[i, pos] = PAD_TOKEN

    # MHC alleles typically have conserved regions
    mhc_pos_freq = {
        0: {"G": 0.5, "D": 0.3},  # First position often G or D
        1: {"S": 0.4, "H": 0.3, "F": 0.2},
        2: {"A": 0.3, "T": 0.3, "N": 0.2},  # Position 3 prefers small residues
        3: {"R": 0.4, "K": 0.3, "Q": 0.2},  # Position 4 prefers basic residues
        4: {"L": 0.3, "I": 0.3, "V": 0.2},  # Position 5 prefers hydrophobic residues
        5: {"E": 0.4, "D": 0.3, "N": 0.2},  # Position 6 prefers charged residues
        6: {"C": 0.3, "P": 0.3, "A": 0.2},  # Position 7 prefers small residues
        7: {"Y": 0.4, "W": 0.3, "F": 0.2},  # Position 8 prefers aromatic residues
        8: {"G": 0.3, "D": 0.3, "E": 0.2},  # Position 9 prefers small/charged residues
        9: {"L": 0.3, "V": 0.3, "I": 0.2},  # Position 10 prefers hydrophobic residues
        10: {"R": 0.4, "K": 0.3, "Q": 0.2},  # Position 11 prefers basic residues
        11: {"A": 0.3, "T": 0.3, "N": 0.2},  # Position 12 prefers small residues
        12: {"S": 0.4, "H": 0.3, "F": 0.2},  # Position 13 prefers polar residues
        13: {"G": 0.5, "D": 0.3},  # Position 14 often G or D
        14: {"A": 0.3, "T": 0.3, "N": 0.2},  # Position 15 prefers small residues
        15: {"R": 0.4, "K": 0.3, "Q": 0.2},  # Position 16 prefers basic residues
        16: {"L": 0.3, "I": 0.3, "V": 0.2},  # Position 17 prefers hydrophobic residues
        17: {"E": 0.4, "D": 0.3, "N": 0.2},  # Position 18 prefers charged residues
        18: {"C": 0.3, "P": 0.3, "A": 0.2},  # Position 19 prefers small residues
        19: {"Y": 0.4, "W": 0.3, "F": 0.2},  # Position 20 prefers aromatic residues
        20: {"G": 0.3, "D": 0.3, "E": 0.2},  # Position 21 prefers small/charged residues
        21: {"-": 0.3, "V": 0.3, "I": 0.2},
        22: {"-": 0.4, "K": 0.3, "Q": 0.2},
        23: {"-": 0.3, "A": 0.3, "T": 0.2},  # Position 24 prefers small residues
        24: {"-": 0.4, "S": 0.3, "H": 0.2},  # Position 25 prefers polar residues
        25: {"-": 0.5, "F": 0.3},  # Position 26 often F
        26: {"-": 0.3, "G": 0.3, "D": 0.2},  # Position 27 prefers small/charged residues
        # Add more positions as needed
    }

    # Generate MHC sequences with more realistic properties
    mhc_lengths = np.random.randint(max_mhc_len-5,max_mhc_len, size=batch_size)  # Less variation in length
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
    print(mhc_OHE.shape)

    # Create masks for MHC sequences
    mask_mhc = np.full((batch_size, max_mhc_len), PAD_TOKEN, dtype=np.float32)
    for i, length in enumerate(mhc_lengths):
        mask_mhc[i, :length] = 1.0
        mhc_EMB[i, length:, :] = PAD_VALUE  # set padding positions
        for pos in range(length):
            if mhc_seqs[i][pos] == '-':
                mask_mhc[i, pos] = PAD_TOKEN

    # Generate MHC IDs (could represent allele types)
    mhc_ids = np.random.randint(0, 100, size=(batch_size, max_mhc_len), dtype=np.int32)

    # # mask 0.15 of the peptide positions update the mask with MASK_TOKEN and zero out the corresponding positions in the OHE
    mask_pep[(mask_pep != PAD_TOKEN) & (np.random.rand(batch_size, max_pep_len) < 0.15)] = MASK_TOKEN
    mask_mhc[(mask_mhc != PAD_TOKEN) & (np.random.rand(batch_size, max_mhc_len) < 0.15)] = MASK_TOKEN

    # convert all inputs tensors
    # pep_OHE = tf.convert_to_tensor(pep_OHE, dtype=tf.float32)
    # mask_pep = tf.convert_to_tensor(mask_pep, dtype=tf.float32)
    # mhc_EMB = tf.convert_to_tensor(mhc_EMB, dtype=tf.float32)
    # mask_mhc = tf.convert_to_tensor(mask_mhc, dtype=tf.float32)
    # mhc_OHE = tf.convert_to_tensor(mhc_OHE, dtype=tf.float32)

    # Cov layers
    ks_dict = determine_ks_dict(initial_input_dim=max_mhc_len, output_dims=[16, 14, 12, 11], max_strides=20, max_kernel_size=60)
    if ks_dict is None:
        raise ValueError("Could not determine valid kernel sizes and strides for MHC Conv layers.")

    return pep_OHE, mask_pep, mhc_EMB, mask_mhc, mhc_OHE, mhc_ids, ks_dict

class MaskedCategoricalCELossLayer(keras.layers.Layer):
    def __init__(self, pad_token=-2, mask_token=-1, name="masked_ce_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_token = pad_token
        self.mask_token = mask_token
    def call(self, inputs):
        """
        inputs: list or tuple of (y_true, y_pred, mask)
        y_true … (B,L,21)  one-hot
        y_pred … (B,L,21)  softmax
        mask   … (B,L)     integer mask (1 = valid, pad_token = ignore, etc.)
        """
        y_true, y_pred, mask = inputs

        loss_per_position = tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=False, axis=-1
        )
        # Convert pad/mask tokens to zeros
        mask = tf.where(
            tf.logical_or(tf.equal(mask, self.pad_token), tf.equal(mask, self.mask_token)),
            0.0,
            1.0
        )
        masked_loss = loss_per_position * mask
        total_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask + 1e-8)  # avoid /0
        y_out = tf.expand_dims(tf.cast(mask, tf.float32), -1) * y_pred  # apply mask to predictions
        self.add_loss(total_loss)
        return y_out  # passthrough for prediction


class MaskPaddingLayer(layers.Layer):
    """
    Layer that sets the last index to 1.0
    mask: (B, N) tensor with -2.0 as padding token
    tensor: (B, N, D) tensor to be masked
    set the D[-1] to 1.0 if the mask is -2.0
    """

    def __init__(self, pad_token=-2.0, **kwargs):
        super().__init__(**kwargs)
        self.pad_token = pad_token

    def call(self, inputs):
        tensor, mask = inputs
        mask_exp = tf.equal(mask, self.pad_token)
        mask_exp = tf.expand_dims(mask_exp, axis=-1)
        # Create a tensor of zeros except for the last feature, which is 1.0 where mask is True
        update = tf.one_hot(tf.shape(tensor)[-1] - 1, tf.shape(tensor)[-1], on_value=1.0, off_value=0.0)
        update = tf.reshape(update, (1, 1, -1))
        update = tf.cast(update, tensor.dtype)
        update = tf.tile(update, [tf.shape(tensor)[0], tf.shape(tensor)[1], 1])
        result = tf.where(mask_exp, update, tensor)
        return result

    def get_config(self):
        config = super().get_config()
        config.update({"pad_token": self.pad_token})
        return config

def get_embed_key(key: str, emb_dict: Dict[str, np.ndarray]) -> str: # why ndarray?
    """
    Get the embedding key for a given allele key.
    If the key is not found in the embedding dictionary, return None.
    # find the matching emb key in the emb_dict.
    Sometimes the emb key is longer than the allele key, so we need to check if the key is a substring of the emb key.
    """
    # Use a generator expression for efficient lookup
    return next((emb_key for emb_key in emb_dict if emb_key.upper().startswith(key.upper())), None)


def process_pdb_distance_matrix(pdb_path, threshold, peptide, chainid='A', carbon='CB'):
    # Parse PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    chain = structure[0][chainid]  # Assumes single chain 'A'
    # Extract CB (or CA for Glycine) coordinates
    coords = []
    carbon_not = 'CA' if carbon == 'CB' else 'CB'
    for res in chain:
        if res.get_resname() == 'GLY':  # Glycine
            atom = res['CA']
        else:
            atom = res[carbon] if carbon in res else res[carbon_not]
        coords.append(atom.get_coord())
    coords = np.array(coords)
    # Compute distance matrix
    dist_matrix = squareform(pdist(coords, metric='euclidean'))
    #mask = np.where(dist_matrix < threshold, 0., 1)
    mask = np.eye(dist_matrix.shape[0], dtype=np.float32) # all masks are==1
    mask = np.where(mask==1, 0, 1) # all masks are==0 now
    dist_matrix = 1 / (dist_matrix + 1e-9)
    result = dist_matrix * mask
    #result = result[len(result)-len(peptide):, :len(result)-len(peptide)]
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    return result

# res = process_pdb_distance_matrix('../pdbs/MHCI_ref_4U6Y.pdb', 9, "FLNKDLEVDGHFVTM", 'C', 1.5)


# # How to use
# thr = 9
# chain = 'A'
# carbons = ['CA', 'CB']
# final_dict = {}
# for i, row in df.iterrows():
#     allele = row['allale']
#     peptide = row['peptide']
#     pdb_path = row['pdb_path']
#     out = []
#     for carbon in carbons:
#         res = process_pdb_distance_matrix(pdb_path, thr, peptide, chain, carbon)
#         out.append(np.expand_dims(res, axis=-1))
#     out = np.concatenate(out, axis=-1) #(P,M,2)
#     final_dict[allele] = out


def cluster_aa_sequences(
    csv_path: str,
    *,
    id_col: str = "allele",
    seq_col: str = "sequence",
    linkage_method: str = "average",
    gap_mode: Literal["count_gaps", "ignore_gaps"] = "ignore_gaps",
    k: int | None = None,
    distance_threshold: float | None = None,
    plot: bool = True,
    mafft_extra: str | None = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Cluster amino-acid sequences contained in `csv_path`.

    Parameters
    ----------
    csv_path : str
        CSV file with at least columns `id_col` and `seq_col`.
    id_col, seq_col : str
        Column names for sequence IDs and sequences.
    linkage_method : str
        Method for scipy.cluster.hierarchy.linkage.
    gap_mode : {"count_gaps", "ignore_gaps"}
        How gaps are treated in the pair-wise distance.
    k : int, optional
        If given, returns exactly k clusters (criterion="maxclust").
    distance_threshold : float, optional
        If given, cut the dendrogram at this height (criterion="distance").
    plot : bool
        If True, show a dendrogram.
    mafft_extra : str, optional
        Extra options appended to the MAFFT command line.

    Returns
    -------
    df_out : pandas.DataFrame
        Original columns plus a new column "Cluster".
    Z : numpy.ndarray
        The linkage matrix from `scipy.cluster.hierarchy.linkage`.
    """
    df = pd.read_csv(csv_path)
    if id_col not in df.columns or seq_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{id_col}' and '{seq_col}'")

    ids  = df[id_col].astype(str).tolist()
    seqs = df[seq_col].astype(str).tolist()

    # 2. Write sequences to a temporary FASTA file
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".fasta") as tmp_fa:
        for ident, seq in zip(ids, seqs):
            tmp_fa.write(f">{ident}\n{seq}\n")
        fasta_path = tmp_fa.name

    # 3. Run MAFFT
    extra = mafft_extra if mafft_extra else ""
    mafft_cline = MafftCommandline(input=fasta_path, auto=True)
    if extra:
        mafft_cline.cline += f" {extra}"
    print("Running MAFFT …")
    mafft_stdout, _ = mafft_cline()
    os.unlink(fasta_path)                      # clean up temp file
    aln = AlignIO.read(StringIO(mafft_stdout), "fasta")
    aligned_ids  = [rec.id  for rec in aln]
    aligned_seqs = [str(rec.seq) for rec in aln]
    print(f"Alignment length = {aln.get_alignment_length()} aa")

    # 4. Pair-wise distances on aligned sequences
    def dist_count_gaps(a, b):
        return sum(c1 != c2 for c1, c2 in zip(a, b)) / len(a)

    def dist_ignore_gaps(a, b):
        same = diff = 0
        for c1, c2 in zip(a, b):
            if c1 == "-" or c2 == "-":
                continue
            same += 1
            if c1 != c2:
                diff += 1
        return 0.0 if same == 0 else diff / same

    seq_dist = dist_count_gaps if gap_mode == "count_gaps" else dist_ignore_gaps

    X = np.array(aligned_seqs, dtype=object)[:, None]            # shape (N,1)
    dist_vector = pdist(X, lambda u, v: seq_dist(u[0], v[0]))

    # 5. Hierarchical clustering
    Z = linkage(dist_vector, method=linkage_method)

    # 6. Retrieve flat clusters
    if k is None and distance_threshold is None:
        raise ValueError("Specify either `k` or `distance_threshold`")

    if k is not None:
        clusters = fcluster(Z, t=k, criterion="maxclust")
    else:
        clusters = fcluster(Z, t=distance_threshold, criterion="distance")

    # 7. Optional dendrogram
    if plot:
        plt.figure(figsize=(25, 5))
        dendrogram(
            Z,
            labels=aligned_ids,
            leaf_rotation=90,
            leaf_font_size=9,
            color_threshold=distance_threshold if distance_threshold else None,
        )
        plt.title("Hierarchical clustering (MAFFT aligned)")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.show()


    df_out = df.copy()
    df_out["cluster"] = clusters
    return df_out, Z


# # Example usage
# if __name__ == "__main__":
#     # Cluster into *exactly* 4 clusters and show a dendrogram
#     df_clusters, Z = cluster_aa_sequences(
#         "allele_stats_class2_with_seq.csv",
#         k=8,
#         linkage_method="average",
#         gap_mode="ignore_gaps",
#         plot=True,
#     )
#
#     print(df_clusters[["allele", "cluster"]].head())


def create_k_fold_leave_one_cluster_out_stratified_cv(
    df: pd.DataFrame,
    k: int = 5,
    target_col: str = "label",
    cluster_col: str = "cluster",
    id_col: str = "allele",
    subset_prop: float = 1.0,
    train_size: float = 0.8,
    random_state: int = 42,
    augmentation: str = "down_sampling",  # {"down_sampling", "GNUSS", None}
    keep_val_only_cluster: bool = True,   # set False if you do NOT want to have a cluster that is only in validation
) -> List[Tuple[pd.DataFrame, pd.DataFrame, Hashable]]:
    """
    Build *k* folds where ***one whole sequence cluster is left out*** of
    both train and validation.

    Workflow for every fold
    -----------------------
    1. Choose one cluster -> `left_out_cluster` (never in train or val).
    2. If `keep_val_only_cluster` is True:
       pick one additional cluster -> `val_only_cluster`
       (present only in validation).
    3. Split remaining rows into train / extra-validation stratified on
       `target_col` with proportion `train_size`.
    4. Apply class-balancing augmentation (down-sampling or GNUSS).

    Returns
    -------
    list[(train_df, val_df, left_out_cluster)]
    """

    rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------
    # (optional) take a subset of the whole frame
    # ------------------------------------------------------------
    if subset_prop < 1.0:
        if not (0.0 < subset_prop <= 1.0):
            raise ValueError("subset_prop must be in (0,1]")
        df = df.sample(frac=subset_prop, random_state=random_state).reset_index(drop=True)
        print(f"Using a {subset_prop:.2%} random subset of the data.")

    # ------------------------------------------------------------
    # 0) basic sanity checks
    # ------------------------------------------------------------
    if cluster_col not in df.columns:
        raise ValueError(f"{cluster_col=} not found in DataFrame")
    if target_col not in df.columns:
        raise ValueError(f"{target_col=} not found in DataFrame")

    unique_clusters = df[cluster_col].unique()
    if k > len(unique_clusters):
        raise ValueError(f"k={k} > #unique clusters ({len(unique_clusters)})")

    left_out_clusters = rng.choice(unique_clusters, size=k, replace=False)
    left_out_ids = df[df[cluster_col].isin(left_out_clusters)][id_col].unique()

    # save the left out ids to a txt file
    with open("left_out_ids.txt", "w") as f:
        for left_out_id in left_out_ids:
            f.write(f"{left_out_id}\n")

    # ------------------------------------------------------------
    # augmentation helpers
    # ------------------------------------------------------------
    def _balance_down_sampling(frame: pd.DataFrame, seed: int):
        min_count = frame[target_col].value_counts().min()
        parts = [
            resample(
                frame[frame[target_col] == lbl],
                replace=False,
                n_samples=min_count,
                random_state=seed,
            )
            for lbl in frame[target_col].unique()
        ]
        return pd.concat(parts, ignore_index=True)

    def _balance_GNUSS(frame: pd.DataFrame, seed: int):
        counts = frame[target_col].value_counts()
        max_count = counts.max()

        numeric_cols = frame.select_dtypes(include="number").columns
        parts = [frame]  # originals

        for label, cnt in counts.items():
            if cnt == max_count:
                continue
            need = max_count - cnt
            sampled = frame[frame[target_col] == label].sample(
                n=need, replace=True, random_state=seed
            )
            # very small noise
            noise = rng.normal(loc=0.0, scale=1e-6, size=(need, len(numeric_cols)))
            sampled.loc[:, numeric_cols] += noise
            parts.append(sampled)

        return pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------
    # build each fold
    # ------------------------------------------------------------
    folds: List[Tuple[pd.DataFrame, pd.DataFrame, Hashable]] = []

    for fold_idx, left_out_cluster in enumerate(left_out_clusters, 1):
        fold_seed = random_state + fold_idx

        # remove the left-out cluster
        mask_left_out = df[cluster_col] == left_out_cluster
        working_df = df.loc[~mask_left_out].copy()

        # pick ONE cluster that will be *only* in validation
        if keep_val_only_cluster:
            gss = GroupShuffleSplit(n_splits=1, test_size=1, random_state=fold_seed)
            (__, val_only_idx), = gss.split(
                X=np.zeros(len(working_df)),
                groups=working_df[cluster_col],
            )
            val_only_cluster = working_df.iloc[val_only_idx][cluster_col].unique()[0]

            mask_val_only = working_df[cluster_col] == val_only_cluster
            df_val_only = working_df[mask_val_only]
            df_eligible = working_df[~mask_val_only]
        else:
            val_only_cluster = None
            df_val_only = pd.DataFrame(columns=df.columns)
            df_eligible = working_df

        # stratified split of the remaining rows
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=train_size, random_state=fold_seed
        )
        train_idx, extra_val_idx = next(
            sss.split(df_eligible, df_eligible[target_col])
        )
        df_train = df_eligible.iloc[train_idx]
        df_val   = pd.concat(
            [df_val_only, df_eligible.iloc[extra_val_idx]],
            ignore_index=True,
        )

        # ---------------------------------------------------- #
        #  balance
        # ---------------------------------------------------- #
        if augmentation == "GNUSS":
            df_train_bal = _balance_GNUSS(df_train, fold_seed)
            df_val_bal   = _balance_GNUSS(df_val, fold_seed)
        elif augmentation == "down_sampling":
            df_train_bal = _balance_down_sampling(df_train, fold_seed)
            df_val_bal   = _balance_down_sampling(df_val, fold_seed)
        elif augmentation is None or augmentation.lower() == "none":
            df_train_bal = df_train.copy()
            df_val_bal   = df_val.copy()
        else:
            raise ValueError(f"Unknown augmentation: {augmentation}")

        df_train_bal = (
            df_train_bal.sample(frac=1.0, random_state=fold_seed).reset_index(drop=True)
        )
        df_val_bal = (
            df_val_bal.sample(frac=1.0, random_state=fold_seed).reset_index(drop=True)
        )

        folds.append((df_train_bal, df_val_bal, left_out_cluster))

        print(
            f"[fold {fold_idx}/{k}] left-out cluster={left_out_cluster} | "
            f"val-only cluster={val_only_cluster} | "
            f"train={len(df_train_bal)}, val={len(df_val_bal)}"
        )

    return folds

# Usage example
#
# df_analysis = pd.read_csv("analysis_dataset_with_clusters.csv")
#
# folds = create_k_fold_leave_one_cluster_out_stratified_cv(
#     df_analysis,
#     k=5,
#     target_col="assigned_label",
#     cluster_col="cluster",
#     id_col="allele",
#     augmentation="down_sampling",
# )