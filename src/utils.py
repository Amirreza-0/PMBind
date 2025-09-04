import tensorflow as tf
from keras.src.api_export import keras_export
from keras.src.losses import LossFunctionWrapper
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.utils import resample
from tensorflow import keras
from tensorflow.keras import layers
from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
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
from collections import Counter


# Constants
BLOSUM62 = {
    'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0],
    'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1],
    'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1],
    'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1],
    'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2],
    'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1],
    'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1],
    'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1],
    'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1],
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1],
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1],
    'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1],
    'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1],
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1],
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2],
    'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0],
    'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0],
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2],
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1],
    'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1],
    'B': [-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1],
    'Z': [-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1],
    'X': [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1],
}

# Create a reverse mapping from a BLOSUM62 vector to an amino acid character.
# The score lists are converted to tuples so they can be used as dictionary keys.
VECTOR_TO_AA = {tuple(v): k for k, v in BLOSUM62.items()}

AA = "ACDEFGHIKLMNPQRSTVWY-"
AA_TO_INT = {a: i for i, a in enumerate(AA)}
UNK_IDX = 20  # Index for "unknown"
MASK_TOKEN = -1.0
NORM_TOKEN = 1.0
PAD_TOKEN = -2.0
PAD_VALUE = 0.0
MASK_VALUE = 0.0


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

def peptides_to_onehot_kmer_windows(seq, max_seq_len, k=9, pad_token=-1.0) -> np.ndarray:
    """
    Converts a peptide sequence into a sliding window of k-mers, one-hot encoded.
    Output shape: (RF, k, 21), where RF = max_seq_len - k + 1
    """
    RF = max_seq_len - k + 1
    RFs = np.zeros((RF, k, 21), dtype=np.float32)
    for window in range(RF):
        if window + k <= len(seq):
            kmer = seq[window:window + k]
            for i, aa in enumerate(kmer):
                idx = AA_TO_INT.get(aa, pad_token)
                RFs[window, i, idx] = 1.0
            # Pad remaining positions in k-mer if sequence is too short
            for i in range(len(kmer), k):
                RFs[window, i, pad_token] = 1.0
        else:
            # Entire k-mer is padding if out of sequence
            RFs[window, :, pad_token] = 1.0
    return np.array(RFs)

class OHEKmerWindows(tf.keras.layers.Layer):
    """
    A TensorFlow layer that converts a batch of one-hot encoded sequences
    into one-hot encoded k-mer windows.

    This layer takes a 3D tensor of one-hot encoded sequences and produces a
    4D tensor of one-hot encoded sliding windows. It is a more direct alternative
    to the string-based version when data is already pre-processed.
    """
    def __init__(self, max_seq_len: int, k: int = 9, alphabet_size: int = 21, name='ohe_kmer_windows'):
        """
        Initializes the layer.

        Args:
            max_seq_len (int): The length of the input sequences.
            k (int, optional): The size of the k-mer window. Defaults to 9.
            alphabet_size (int, optional): The depth of the one-hot encoding
                                         (e.g., 21 for amino acids + unk). Defaults to 21.
        """
        super(OHEKmerWindows).__init__(name=name)
        self.max_seq_len = max_seq_len
        self.k = k
        self.alphabet_size = alphabet_size
        self.rf = max_seq_len - k + 1  # Number of receptive fields (windows)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Processes the input tensor of one-hot encoded sequences.

        Args:
            inputs (tf.Tensor): A 3D tensor of one-hot encoded sequences.
                                Shape: (batch_size, max_seq_len, alphabet_size)

        Returns:
            tf.Tensor: The one-hot encoded k-mer windows.
                       Shape: (batch_size, self.rf, self.k, self.alphabet_size)
        """
        # --- Input Validation (Optional but Recommended) ---
        input_shape = tf.shape(inputs)
        tf.debugging.assert_equal(input_shape[1], self.max_seq_len,
            message=f"Input sequence length must be {self.max_seq_len}")
        tf.debugging.assert_equal(input_shape[2], self.alphabet_size,
            message=f"Input alphabet size must be {self.alphabet_size}")

        # 1. Reshape the input to be compatible with `extract_patches`
        # We treat the sequence as a 1D "image" with `alphabet_size` channels.
        # Shape: (batch, max_seq_len, alphabet_size) -> (batch, max_seq_len, 1, alphabet_size)
        images = tf.expand_dims(inputs, axis=2)

        # 2. Extract sliding windows (patches) of size k
        # We slide a window of size `k` along the `max_seq_len` dimension.
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.k, 1, 1],      # Window size: (batch, height, width, channels)
            strides=[1, 1, 1, 1],    # Slide one step at a time
            rates=[1, 1, 1, 1],
            padding='VALID'          # 'VALID' ensures we only take full windows
        )
        # The output shape of extract_patches is (batch, num_windows_h, num_windows_w, k * 1 * alphabet_size)
        # In our case: (batch_size, self.rf, 1, k * self.alphabet_size)

        # 3. Reshape the patches into the desired final format
        # Shape: (batch_size, self.rf, 1, k * alphabet_size) -> (batch_size, self.rf, k, alphabet_size)
        kmer_windows = tf.reshape(
            patches,
            [-1, self.rf, self.k, self.alphabet_size]
        )

        return kmer_windows

    def get_config(self):
        """Enables layer serialization."""
        config = super().get_config()
        config.update({
            "max_seq_len": self.max_seq_len,
            "k": self.k,
            "alphabet_size": self.alphabet_size,
        })
        return config


def seq_to_blossom62(sequence: str, max_seq_len: int) -> np.ndarray:
    """
    Converts a peptide sequence into a matrix using BLOSUM62 substitution scores.

    This function maps each amino acid in the input sequence to its corresponding
    vector of substitution scores from the BLOSUM62 matrix. The resulting matrix
    is padded or truncated to a specified maximum length.

    Args:
        sequence (str): The input peptide sequence.
        max_seq_len (int): The target length for the output matrix. Sequences
                           shorter than this will be padded with zeros, and
                           longer ones will be truncated.

    Returns:
        np.ndarray: A NumPy array of shape (max_seq_len, 23) where each row
                    corresponds to an amino acid's BLOSUM62 scores.
    """
    # The BLOSUM62 matrix has 23 columns corresponding to the score list length.
    num_features = 23

    # Initialize the output array with the padding value (0.0).
    arr = np.full((max_seq_len, num_features), PAD_VALUE, dtype=np.float32)

    # Use the vector for 'X' (unknown) as the default for any character
    # not found in the BLOSUM62 dictionary, including gaps ('-').
    default_vector = BLOSUM62['X']

    # Iterate over the sequence up to the maximum length.
    for i, aa in enumerate(sequence.upper()[:max_seq_len]):
        # Retrieve the BLOSUM62 vector for the current amino acid.
        # If the amino acid is not a key in the dictionary, use the default vector.
        blosum_vector = BLOSUM62.get(aa, default_vector)
        arr[i, :] = blosum_vector

    return arr


def blosum62_to_seq(blosum_matrix: np.ndarray) -> str:
    """
    Converts a BLOSUM62 matrix back into a peptide sequence.

    This function iterates through each row of the input matrix, finds the
    corresponding amino acid from the BLOSUM62 mapping, and reconstructs the
    sequence. It stops when it encounters a padding row (all zeros).

    Args:
        blosum_matrix (np.ndarray): A NumPy array of shape (N, 23) containing
                                    BLOSUM62 scores.

    Returns:
        str: The reconstructed peptide sequence.
    """
    sequence = []

    # Iterate through each row (vector) in the input matrix.
    for row in blosum_matrix:
        # Check if the row is a padding vector (all elements are PAD_VALUE).
        # If so, we have reached the end of the sequence.
        if not np.any(row):
            break

        # Convert the NumPy array row to a tuple to make it hashable.
        row_tuple = tuple(np.array(row))

        # Look up the amino acid in the reverse map. Default to 'X' if not found.
        amino_acid = VECTOR_TO_AA.get(row_tuple, 'X')
        sequence.append(amino_acid)

    return "".join(sequence)

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
                 epsilon=1e-6, gate=True, mask_token=-1., pad_token=-2., dtype="float32"):
        super().__init__(name=name, dtype=dtype)
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

    def build(self, input_shape):
        # Projection weights
        self.q_proj = self.add_weight(shape=(self.heads, self.query_dim, self.att_dim),
                                      initializer='random_normal', trainable=True, name=f'q_proj_{self.name}', dtype=self.dtype)
        self.k_proj = self.add_weight(shape=(self.heads, self.context_dim, self.att_dim),
                                      initializer='random_normal', trainable=True, name=f'k_proj_{self.name}', dtype=self.dtype)
        self.v_proj = self.add_weight(shape=(self.heads, self.context_dim, self.att_dim),
                                      initializer='random_normal', trainable=True, name=f'v_proj_{self.name}', dtype=self.dtype)
        if self.gate:
            self.g = self.add_weight(shape=(self.heads, self.query_dim, self.att_dim),
                                     initializer='random_uniform', trainable=True, name=f'gate_{self.name}', dtype=self.dtype)
        self.norm = layers.LayerNormalization(epsilon=self.epsilon, name=f'ln_{self.name}', dtype=self.dtype)
        if self.type == 'cross':
            self.norm_context = layers.LayerNormalization(epsilon=self.epsilon, name=f'ln_context_{self.name}', dtype=self.dtype)
        self.norm_out = layers.LayerNormalization(epsilon=self.epsilon, name=f'ln_out_{self.name}', dtype=self.dtype)
        self.out_w = self.add_weight(shape=(self.heads * self.att_dim, self.output_dim),
                                     initializer='random_normal', trainable=True, name=f'outw_{self.name}', dtype=self.dtype)
        self.out_b = self.add_weight(shape=(self.output_dim,), initializer='zeros',
                                     trainable=True, name=f'outb_{self.name}', dtype=self.dtype)
        self.scale = 1.0 / tf.math.sqrt(tf.cast(self.att_dim, tf.float32))

    def call(self, x, mask, context=None, context_mask=None):
        """
        Args:
            x: Tensor of shape (B, N, query_dim) for query.
            mask: Tensor of shape (B, N).
            context: Tensor of shape (B, M, context_dim) for key/value in cross-attention.
            context_mask: Tensor of shape (B, M) for context.
        """
        mask = tf.cast(mask, self.compute_dtype)
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

        out = self.norm_out(out)
        mask_exp = tf.expand_dims(mask_q, axis=-1)
        out *= mask_exp

        return (out, att) if self.return_att_weights else out


class SubtractAttentionLayer(keras.layers.Layer):
    """
    Multi-head self-attention for the subtraction tensor  (B, M, P·D).

    Inputs
    ──────
        x_sub          : (B, M, P*D)
        combined_mask  : (B, M, P*D)   (True = valid, False = padding)

    All other arguments / behaviour are identical to the original
    `AttentionLayer` (projection sizes, heads, residual, gating, …).
    """
    def __init__(self,
                 feature_dim,          # == P*D   (input & output dim)
                 heads=4,
                 resnet=True,
                 return_att_weights=False,
                 name="sub_attention",
                 epsilon=1e-6,
                 gate=True):
        super().__init__(name=name)

        assert feature_dim % heads == 0, "feature_dim must be divisible by heads"
        self.feature_dim = feature_dim
        self.output_dim  = feature_dim
        self.heads       = heads
        self.resnet      = resnet
        self.return_att  = return_att_weights
        self.epsilon     = epsilon
        self.gate        = gate
        self.att_dim     = feature_dim // heads          # per-head dim
        self.scale       = 1.0 / tf.math.sqrt(tf.cast(self.att_dim,
                                                      tf.float32))

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def build(self, _):
        hd, fd = self.heads, self.feature_dim

        self.q_proj = self.add_weight(name="q_proj", shape=(hd, fd, self.att_dim), initializer="random_normal")
        self.k_proj = self.add_weight(name="k_proj", shape=(hd, fd, self.att_dim), initializer="random_normal")
        self.v_proj = self.add_weight(name="v_proj", shape=(hd, fd, self.att_dim), initializer="random_normal")

        if self.gate:
            self.g = self.add_weight(name="gate", shape=(hd, fd, self.att_dim), initializer="random_uniform")

        self.ln_in   = layers.LayerNormalization(epsilon=self.epsilon,
                                                 name="ln_in")
        self.ln_out  = layers.LayerNormalization(epsilon=self.epsilon,
                                                 name="ln_out")
        if self.resnet:
            self.ln_res = layers.LayerNormalization(epsilon=self.epsilon,
                                                    name="ln_res")

        self.out_w = self.add_weight(name="out_w", shape=(hd * self.att_dim, fd), initializer="random_normal")
        self.out_b = self.add_weight(name="out_b", shape=(fd,), initializer="zeros")

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def call(self, x_sub, combined_mask):
        """
        x_sub         : (B, M, P*D)
        combined_mask : (B, M, P*D)  (bool)
        """
        B = tf.shape(x_sub)[0]
        M = tf.shape(x_sub)[1]

        # ── token-level mask (B,M) – “is *any* feature in this row valid?” ──
        token_mask = tf.reduce_any(combined_mask, axis=-1)            # bool
        token_mask_f = tf.cast(token_mask, tf.float32)                # 1/0

        # ── layer-norm on valid data only (invalid features are already 0) ──
        x_norm = self.ln_in(x_sub)

        # ── project Q K V ──────────────────────────────────────────────
        #    x_norm : (B, M, F) ;  proj  : (H, F, d)  →  (B, H, M, d)
        q = tf.einsum('bmd,hdf->bhmf', x_norm, self.q_proj)
        k = tf.einsum('bmd,hdf->bhmf', x_norm, self.k_proj)
        v = tf.einsum('bmd,hdf->bhmf', x_norm, self.v_proj)

        # ── scaled dot-product attention ───────────────────────────────
        att = tf.einsum('bhmf,bhnf->bhmn', q, k) * self.scale  # (B,H,M,M)

        # build broadcast mask  (B,H,M,M)
        mask_q = tf.expand_dims(token_mask_f, axis=1)          # (B,1,M)
        mask_k = tf.expand_dims(token_mask_f, axis=1)          # (B,1,M)
        att_mask = tf.einsum('bqm,bkn->bqmn', mask_q, mask_k)  # (B,1,M,M)
        att_mask = tf.broadcast_to(att_mask, tf.shape(att))

        att += (1.0 - att_mask) * -1e9
        att  = tf.nn.softmax(att, axis=-1) * att_mask          # masked softmax

        # ── attention output ───────────────────────────────────────────
        out = tf.einsum('bhmn,bhnf->bhmf', att, v)             # (B,H,M,d)

        # optional gating
        if self.gate:
            g = tf.einsum('bmd,hdf->bhmf', x_norm, self.g)
            out *= tf.nn.sigmoid(g)

        # ── merge heads ────────────────────────────────────────────────
        out = tf.transpose(out, [0, 2, 1, 3])                  # (B,M,H,d)
        out = tf.reshape(out, [B, M, self.heads * self.att_dim])  # (B,M,F)
        out = tf.matmul(out, self.out_w) + self.out_b

        # residual + norms
        if self.resnet:
            out = self.ln_res(out + x_sub)
        out = self.ln_out(out)

        # finally zero-out padded tokens again (safety)
        out *= token_mask_f[..., tf.newaxis]

        if self.return_att:
            return out, att
        return out


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


    def get_config(self):
        """Serializes the layer's configuration."""
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'max_len': self.max_len,
            'mask_token': self.mask_token,
            'pad_token': self.pad_token,
        })
        return config


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


class RotaryPositionalEncoding(keras.layers.Layer):
    """
    Rotary Positional Encoding layer for transformer models.
    Applies rotary embeddings to the last dimension of the input.
    Args:
        embed_dim (int): Embedding dimension (must be even).
        max_len (int): Maximum sequence length.
        mask_token (float): Value representing a masked token in the mask.
        pad_token (float): Value representing a padded token in the mask.
    """

    def __init__(self, embed_dim=None, max_len: int = 100, mask_token: float = -1., pad_token: float = -2.,
                 name: str = 'rotary_positional_encoding'):
        super().__init__(name=name)
        if embed_dim is not None:
            assert embed_dim % 2 == 0, "embed_dim must be even for rotary encoding"
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.mask_token = mask_token
        self.pad_token = pad_token

    def build(self, input_shape):
        """Precomputes the rotary frequency sinusoids."""
        if self.embed_dim is None:
            self.embed_dim = input_shape[-1]
        assert self.embed_dim % 2 == 0, f"Input feature dimension {self.embed_dim} must be even for rotary encoding"

        # Precompute rotary frequencies
        pos = tf.range(self.max_len, dtype=tf.float32)[:, tf.newaxis]  # (max_len, 1)
        dim = tf.range(self.embed_dim // 2, dtype=tf.float32)[tf.newaxis, :]  # (1, embed_dim//2)

        # Using a base of 10000 is common in RoPE implementations
        inv_freq = tf.pow(10000.0, -(2 * dim) / tf.cast(self.embed_dim, tf.float32))
        freqs = pos * inv_freq  # (max_len, embed_dim//2)

        self.cos_cached = tf.cast(tf.cos(freqs), tf.float32)  # (max_len, embed_dim//2)
        self.sin_cached = tf.cast(tf.sin(freqs), tf.float32)  # (max_len, embed_dim//2)
        super().build(input_shape)

    def call(self, x, mask):
        """
        Applies the rotary positional encoding to the input tensor.
        Args:
            x: Input tensor of shape (B, N, D)
            mask: Tensor of shape (B, N)
        Returns:
            Tensor with rotary positional encoding applied.
        """
        # --- FIX STARTS HERE ---
        # Add a runtime assertion to ensure the last dimension of the input is even.
        # This provides a clearer error if an incorrectly shaped tensor is passed.
        input_shape = tf.shape(x)
        last_dim = input_shape[-1]
        tf.Assert(tf.equal(last_dim % 2, 0),
                  [f"The last dimension of the input tensor to RotaryPositionalEncoding must be even, but received shape {input_shape}."])
        # --- FIX ENDS HERE ---

        seq_len = tf.shape(x)[1]

        # Slice the precomputed frequencies to match the sequence length
        cos = self.cos_cached[:seq_len, :]  # (N, D//2)
        sin = self.sin_cached[:seq_len, :]  # (N, D//2)

        # Add a batch dimension for broadcasting
        cos = tf.expand_dims(cos, 0)  # (1, N, D//2)
        sin = tf.expand_dims(sin, 0)  # (1, N, D//2)

        # Split the input tensor into two halves along the feature dimension
        x1, x2 = tf.split(x, 2, axis=-1)  # (B, N, D//2), (B, N, D//2)

        # Apply the rotary transformation
        x_rot = tf.concat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)  # (B, N, D)

        # Apply the padding mask to zero out padded positions
        mask_expanded = tf.cast(mask[:, :, tf.newaxis], tf.float32)  # (B, N, 1)
        # Create a binary mask where padded tokens are 0 and others are 1
        binary_mask = tf.where(mask_expanded == self.pad_token, 0.0, 1.0)
        x_rot = x_rot * binary_mask  # Zero out positions where mask is 0

        return x_rot

# @tf.function
# def select_indices(ind, n, m_range):
#     """
#     Select top-n indices from `ind` (descending sorted) such that:
#     - First index is always selected.
#     - Each subsequent index has a distance from all previously selected
#       indices between m_range[0] and m_range[1], inclusive.
#     Args:
#         ind: Tensor of shape (B, N) with descending sorted indices.
#         n: Number of indices to select.
#         m_range: List or tuple [min_distance, max_distance]
#     Returns:
#         Tensor of shape (B, n) with selected indices per batch.
#     """
#     m_min = tf.constant(m_range[0], dtype=tf.int32)
#     m_max = tf.constant(m_range[1], dtype=tf.int32)
#
#     def per_batch_select(indices):
#         top = indices[0]
#         selected = tf.TensorArray(dtype=tf.int32, size=n)
#         selected = selected.write(0, top)
#         count = tf.constant(1)
#         i = tf.constant(1)
#
#         def cond(i, count, selected):
#             return tf.logical_and(i < tf.shape(indices)[0], count < n)
#
#         def body(i, count, selected):
#             candidate = indices[i]
#             selected_vals = selected.stack()[:count]
#             distances = tf.abs(selected_vals - candidate)
#             if_valid = tf.reduce_all(
#                 tf.logical_and(distances >= m_min, distances <= m_max)
#             )
#             selected = tf.cond(if_valid,
#                                lambda: selected.write(count, candidate),
#                                lambda: selected)
#             count = tf.cond(if_valid, lambda: count + 1, lambda: count)
#             return i + 1, count, selected
#
#         _, _, selected = tf.while_loop(
#             cond, body, [i, count, selected],
#             shape_invariants=[i.get_shape(), count.get_shape(), tf.TensorShape(None)]
#         )
#         return selected.stack()
#
#     return tf.map_fn(per_batch_select, ind, dtype=tf.int32)
#
#
# class AnchorPositionExtractor(keras.layers.Layer):
#     def __init__(self, num_anchors, dist_thr, name='anchor_extractor', project=True,
#                  mask_token=-1., pad_token=-2., return_att_weights=False):
#         super().__init__()
#         assert isinstance(dist_thr, list) and len(dist_thr) == 2
#         assert num_anchors > 0
#         self.num_anchors = num_anchors
#         self.dist_thr = dist_thr
#         self.name = name
#         self.project = project
#         self.mask_token = mask_token
#         self.pad_token = pad_token
#         self.return_att_weights = return_att_weights
#
#     def build(self, input_shape):  # att_out (B,N,E)
#         b, n, e = input_shape[0], input_shape[1], input_shape[2]
#         self.barcode = tf.random.uniform(shape=(1, 1, e))  # add as a token to input
#         self.q = self.add_weight(shape=(e, e),
#                                  initializer='random_normal',
#                                  trainable=True, name=f'query_{self.name}')
#         self.k = self.add_weight(shape=(e, e),
#                                  initializer='random_normal',
#                                  trainable=True, name=f'key_{self.name}')
#         self.v = self.add_weight(shape=(e, e),
#                                  initializer='random_normal',
#                                  trainable=True, name=f'value_{self.name}')
#         self.ln = layers.LayerNormalization(name=f'ln_{self.name}')
#         if self.project:
#             self.g = self.add_weight(shape=(self.num_anchors, e, e),
#                                      initializer='random_uniform',
#                                      trainable=True, name=f'gate_{self.name}')
#             self.w = self.add_weight(shape=(1, self.num_anchors, e, e),
#                                      initializer='random_normal',
#                                      trainable=True, name=f'w_{self.name}')
#
#     def call(self, input, mask):  # (B,N,E) this is peptide embedding and (B,N) for mask
#
#         mask = tf.cast(mask, tf.float32)  # (B, N)
#         mask = tf.where(mask == self.pad_token, 0., 1.)
#
#         barcode = self.barcode
#         barcode = tf.broadcast_to(barcode, (tf.shape(input)[0], 1, tf.shape(input)[-1]))  # (B,1,E)
#         q = tf.matmul(barcode, self.q)  # (B,1,E)*(E,E)->(B,1,E)
#         k = tf.matmul(input, self.k)  # (B,N,E)*(E,E)->(B,N,E)
#         v = tf.matmul(input, self.v)  # (B,N,E)*(E,E)->(B,N,E)
#         scale = 1 / tf.math.sqrt(tf.cast(tf.shape(input)[-1], tf.float32))
#         barcode_att = tf.matmul(q, k, transpose_b=True) * scale  # (B,1,E)*(B,E,N)->(B,1,N)
#         # mask: (B,N) => (B,1,N)
#         mask_exp = tf.expand_dims(mask, axis=1)
#         additive_mask = (1.0 - mask_exp) * -1e9
#         barcode_att += additive_mask
#         barcode_att = tf.nn.softmax(barcode_att)
#         barcode_att *= mask_exp  # to remove the impact of row wise attention of padded tokens. since all are 1e-9
#         barcode_out = tf.matmul(barcode_att, v)  # (B,1,N)*(B,N,E)->(B,1,E)
#         # barcode_out represents a vector for all information from peptide
#         # barcode_att represents the anchor positions which are the tokens with highest weights
#         inds, weights, outs = self.find_anchor(input,
#                                                barcode_att)  # (B,num_anchors) (B,num_anchors) (B, num_anchors, E)
#         if self.project:
#             pos_encoding = tf.broadcast_to(
#                 tf.expand_dims(inds, axis=-1),
#                 (tf.shape(outs)[0], tf.shape(outs)[1], tf.shape(outs)[2])
#             )
#             pos_encoding = tf.cast(pos_encoding, tf.float32)
#             dim = tf.cast(tf.shape(outs)[-1], tf.float32)
#             ra = tf.range(dim, dtype=tf.float32) / dim
#             pos_encoding = tf.sin(pos_encoding / tf.pow(40., ra))
#             outs += pos_encoding
#
#             weights_bc = tf.expand_dims(weights, axis=-1)
#             weights_bc = tf.broadcast_to(weights_bc, (tf.shape(weights_bc)[0],
#                                                       tf.shape(weights_bc)[1],
#                                                       tf.shape(outs)[-1]
#                                                       ))  # (B,num_anchors, E)
#             outs = tf.expand_dims(outs, axis=-2)  # (B, num_anchors, 1, E)
#             outs_w = tf.matmul(outs, self.w)  # (B,num_anchors,1,E)*(1,num_anchors,E,E)->(B,num_anchors,1,E)
#             outs_g = tf.nn.sigmoid(tf.matmul(outs, self.g))
#             outs_w = tf.squeeze(outs_w, axis=-2)  # (B,num_anchors,E)
#             outs_g = tf.squeeze(outs_g, axis=-2)
#             # multiply by attention weights from barcode_att to choose best anchors and additional feature gating
#             outs = outs_w * outs_g * weights_bc  # (B, num_anchors, E)
#         outs = self.ln(outs)
#         # outs -> anchor info, inds -> anchor indeces, weights -> anchor att weights, barcode_out -> whole peptide features
#         # (B,num_anchors,E), (B,num_anchors), (B,num_anchors), (B,E)
#         if self.return_att_weights:
#             return outs, inds, weights, tf.squeeze(barcode_out, axis=1), barcode_att
#         else:
#             return outs, inds, weights, tf.squeeze(barcode_out, axis=1)
#
#     def find_anchor(self, input, barcode_att):  # (B,N,E), (B,1,N)
#         inds = tf.argsort(barcode_att, axis=-1, direction='DESCENDING', stable=False)  # (B,1,N)
#         inds = tf.squeeze(inds, axis=1)  # (B,N)
#         selected_inds = select_indices(inds, n=self.num_anchors, m_range=self.dist_thr)  # (B,num_anchors)
#         sorted_selected_inds = tf.sort(selected_inds)
#         sorted_selected_weights = tf.gather(tf.squeeze(barcode_att, axis=1),
#                                             sorted_selected_inds,
#                                             axis=1,
#                                             batch_dims=1)  # (B,num_anchors)
#         sorted_selected_output = tf.gather(input, sorted_selected_inds, axis=1, batch_dims=1)  # (B,num_anchors,E)
#         return sorted_selected_inds, sorted_selected_weights, sorted_selected_output


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
    def __init__(self, num_anchors, dist_thr, initial_temperature=1.0, name='anchor_extractor', project=True,
                 mask_token=-1., pad_token=-2., return_att_weights=False):
        super().__init__()
        assert isinstance(dist_thr, list) and len(dist_thr) == 2
        assert num_anchors > 0
        self.num_anchors = num_anchors
        self.dist_thr = dist_thr
        self.name = name
        self.initial_temperature = initial_temperature
        self.project = project
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.return_att_weights = return_att_weights

    def build(self, input_shape):  # att_out (B,N,E)
        b, n, e = input_shape[0], input_shape[1], input_shape[2]
        self.barcode = tf.random.uniform(shape=(1, 1, e))  # add as a token to input
        self.q = self.add_weight(shape=(e, e),
                                 initializer='orthogonal',
                                 trainable=True, name=f'query_{self.name}')
        self.k = self.add_weight(shape=(e, e),
                                 initializer='orthogonal',
                                 trainable=True, name=f'key_{self.name}')
        self.v = self.add_weight(shape=(e, e),
                                 initializer='orthogonal',
                                 trainable=True, name=f'value_{self.name}')
        self.ln = layers.LayerNormalization(name=f'ln_{self.name}')
        if self.project:
            self.g = self.add_weight(shape=(self.num_anchors, e, e),
                                     initializer='random_uniform',
                                     trainable=True, name=f'gate_{self.name}')
            self.w = self.add_weight(shape=(1, self.num_anchors, e, e),
                                     initializer='random_normal',
                                     trainable=True, name=f'w_{self.name}')
        self.log_temperature = self.add_weight(
            shape=(),  # Scalar variable
            initializer=tf.keras.initializers.Constant(tf.math.log(self.initial_temperature)),
            trainable=True,
            name='log_temperature'
        )

    def call(self, input, mask):  # (B,N,E) this is peptide embedding and (B,N) for mask

        mask = tf.cast(mask, tf.float32)  # (B, N)
        mask = tf.where(mask == self.pad_token, 0., 1.)

        barcode = self.barcode
        barcode = tf.broadcast_to(barcode, (tf.shape(input)[0], 1, tf.shape(input)[-1]))  # (B,1,E)
        q = tf.matmul(barcode, self.q)  # (B,1,E)*(E,E)->(B,1,E)
        k = tf.matmul(input, self.k)  # (B,N,E)*(E,E)->(B,N,E)
        v = tf.matmul(input, self.v)  # (B,N,E)*(E,E)->(B,N,E)
        scale = 1 / tf.math.sqrt(tf.cast(tf.shape(input)[-1], tf.float32))
        barcode_att = tf.matmul(q, k, transpose_b=True) * scale  # (B,1,E)*(B,E,N)->(B,1,N)
        # mask: (B,N) => (B,1,N)
        mask_exp = tf.expand_dims(mask, axis=1)
        additive_mask = (1.0 - mask_exp) * -1e9
        barcode_att += additive_mask
        temperature = tf.exp(self.log_temperature)
        barcode_att = tf.nn.softmax(barcode_att / temperature)
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

def masked_categorical_crossentropy(y_true_and_pred, mask, pad_token=-2.0, sample_weight=None, type='cce'):
    """
    Compute masked categorical cross-entropy loss.

    Args:
        y_true_and_pred: Concatenated tensor of true labels and predictions.
        mask: Mask tensor indicating positions to include in the loss.
        pad_token: Value of the padding token in the mask.
        sample_weight: Optional tensor of shape (B, 1) or (B,) to weight samples.

    Returns:
        Mean masked loss (tensor).
    """
    assert type in ['cce', 'mse']
    y_true, y_pred = tf.split(y_true_and_pred, num_or_size_splits=2, axis=-1)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.0)

    # Build a 0/1 mask for non-pad tokens
    mask = tf.cast(tf.not_equal(mask, pad_token), tf.float32)

    # If mask has an extra trailing dim of 1, squeeze it (static check only)
    if mask.shape.rank is not None and mask.shape.rank > 2 and mask.shape[-1] == 1:
        mask = tf.squeeze(mask, axis=-1)
    if type == 'cce':
        loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)  # (B, N)
    elif 'mse':
        loss = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    else:
        raise ValueError(f"Unsupported loss type: {type}")

    # Ensure shape compatibility with loss
    mask = tf.cast(tf.broadcast_to(mask, tf.shape(loss)), tf.float32)

    masked_loss = loss * mask

    # Apply sample weights if provided
    if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, tf.float32)
        if sample_weight.shape.rank == 2 and sample_weight.shape[1] == 1:
            sample_weight = tf.squeeze(sample_weight, axis=-1) # (B,)
        # Broadcast sample_weight from (B,) to (B, N) to match masked_loss
        masked_loss *= sample_weight[:, tf.newaxis]
        mask *= sample_weight[:, tf.newaxis]

    total_loss = tf.reduce_sum(masked_loss)
    total_weight = tf.reduce_sum(mask)
    return tf.math.divide_no_nan(total_loss, total_weight)


def split_y_true_y_pred(y_true_y_pred):
    """
    Split concatenated y_true and y_pred tensor into separate tensors.

    Args:
        y_true_y_pred: Tensor of shape (B, N, 2 * C) where C is the number of classes.

    Returns:
        Tuple of (y_true, y_pred) tensors.
    """
    y_true, y_pred = tf.split(y_true_y_pred, num_or_size_splits=2, axis=-1)
    return y_true, y_pred


class MaskedCategoricalCrossentropy(tf.keras.losses.Loss):
    """
    Cross‑entropy that **ignores padded tokens via `sample_weight`.**

    Expected shapes
    ----------------
    y_pred : (B, N, 21)  – logits / probabilities for 21 classes
    y_true : (B, N, 21)  – one‑hot labels
    sample_weight : (B, N) – 1 for real token, 0 for pad (passed by Keras)
    """

    def __init__(self, *, pad_token=-2., masktoken=-.1, from_logits: bool = True, name: str = "masked_categorical_crossentropy"):
        # We let Keras do the masking & **reduction**; so we pick SUM_OVER_BATCH_SIZE.
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self._from_logits = from_logits
        self.pad_token = pad_token
        self.mask_token = masktoken

def call(self, y_true, y_pred):
    # Compute per-token cross-entropy
    ce = tf.keras.losses.categorical_crossentropy(
        y_true,
        y_pred,
        from_logits=self._from_logits,
        axis=-1,
    )
    # Mask padded tokens (tokens with zero label vectors)
    mask = tf.cast(tf.reduce_sum(y_true, axis=-1) > 0, ce.dtype)
    # Apply mask and average over valid tokens
    return tf.reduce_sum(ce * mask) / (tf.reduce_sum(mask) + 1e-8)


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

class SelfAttentionWith2DMask(keras.layers.Layer):
    """
    Custom self-attention layer that supports 2D masks.
    """
    def __init__(self, query_dim, context_dim, output_dim, heads=4,
                 return_att_weights=False, name='SelfAttentionWith2DMask',
                 epsilon=1e-6, mask_token=-1., pad_token=-2., self_attn_mhc=True, apply_rope=True):
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
        self.self_attn_mhc = self_attn_mhc
        self.apply_rope = apply_rope # flag for rotary positional embedding

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

        if self.apply_rope:
            if (self.att_dim % 2) != 0:
                raise ValueError(f"RotaryEmbedding requires even att_dim, got {self.att_dim}.")
            # q/k have shape (B, H, S, D): sequence_axis=2, feature_axis=-1
            self.rope = RotaryEmbedding(sequence_axis=2, feature_axis=-1, name=f'rope_{self.name}')

    def call(self, x_pmhc, p_mask, m_mask):
        """
        Args:
            x: Tensor of shape (B, N+M, query_dim) for query.
            p_mask: Tensor of shape (B, N) for peptide mask.
            m_mask: Tensor of shape (B, M) for mhc mask.
        Returns:
            Tensor of shape (B, N+M, output_dim)
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

        if self.apply_rope:
            q = self.rope(q)
            k = self.rope(k)

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
        if self.self_attn_mhc:
            self_mhc_mask = m_mask
        else:
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
        final_mask_t = tf.transpose(final_mask, perm=[0,2,1]) # (B,... same)
        final_mask = tf.multiply(final_mask, final_mask_t)
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


class SamplingMuMean(keras.layers.Layer):
    '''
    Reparameterization Trick
    '''
    def __init__(self, latent_dim, **kwargs):
        super(SamplingMuMean, self).__init__(**kwargs)
        self.latent_dim = latent_dim

    def call(self, inputs):
        '''
        Args:
            inputs - A tuple containing (mean, variance)
            output - A vector of shape (batch_size, latent_dim)
        '''
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        sample = tf.random.normal([batch, self.latent_dim]) * tf.exp(log_var / 2) + mean

        return sample


class SubtractLayer(keras.layers.Layer):
    """
    Custom layer to subtract a tensor from another tensor.
    Tensor1: (B, P, D) -> (B, P*D) -> (B, M, P*D)
    Tensor2: (B, M, D) -> (B, M, P*D)
    Output: = Tensor2 - Tensor1
    """
    def __init__(self, mask_token=-1., pad_token=-2., **kwargs):
        """Initialize the layer."""
        super(SubtractLayer, self).__init__(**kwargs)
        self.mask_token = mask_token
        self.pad_token = pad_token

    def call(self, peptide, pep_mask, mhc, mhc_mask):
        B = tf.shape(peptide)[0]
        P = tf.shape(peptide)[1]
        D = tf.shape(peptide)[2]
        M = tf.shape(mhc)[1]
        P_D = P * D

        pep_mask = tf.cast(pep_mask, tf.float32)
        mhc_mask = tf.cast(mhc_mask, tf.float32)

        pep_mask = tf.where(pep_mask == self.pad_token, x=0., y=1.)  # (B,P)
        mhc_mask = tf.where(mhc_mask == self.pad_token, x=0., y=1.)

        # peptide  (B,P,D) -> (B,P*D) -> (B,M,P*D)
        peptide_flat = tf.reshape(peptide, (B, P_D))
        peptide_exp = tf.repeat(peptide_flat[:, tf.newaxis, :], repeats=M, axis=1)
        # mhc       (B,M,D) -> tile last axis P times -> (B,M,P*D)
        mhc_exp = tf.tile(mhc, [1, 1, P])
        result = mhc_exp - peptide_exp  # (B,M,P*D)
        # peptide mask  (B,P) -> (B,P,D) -> flatten -> (B,P*D) -> (B,M,P*D)
        pep_mask_PD = tf.tile(pep_mask[:, :, tf.newaxis], [1, 1, D])  # (B,P,D)
        pep_mask_PD = tf.reshape(pep_mask_PD, (B, P_D))  # (B,P*D)
        pep_mask_PD = tf.repeat(pep_mask_PD[:, tf.newaxis, :], repeats=M, axis=1)  # (B,M,P*D)
        # mhc mask      (B,M) -> (B,M,1) -> repeat P*D along last axis
        mhc_mask_PD = tf.repeat(mhc_mask[:, :, tf.newaxis], repeats=P_D, axis=2)  # (B,M,P*D)
        combined_mask = tf.logical_and(tf.cast(pep_mask_PD, tf.bool), tf.cast(mhc_mask_PD, tf.bool))
        masked_result = tf.where(combined_mask, result, tf.zeros_like(result))
        return masked_result


def get_embed_key(key: str, emb_dict: Dict[str, str]) -> str: # why ndarray?
    """
    Get the embedding key for a given allele key.
    If the key is not found in the embedding dictionary, return None.
    # find the matching emb key in the emb_dict.
    Sometimes the emb key is longer than the allele key, so we need to check if the key is a substring of the emb key.
    """
    # Use a generator expression for efficient lookup
    return next((emb_key for emb_key in emb_dict if emb_key.upper().startswith(key.upper())), None)


def clean_key(allele_key: str) -> str:
    """
    Clean allele keys by removing special characters and converting to uppercase.
    This is useful for matching keys in embedding dictionaries.
    """
    if allele_key is None:
        return "None"
    mapping = str.maketrans({'*': '', ':': '', ' ': '', '/': '_'})
    return allele_key.translate(mapping).upper()


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


class GlobalMeanPooling1D(layers.Layer):
    """
    Global mean pooling layer.
    Computes the mean across the last axis (features).
    """
    def __init__(self,axis=-1, name="global_mean_pooling_"):
        super(GlobalMeanPooling1D, self).__init__(name=name)
        self.name = name
        self.axis = axis

    def call(self, input_tensor, ):
        """
        Computes the global mean pooling over the input tensor.
        :param input_tensor:
        :return:
        """
        # inputs: (B, N, D)
        mean = tf.math.reduce_mean(
            input_tensor, axis=self.axis, keepdims=False
        )
        return mean  # (B, D)

class GlobalSTDPooling1D(layers.Layer):
    """
    Global Standard Deviation Pooling layer that computes the standard deviation
    across the sequence dimension for each feature.
    Args:
        name (str): Layer name.
    """
    def __init__(self, axis=1, name='global_std_pooling'):
        super(GlobalSTDPooling1D, self).__init__(name=name)
        self.axis = axis

    def call(self, input_tensor, ):
        """
        Args:
            inputs: Input tensor of shape (B, N, D).
        Returns:
            Tensor of shape (B, N) containing the standard deviation across D.
        """
        pooled_std = tf.math.reduce_std(
            input_tensor, axis=self.axis, keepdims=False, name=None
            )
        return pooled_std


class GumbelSoftmax(keras.layers.Layer):
    """
    Gumbel-Softmax activation layer.

    Args:
        temperature (float): Temperature parameter for Gumbel-Softmax.
    """
    def __init__(self, temperature=0.2, name="gumble_softmax_layer"):
        super(GumbelSoftmax, self).__init__(name=name)
        self.temperature = temperature

    def call(self, logits, training=None):
        """
        Applies Gumbel-Softmax.

        Args:
            logits: Input tensor of shape (B, N).
            training: Whether the layer is in training mode (not used here but required by Layer API).

        Returns:
            Tensor of shape (B, N) with Gumbel-Softmax applied.
        """
        # Sample Gumbel noise
        # Use tf.random.uniform for TensorFlow compatibility
        U = tf.random.uniform(tf.shape(logits), minval=0, maxval=1)
        gumbel_noise = -tf.math.log(-tf.math.log(U + 1e-20) + 1e-20) # Add small epsilon for numerical stability

        # Apply Gumbel-Softmax formula
        y = tf.exp((logits + gumbel_noise) / self.temperature)
        return y / tf.reduce_sum(y, axis=-1, keepdims=True)

# Reduced alphabet mapping based on MMseqs2
mmseqs2_reduced_alphabet = {
    "L": ["L", "M"],
    "I": ["I", "V"],
    "K": ["K", "R"],
    "E": ["E", "Q"],
    "A": ["A", "S", "T"],
    "N": ["N", "D"],
    "F": ["F", "Y"],
    "G": ["G"],
    "H": ["H"],
    "C": ["C"],
    "P": ["P"],
    "W": ["W"],
}

mmseqs2_reduced_alphabet_rev = {
    "A": "A",
    "C": "C",
    "D": "N",
    "E": "E",
    "F": "F",
    "G": "G",
    "H": "H",
    "I": "I",
    "K": "K",
    "L": "L",
    "M": "L",
    "N": "N",
    "P": "P",
    "Q": "E",
    "R": "K",
    "S": "A",
    "T": "A",
    "V": "I",
    "W": "W",
    "Y": "F",
}

def reduced_anchor_pair(peptide_seq: str) -> str:
    """ Gets the second and last amino acid, maps them to the MMseqs2 reduced alphabet, and returns them as a pair string. """
    # remove gaps
    peptide_seq = peptide_seq.replace("-", "").replace("*", "").replace(" ", "").replace("X", "").upper()
    # print(peptide_seq)
    if len(peptide_seq) < 2:
        return "Short"
    p2 = peptide_seq[1].upper() # second amino acid
    p_omega = peptide_seq[-1].upper() # last amino acid

    # Map to reduced alphabet, handle missing keys gracefully
    p2_reduced = mmseqs2_reduced_alphabet_rev.get(p2, 'X')
    # print(f"p2: {p2}, reduced: {p2_reduced}")
    p_omega_reduced = mmseqs2_reduced_alphabet_rev.get(p_omega, 'X')
    # print(f"p_omega: {p_omega}, reduced: {p_omega_reduced}")

    return f"{p2_reduced}-{p_omega_reduced}" # e.g. "A-S" should return 144 combinations

 # --- Helper function and data for physiochemical properties ---
from collections import Counter
from typing import Dict

from Bio.SeqUtils.ProtParam import ProteinAnalysis

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
POLAR_UNCHARGED = set("STNQCY")
CHARGED = set("KRHDE")
POLAR_SET = POLAR_UNCHARGED | CHARGED

def peptide_properties_biopython(seq: str, pH: float = 7.4) -> Dict[str, float]:
    """
    Compute peptide properties using Biopython's ProtParam:
      - Kyte–Doolittle average hydrophobicity (GRAVY)
      - Net charge at given pH (includes termini)
      - Fraction of polar residues (charged + polar uncharged)
      - Isoelectric point (pI)
      - Aromaticity
      - Instability index
      - Molecular weight
      - Extinction coefficients (reduced and with cystines)
    Unknowns/gaps (e.g., X, - , *) are removed before analysis.
    """
    if not seq:
        return {
            "original_length": 0,
            "analyzed_length": 0,
            "hydrophobicity": 0.0,
            "charge": 0.0,
            "fraction_polar": 0.0,
            "pI": 0.0,
            "aromaticity": 0.0,
            "instability_index": 0.0,
            "molecular_weight": 0.0,
            "extinction_coeff_reduced": 0,
            "extinction_coeff_cystine": 0,
        }

    seq = seq.upper()
    cleaned = "".join(aa for aa in seq if aa in VALID_AA)
    L = len(cleaned)

    if L == 0:
        return {
            "original_length": len(seq),
            "analyzed_length": 0,
            "hydrophobicity": 0.0,
            "charge": 0.0,
            "fraction_polar": 0.0,
            "pI": 0.0,
            "aromaticity": 0.0,
            "instability_index": 0.0,
            "molecular_weight": 0.0,
            "extinction_coeff_reduced": 0,
            "extinction_coeff_cystine": 0,
        }

    pa = ProteinAnalysis(cleaned)
    counts = Counter(cleaned)

    gravy = pa.gravy()  # Kyte–Doolittle average hydropathy
    charge = pa.charge_at_pH(pH)  # includes side chains + N/C termini
    frac_polar = sum(n for aa, n in counts.items() if aa in POLAR_SET) / L

    pI = pa.isoelectric_point()
    aromaticity = pa.aromaticity()
    instability = pa.instability_index()
    mw = pa.molecular_weight()
    # Returns (reduced, with cystines)
    ext_reduced, ext_cystine = pa.molar_extinction_coefficient()

    return {
        "original_length": len(seq),
        "analyzed_length": L,
        "hydrophobicity": gravy,
        "charge": charge,
        "fraction_polar": frac_polar,
        "pI": pI,
        "aromaticity": aromaticity,
        "instability_index": instability,
        "molecular_weight": mw,
        "extinction_coeff_reduced": ext_reduced,
        "extinction_coeff_cystine": ext_cystine,
    }


def cn_terminal_amino_acids(peptide: str, n_term_len: int = 3, c_term_len: int = 3) -> dict:
    """
    Analyzes a peptide sequence to determine the dominant chemical property
    (polar, hydrophobic, or charged) of its N-terminal, C-terminal, and core regions.

    Args:
        peptide (str): The amino acid sequence of the peptide.
        n_term_len (int): The number of amino acids to consider for the N-terminus.
        c_term_len (int): The number of amino acids to consider for the C-terminus.

    Returns:
        dict: A dictionary with keys 'N-term', 'Core', and 'C-term', where the values
              are strings indicating the region and its dominant property
              (e.g., 'N-term_polar').
    """
    # --- Define Amino Acid Properties ---
    # Using sets for efficient membership checking
    polar_aa = {'S', 'T', 'C', 'N', 'Q', 'Y'}
    hydrophobic_aa = {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'P', 'G'}
    # Both acidic (-) and basic (+) are considered 'charged'
    charged_aa = {'D', 'E', 'R', 'K', 'H'}

    def get_property(amino_acid: str) -> str:
        """Helper function to classify a single amino acid."""
        if amino_acid in polar_aa:
            return 'polar'
        if amino_acid in hydrophobic_aa:
            return 'hydrophobic'
        if amino_acid in charged_aa:
            return 'charged'
        return 'unknown'  # Should not happen with standard peptides

    def get_dominant_property(sequence: str) -> str:
        """
        Helper function to find the most common property in a sequence.
        In case of a tie, the priority is hydrophobic > polar > charged.
        """
        if not sequence:
            return 'none'  # Handle empty core sequences

        properties = [get_property(aa) for aa in sequence]

        # Count occurrences of each property
        counts = Counter(properties)

        # Determine the dominant property with a tie-breaking rule
        # The key for sorting is a tuple of the count and a priority value
        # Higher count is better, lower priority value is better for ties
        priority = {'hydrophobic': 0, 'polar': 1, 'charged': 2, 'none': 3, 'unknown': 4}

        # Find the property with the maximum count, using priority to break ties
        dominant_prop = max(counts, key=lambda prop: (counts[prop], -priority[prop]))

        return dominant_prop

    # --- Define Peptide Regions ---
    # Ensure the function handles peptides shorter than the defined terminal lengths
    actual_n_len = min(n_term_len, len(peptide))
    actual_c_len = min(c_term_len, len(peptide))

    n_terminal_seq = peptide[:actual_n_len]
    c_terminal_seq = peptide[-actual_c_len:]

    # The core is what's left in the middle
    # Handle cases where terminals overlap or there is no core
    if len(peptide) > actual_n_len + actual_c_len:
        core_seq = peptide[actual_n_len:-actual_c_len]
    else:
        core_seq = ""  # No distinct core

    # --- Classify Each Region ---
    n_term_prop = get_dominant_property(n_terminal_seq)
    c_term_prop = get_dominant_property(c_terminal_seq)
    core_prop = get_dominant_property(core_seq)

    # --- Format Output ---
    # This format matches the keys in your `segment_color_map`
    result = {
        'N-term': f'N-term_{n_term_prop}',
        'Core': f'Core_{core_prop}',
        'C-term': f'C-term_{c_term_prop}'
    }

    return result


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent vector."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        # KL Divergence Regularization Loss
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        return z, kl_loss


def create_k_fold_leave_one_out_stratified_cv(
    df: pd.DataFrame,
    k: int = 5,
    target_col: str = "label",
    id_col: str = "allele",
    subset_prop: float = 1.0,
    train_size: float = 0.8,
    random_state: int = 42,
    n_val_ids: int = 1,
    augmentation: str = None  # "down_sampling" or "GNUSS"
):
    """
    Build *k* folds such that

    1. **One whole ID (group) is left out of both train & val** (`left_out_id`).
    2. **Validation contains exactly one additional ID** (`val_only_id`)
       that never appears in train.
    3. Remaining rows are split *stratified* on `target_col`
       (`train_size` fraction for training).
    4. Train & val are **down-sampled** to perfectly balanced label counts.

    Returns
    -------
    list[tuple[pd.DataFrame, pd.DataFrame, Hashable]]
        Each tuple = (train_df, val_df, left_out_id).
    """
    rng = np.random.RandomState(random_state)
    if subset_prop < 1.0:
        if subset_prop <= 0.0 or subset_prop > 1.0:
            raise ValueError(f"subset_prop must be in (0, 1], got {subset_prop}")
        # Take a random subset of the DataFrame
        print(f"Taking {subset_prop * 100:.2f}% of the data for k-fold CV")
        df = df.sample(frac=subset_prop, random_state=random_state).reset_index(drop=True)

    # --- pick the k IDs that will be held out completely -------------------
    unique_ids = df[id_col].unique()
    if k > len(unique_ids):
        raise ValueError(f"k={k} > unique {id_col} count ({len(unique_ids)})")
    left_out_ids = rng.choice(unique_ids, size=k, replace=False)

    folds = []
    for fold_idx, left_out_id in enumerate(left_out_ids, 1):
        fold_seed = random_state + fold_idx
        mask_left_out = df[id_col] == left_out_id
        working_df = df.loc[~mask_left_out].copy()
        available_ids = working_df[id_col].unique()
        if len(available_ids) < n_val_ids:
            raise ValueError(f"Not enough unique IDs ({len(available_ids)}) to select 10 for validation")

        # # ---------------------------------------------------------------
        # # 1) choose ONE id that will appear *only* in validation
        # #    (GroupShuffleSplit with test_size=1 group)
        # # ---------------------------------------------------------------
        # gss = GroupShuffleSplit(
        #     n_splits=1, test_size=1.0, random_state=fold_seed
        # )
        # (train_groups_idx, val_only_groups_idx), = gss.split(
        #     X=np.zeros(len(working_df)), y=None, groups=working_df[id_col]
        # )

        val_only_group_ids = rng.choice(available_ids, size=n_val_ids, replace=False)

        # Ensure we get all rows for the selected validation groups
        mask_val_only = working_df[id_col].isin(val_only_group_ids)
        df_val_only = working_df[mask_val_only].copy()
        df_eligible = working_df[~mask_val_only].copy()

        # Verify the selection worked correctly
        actual_val_ids = df_val_only[id_col].unique()
        if len(actual_val_ids) != n_val_ids:
            print(f"Warning: Expected {n_val_ids} validation IDs, got {len(actual_val_ids)}")

        # ---------------------------------------------------------------
        # 2) stratified split of *eligible* rows
        # ---------------------------------------------------------------
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=train_size, random_state=fold_seed
        )
        train_idx, extra_val_idx = next(
            sss.split(df_eligible, df_eligible[target_col])
        )
        df_train = df_eligible.iloc[train_idx]
        # df_val   = pd.concat(
        #     [df_val_only, df_eligible.iloc[extra_val_idx]], ignore_index=True
        # )
        df_val = df_val_only

        print(f"Fold size: train={len(df_train)}, val={len(df_val)} | ")

        # ---------------------------------------------------------------
        # 3) balance train and val via down-sampling
        # ---------------------------------------------------------------
        def _balance_down_sampling(frame: pd.DataFrame) -> pd.DataFrame:
            min_count = frame[target_col].value_counts().min()
            print(f"Balancing {len(frame)} rows to {min_count} per class")
            balanced_parts = [
                resample(
                    frame[frame[target_col] == lbl],
                    replace=False,
                    n_samples=min_count,
                    random_state=fold_seed,
                )
                for lbl in frame[target_col].unique()
            ]
            return pd.concat(balanced_parts, ignore_index=True)

        def _balance_GNUSS(frame: pd.DataFrame) -> pd.DataFrame:
            """
            Balance the DataFrame by upsampling the minority class with Gaussian noise.
            """
            # Determine label counts and the maximum class size
            counts = frame[target_col].value_counts()
            max_count = counts.max()

            # Identify numeric columns for noise injection
            numeric_cols = frame.select_dtypes(include="number").columns

            balanced_parts = []
            for label, count in counts.items():
                df_label = frame[frame[target_col] == label]
                balanced_parts.append(df_label)
                if count < max_count:
                    # Upsample with replacement
                    n_needed = max_count - count
                    sampled = df_label.sample(n=n_needed, replace=True, random_state=fold_seed)
                    # Add Gaussian noise to numeric features
                    noise = pd.DataFrame(
                        rng.normal(loc=0, scale=1e-6, size=(n_needed, len(numeric_cols))),
                        columns=numeric_cols,
                        index=sampled.index
                    )
                    sampled[numeric_cols] = sampled[numeric_cols] + noise
                    balanced_parts.append(sampled)

            # Combine and return
            return pd.concat(balanced_parts).reset_index(drop=True)

        if augmentation == "GNUSS":
            df_train_bal = _balance_GNUSS(df_train)
            df_val_bal   = _balance_GNUSS(df_val)
        elif augmentation == "down_sampling":  # default to down-sampling
            df_train_bal = _balance_down_sampling(df_train)
            df_val_bal   = _balance_down_sampling(df_val)
        elif not augmentation:
            df_train_bal = df_train.copy()
            df_val_bal = df_val.copy()
            print("No augmentation applied, using original train and val sets.")
        else:
            raise ValueError(f"Unknown augmentation method: {augmentation}")

        # Shuffle both datasets to avoid any ordering bias
        df_train_bal = df_train_bal.sample(frac=1.0, random_state=fold_seed).reset_index(drop=True)
        df_val_bal = df_val_bal.sample(frac=1.0, random_state=fold_seed).reset_index(drop=True)
        folds.append((df_train_bal, df_val_bal, left_out_id))

        print(
            f"[fold {fold_idx}/{k}] left-out={left_out_id} | "
            f"val-only={val_only_group_ids} | "
            f"train={len(df_train_bal)}, val={len(df_val_bal)}"
        )

    return folds

# # --- Example Usage ---
# if __name__ == '__main__':
#     # Create a dummy DataFrame similar to your use case
#     data = {'long_mer': ['SKIVYWQPL', 'GLACGTGVN', 'RGYVYQGL', 'LLFGYPVYV']}
#     df = pd.DataFrame(data)
#
#     # Apply the function to the DataFrame column
#     # The .apply(pd.Series) part expands the dictionary output into separate columns
#     cn_peptide_df = df['long_mer'].apply(cn_terminal_amino_acids).apply(pd.Series)
#
#     print("--- Original Peptides ---")
#     print(df)
#     print("\n--- Classified Peptide Regions ---")
#     print(cn_peptide_df)
#
#     # Example 1: SKIVYWQPL
#     # N-term: SKI -> charged, polar, hydrophobic -> tie, resolved to hydrophobic
#     # C-term: QPL -> polar, hydrophobic, hydrophobic -> hydrophobic
#     # Core: VYW -> hydrophobic, polar, polar -> polar
#     # Expected: N-term_hydrophobic, Core_polar, C-term_hydrophobic
#     print("\n--- Detailed check for 'SKIVYWQPL' ---")
#     print(cn_terminal_amino_acids('SKIVYWQPL'))

# Usage example
#
# df_analysis = pd.read_csv("analysis_dataset_with_clusters.csv")
#