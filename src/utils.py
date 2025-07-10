import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Constants
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {a: i for i, a in enumerate(AA)}
UNK_IDX = 20  # Index for "unknown"
MASK_TOKEN = -1.0
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
    num_unks = np.sum(arr[:, UNK_IDX])
    if num_unks > 0:
        print(f"Warning: {num_unks} unknown amino acids in sequence '{sequence}'")
    return arr


def OHE_to_seq(ohe: np.ndarray) -> list:
    """
    Convert a one-hot encoded matrix back to a peptide sequence.
    # (B, max_pep_len, 21) -> (B, max_pep_len)
    Args:
        ohe: One-hot encoded matrix of shape (B, N, 21).
    Returns:
        sequence: Peptide sequence as a string. (B,)
    """
    AA = "ACDEFGHIKLMNPQRSTVWY"
    sequence = []
    for i in range(ohe.shape[0]):  # Iterate over batch dimension
        seq = []
        for j in range(ohe.shape[1]):  # Iterate over sequence length
            aa_index = np.argmax(ohe[i, j])  # Get index of the max value in one-hot encoding
            if aa_index < len(AA):  # Check if it's a valid amino acid index
                seq.append(AA[aa_index])
            else:
                seq.append('X')  # Use 'X' for unknown amino acids
        sequence.append(''.join(seq))  # Join the list into a string
    return sequence  # Return list of sequences





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
        angle_rates = 1 / tf.pow(300.0, (2 * (i // 2)) / tf.cast(self.embed_dim, tf.float32))
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