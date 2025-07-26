# visualize tensorflow model

import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from utils import AttentionLayer, PositionalEncoding, AnchorPositionExtractor, SplitLayer,ConcatMask,ConcatBarcode, MaskedEmbedding
import uuid
import matplotlib.pyplot as plt
import seaborn as sns





def visualize_tf_model(model_path='h5/bicross_encoder_decoder.h5'):
    """
    Visualizes the TensorFlow model architecture and saves it as an image.
    :return:
        None
    """

    def wrap_layer(layer_class):
        def fn(**config):
            config.pop('trainable', None)
            config.pop('dtype', None)
            # assign a unique name to avoid duplicates
            config['name'] = f"{layer_class.__name__.lower()}_{uuid.uuid4().hex[:8]}"
            return layer_class.from_config(config)

        return fn
    # Load the model with wrapped custom objects
    model = load_model(
            model_path,
        custom_objects={
            'AttentionLayer': wrap_layer(AttentionLayer),
            'PositionalEncoding': wrap_layer(PositionalEncoding),
            'AnchorPositionExtractor': wrap_layer(AnchorPositionExtractor),
            'SplitLayer': wrap_layer(SplitLayer),
            'ConcatMask': wrap_layer(ConcatMask),
            'ConcatBarcode': wrap_layer(ConcatBarcode),
            'MaskedEmbedding': wrap_layer(MaskedEmbedding),
        }
    )

    # Display model summary
    model.summary()

    # Create better visualization as SVG with cleaner layout
    tf.keras.utils.plot_model(
        model,
        to_file='h5/model_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',  # Top to bottom layout
        dpi=200,       # Higher resolution
        expand_nested=True,  # Expand nested models to show all layers
        show_layer_activations=True  # Show activation functions
    )


def visualize_cross_attention_weights(cross_attn_scores, peptide_seq, mhc_seq):
    """
    Visualize cross-attention weights between peptide and MHC sequences.

    Args:
        cross_attn_scores: Tensor of shape (B, N_peptide, N_mhc) with attention scores.
        peptide_seq: List of peptide sequences.
        mhc_seq: List of MHC sequences.
        top_n: Number of top attention scores to visualize.

    Returns:
        None
    """

    # Convert to numpy for visualization
    try:
        cross_attn_scores = cross_attn_scores.numpy()
    except:
        cross_attn_scores = cross_attn_scores

    cross_attn_scores = cross_attn_scores.mean(axis=0)  # Average over heads dimension
    plt.figure(figsize=(12, 8))
    if cross_attn_scores.shape[0] == cross_attn_scores.shape[1]:
        # If square matrix, use heatmap
        sns.heatmap(cross_attn_scores, annot=True, fmt=".2f",
                    yticklabels=peptide_seq, xticklabels=peptide_seq,
                    cmap='viridis', cbar_kws={'label': 'Attention Score'})
    else:
        sns.heatmap(cross_attn_scores, annot=True, fmt=".2f",
                    yticklabels=mhc_seq, xticklabels=peptide_seq,
                    cmap='viridis', cbar_kws={'label': 'Attention Score'})
    plt.title(f'Cross-Attention Weights for Sample')
    plt.xlabel('Sequence')
    plt.ylabel('Sequence')
    plt.show()

# if __name__ == "__main__":
#     visualize_tf_model()
#     print("Model visualization saved as 'h5/model_architecture_decoder.png'")
import numpy as np
import matplotlib.pyplot as plt


def plot_1d_heatmap(data, cmap='viridis', title='1D Heatmap', xlabel='', ylabel=''):
    """
    Plots a heatmap of a 1D array by expanding it to 2D (N, 1).

    Parameters:
        data (array-like): 1D input array of shape (N,)
        cmap (str): Matplotlib colormap
        title (str): Plot title
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
    """
    data = np.array(data).reshape(-1, 1)  # Shape (N,) → (N, 1)

    plt.figure(figsize=(10, 7))  # Adjust figure size based on N
    plt.imshow(data, aspect='auto', cmap=cmap)
    plt.colorbar(label='Value')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks([0])  # Only one column
    plt.yticks(np.arange(len(data)))
    plt.tight_layout()
    plt.show()


