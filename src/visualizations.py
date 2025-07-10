# visualize tensorflow model

import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from utils import AttentionLayer, PositionalEncoding, AnchorPositionExtractor, SplitLayer,ConcatMask,ConcatBarcode, MaskedEmbedding
import uuid





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

if __name__ == "__main__":
    visualize_tf_model()
    print("Model visualization saved as 'h5/model_architecture_decoder.png'")