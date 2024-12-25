import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization
from network.attention import ScaledMultiHeadAttention
from network.ffn import FFN


class Encoder(Layer):
    """
    Args:
        embedding_dim: words embedding dimension
        n_heads: number of heads for attention mechanism. Default value is 1
    
    Attributes:
        mha: Scaled multihead self attention layer according to embedding dim and number of heads
        norm1: normalization layer after residual and scaled multihead attention
        ffn: 2 Dense layers for emebddings nonlinearity
        norm2: normalization layer after ffn and residual connection
    """
    def __init__(self, embedding_dim: int, n_heads: int=1) -> None:
        super(Encoder, self).__init__()
        self.mha = ScaledMultiHeadAttention(n_heads, embedding_dim)
        self.norm1 = LayerNormalization() # norm should be different for each norm layer of the encoder and decoder
        self.ffn = FFN(embedding_dim)
        self.norm2 = LayerNormalization()


    def call(self, input_layer: tf.Tensor, pad_mask: tf.Tensor) -> tf.Tensor:
        """
        Args:
            input_layer: encoder input after positional encoding, representing Q, K, and V
            pad_mask: masking tensor of the inputs for padded tokens - to pass it to attention

        Returns:
            x: rich embeddings of the destination input tokens

                  -----              -----     -----      -----
        input -> | mha | + input -> |norm1| + | ffn | -> |norm2| -> rich_embeddings   
                  -----              -----     -----      -----
        
        """
        y = self.mha(input_layer, input_layer, input_layer, pad_mask)
        res2 = self.norm1(y + input_layer)
        x = self.ffn(res2)
        x = self.norm2(x + res2)
        return x
