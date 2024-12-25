import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization
from network.attention import ScaledMultiHeadAttention
from network.ffn import FFN

class Decoder(Layer):
    """
    Args:
        embedding_dim: words embedding dimension
        n_heads: Scaled multihead self attention layer according to embedding dim and number of heads
    
    Attributes:
        mmha: masked scaled multihead attention layer
        norm1: normalization layer after mmha and residual connection
        mha: scaled multihead attention layer - CROSS ATTENTION
        norm2: normalization layer after cross attention and 
        ffn: 2 Dense layers for emebddings nonlinearity
        norm2: normalization layer after ffn and residual connection

              -----               -----     -----      -----     -----      -----
    input -> | mmha | + input -> |norm1| + | mha | -> |norm2| + | ffn | -> |norm3| -> rich cross embeddings
              -----               -----     -----      -----     -----      -----
    """

    def __init__(self, embedding_dim: int, n_heads: int=1) -> None:
        super(Decoder, self).__init__()
        self.mmha = ScaledMultiHeadAttention(n_heads, embedding_dim, mask=True)
        self.norm1 = LayerNormalization()
        self.mha = ScaledMultiHeadAttention(n_heads, embedding_dim)
        self.norm2 = LayerNormalization()
        self.ffn = FFN(embedding_dim)
        self.norm3 = LayerNormalization()


    def call(self, input_layer: tf.Tensor, K: tf.Tensor, V: tf.Tensor, pad_mask: tf.Tensor) -> tf.Tensor:
        """
        Args:
            input_layer: decoder input after positional encoding - Query matrix
            K: Key matrix
            V: Value matrix
            pad_mask: masking tensor of the inputs for padded tokens - to pass it to attention

        Returns:
            x: rich embeddings from the combination of source and destination inputs
        """
    
        x = self.mmha(input_layer, input_layer, input_layer, pad_mask)
        res2 = self.norm1(x + input_layer)
        x = self.mha(res2, K, V, pad_mask)
        res3 = self.norm2(x + res2)
        x = self.ffn(res3)
        x = self.norm3(x + res3)
        return x