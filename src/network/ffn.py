import tensorflow as tf

class FFN(tf.keras.layers.Layer):
    """
    Args:
        embedding_dim: words embedding dimension
    
    Attributes:
        embedding_dim
        node: an intermediate Dense layer
        out: final Dense layer
        
    """
    def __init__(self, embedding_dim: int) -> None:
        super(FFN, self).__init__()
        self.embedding_dim = embedding_dim
        self.node = tf.keras.layers.Dense(4*self.embedding_dim, activation='relu') #according to the paper it maps to 4*emb
        self.out = tf.keras.layers.Dense(self.embedding_dim)

    def call(self, x: tf.Tensor) -> tf.Tensor: # dense is applied to the last dimension automatically
        return self.out(self.node(x)) 