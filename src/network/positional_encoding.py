import math
import numpy as np
import tensorflow as tf

class PositionalEncoding():
    """
    Args:
        sequence_length: length of the input sequences
        embedding_dim: words embedding dimension
    
    Attributes:
        sequence_length: length of the sequences (rows)
        embedding_dim: size of each sequence (columns)
        embedding_matrix: positional encoding matrix according to the input specifications
    """
    def __init__(self, sequence_length, embedding_dim) -> None:
        #super(PositionalEncoding, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim // 2
        self.embedding_matrix = np.zeros((sequence_length, embedding_dim))
        
    def __call__(self) -> tf.Tensor:
        """
        Returns:
            sin/cos positional encoding matrix with one added dimension at axis=0 for batch_size (to be broadcasted when summed)
        """
        for pos in range(self.sequence_length):
            for i in range(self.embedding_dim):
                self.embedding_matrix[pos, 2*i] = math.sin( pos / ( math.pow(10000, (2*i)/self.embedding_dim) ) )
                self.embedding_matrix[pos, 2*i+1] = math.cos( pos / ( math.pow(10000, (2*i)/self.embedding_dim )))
        
        return tf.expand_dims(tf.convert_to_tensor(self.embedding_matrix, dtype=tf.float32), axis=0)
    