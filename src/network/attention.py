import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax

INF_NUMBER = -1e9

class ScaledMultiHeadAttention():
    """
    Args:
        n_heads: number of heads for multihead attention
        embedding_dim: words embedding dimension
        mask: attention mask - used for decoder
    
    Attributes:
        n_heads: number of heads for multihead attention
        mask: attention mask for decoder. False by default
        embedding_dim: final embedding dimension which should be equal to the inputs embedding dimensions
        d_k: each head's embedding dimension which is relative to the number of heads and embedding_dim
        scale: denominator of the scaled dot product attention
        activation: activation function of the attention mechanism - Softmax for Luong
        att_weight: to control final output's dimensions (equal to the input dimensions)
        Q_enc, K_enc, V_enc: to add flexibility and control over Q,K,V at each head
                             pay carefull attention that the weights of each head should be independent.
                             The weights cannot be defined as a function in __call__ since the optimizer should be able to track weights.
    """

    def __init__(self, n_heads, embedding_dim, mask=False) -> None:
        self.n_heads = n_heads
        self.mask = mask
        self.embedding = embedding_dim
        self.d_k = embedding_dim // n_heads
        self.scale = embedding_dim**0.5
        self.activation = Softmax(axis=-1)
        self.att_weight = Dense(embedding_dim)
        self.Q_enc = [Dense(self.d_k) for _ in range(n_heads)]
        self.K_enc = [Dense(self.d_k) for _ in range(n_heads)]
        self.V_enc = [Dense(self.d_k) for _ in range(n_heads)] 

    def __call__(self, Q: tf.Tensor, K: tf.Tensor, V: tf.Tensor, pad_mask: tf.Tensor) -> tf.Tensor:
        """
        Calculates scaled multi head attention.

        Args:
            Q: Query matrix
            K: Key matrix
            V: Value matrix
            pad_mask: masking tensor for ignoring padded tokens in Value matrix
        
        Returns:
            W_att*concatenated_heads: concatenates heads and applies a Dense layer to convert results to the original embedding dimension
        
        """
        logit_matrix = None
        
        if self.mask == True:
            mask_matrix = np.zeros(((Q.shape)[1], (Q.shape)[1]))
            for i in range((Q.shape)[1]):
                mask_matrix[i, i+1:] = INF_NUMBER
            # tf.expand_dims(mask_matrix, axis=0)
            
        for i in range(self.n_heads):
            scaled_e_matrix = (self.Q_enc[i](Q) @ tf.transpose(self.K_enc[i](K), perm=(0, 2, 1))) / self.scale
            # don't consider batch since the Model and Keras handles it automatically

            scaled_e_matrix += (1.0 - pad_mask * tf.transpose(pad_mask, perm=[0, 2, 1])) * INF_NUMBER 
            #pad_masking in rows and columns (making sure broadcasting is applied on columns as well)

            if self.mask == True:
                scaled_e_matrix = scaled_e_matrix + mask_matrix
            
            # Softmax doesn't 0 out the padded masks in rows (it makes their sum to be 1). It only applies in columns -> *pad_mask
            temp = self.activation(scaled_e_matrix)*pad_mask @ self.V_enc[i](V)
           
            if logit_matrix is None:
                logit_matrix = temp
            else:
                logit_matrix = tf.concat([logit_matrix, temp], axis = -1) # (batchsize, seq_length, embedding_dim  -> concatenation should be applied on embedding_dim axis

        # calculate each head in smaller dimension separately and then concat or concat and then convert to smaller dimensions?
        # concat and then weight to make sure that the dimensionality preserves
        return self.att_weight(logit_matrix)

