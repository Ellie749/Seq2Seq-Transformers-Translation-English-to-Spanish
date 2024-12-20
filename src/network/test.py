import os
os.environ["TF_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings('ignore')
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Embedding, LayerNormalization, Layer, Softmax
from tensorflow.keras import Model


class PositionalEncoding():
    def __init__(self, sequence_length, embedding_dim):
        #super(PositionalEncoding, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim // 2
        self.embedding_matrix = np.zeros((sequence_length, embedding_dim))
        
    def __call__(self):
        for pos in range(self.sequence_length):
            for i in range(self.embedding_dim):
                self.embedding_matrix[pos, 2*i] = math.sin( pos / ( math.pow(10000, (2*i)/self.embedding_dim) ) )
                self.embedding_matrix[pos, 2*i+1] = math.cos( pos / ( math.pow(10000, (2*i)/self.embedding_dim )))
        
        return tf.expand_dims(tf.convert_to_tensor(self.embedding_matrix, dtype=tf.float32), axis=0)
    

class ScaledMultiHeadAttention():
    def __init__(self, n_heads, embedding_dim, mask=False): #add different shapes for dk and dmodel and test
        #super(ScaledMultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.mask = mask
        self.embedding = embedding_dim
        self.scale = embedding_dim**0.5
        self.activation = Softmax()
        self.att_weight = Dense(embedding_dim)


    def __call__(self, Q, K, V):

        print(f"Q: {Q}")
        logit_matrix = None

        if self.mask == True:
            inf_number = -1e9
            mask_matrix = np.zeros(((Q.shape)[1], (Q.shape)[1]))
            for i in range((Q.shape)[1]):
                mask_matrix[i, i+1:] = inf_number
            # print(mask_matrix)
            
        def mlp_layer():
            Q_enc = Dense(self.embedding)
            K_enc = Dense(self.embedding)
            V_enc = Dense(self.embedding)
            return [Q_enc, K_enc, V_enc]

        for i in range(self.n_heads):
            weights = mlp_layer() # every time different weights are created
            scaled_e_matrix = (weights[0](Q) @ tf.transpose(weights[1](K), perm=(0, 1, 3, 2))) / self.scale

            if self.mask == True:
                scaled_e_matrix = scaled_e_matrix + mask_matrix

            temp = self.activation(scaled_e_matrix) @ weights[2](V)
            print(temp)
            if logit_matrix is None:
                logit_matrix = temp
            else:
                logit_matrix = tf.concat([logit_matrix, temp], axis = -1) # (batchsize, seq_length, embedding_dim  -> comcatenation should be applied on embedding_dim axis

        return self.att_weight(logit_matrix)




data_enc = tf.convert_to_tensor(np.array([[[2, 3, 4, 5, 0], [6, 7, 8, 9, 0]]]), dtype=tf.float32)
enc_embedding = Embedding(10, 8, mask_zero=True)(data_enc) # 1*2*5*8
pe = PositionalEncoding(5, 8) # 1*5*8
mha_input = pe() + enc_embedding # 1*2*5*8
smha = ScaledMultiHeadAttention(4, 8)
print(smha(mha_input, mha_input, mha_input)) # 1*2*5*8


