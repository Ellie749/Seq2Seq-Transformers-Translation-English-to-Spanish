import os
os.environ["TF_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings('ignore')
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, MultiHeadAttention, LayerNormalization, Layer, Softmax
from tensorflow.keras import Model


class PositionalEncoding():
    def __init__(self, sequence_length, embedding_dim):
        super(PositionalEncoding, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = (embedding_dim // 2) - 1
        self.embedding_matrix = np.zeros((sequence_length, embedding_dim))
        
    def __call__(self, ):
        for pos in range(self.sequence_length):
            for i in range(self.embedding_dim):
                self.embedding_matrix[pos, 2*i] = math.sin(pos / (math.pow(10000, (2*i / self.embedding_dim))))
                self.embedding_matrix[pos, 2*i+1] = math.cos(pos / (math.pow(10000, (2*i / self.embedding_dim))))

        return tf.convert_to_tensor(self.embedding_matrix, dtype=tf.float32)
    

class ScaledMultiHeadAttention():
    def __init__(self, n_heads, embedding_dim, **mask:bool): #add different shapes for dk and dmodel and test
        super(ScaledMultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.mask = mask
        self.embedding = embedding_dim
        self.scale = embedding_dim**0.5
        self.activation = Softmax()
        self.att_weight = Dense(embedding_dim)

    def __call__(self, Q, K, V):
        logit_matrix = None
        def mlp_layer():
            Q_enc = Dense(self.embedding)
            K_enc = Dense(self.embedding)
            V_enc = Dense(self.embedding)
            return [Q_enc, K_enc, V_enc]

        for i in range(self.n_heads):
            weights = mlp_layer() # every time different weights are created
            temp = (self.activation(weights[0](Q) @ tf.transpose(weights[1](K), perm=(0, 2, 1))/self.scale) ) @ weights[2](V)
            if logit_matrix is None:
                logit_matrix = temp
            else:
                logit_matrix = tf.concat([logit_matrix, temp], axis = 2) # (batchsize, seq_length, embedding_dim  -> comcatenation should be applied on embedding_dim axis

        return self.att_weight(logit_matrix)


class FFN(Layer):
    def __init__(self, embedding_dim):
        super(FFN, self).__init__()
        self.embedding_dim = embedding_dim

    def call(self, x): #dense is applied to the last dimension automatically
        return Dense(self.embedding_dim)(Dense(4*self.embedding_dim, activation='relu')(x)) #according to the paper


class Encoder(Layer):
    def __init__(self, vocab_size, sequence_length, embedding_dim):
        super(Encoder, self).__init__()
        self.e_input = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.positional_encoding = PositionalEncoding(sequence_length, embedding_dim)
        # self.input_matrix = self.e_input + self.positional_encoding
        self.mha = ScaledMultiHeadAttention(4, embedding_dim)
        self.norm = LayerNormalization()
        self.ffn = FFN(embedding_dim)


    def call(self, input_layer):
        x = self.e_input(input_layer)
        print(x)
        pe = self.positional_encoding()
        x = pe + x
        y = self.mha(x, x, x)
        z = self.norm(y + x)
        x = self.ffn(z)
        x = self.norm(x + z)
        return x



class Decoder(Layer):
    def __init(self):
        super(self).__init__()
        


class Transformers():
    pass



'''
Q = tf.convert_to_tensor([[2,3], [4,2]])
mha = ScaledMultiHeadAttention(4, 4)
print(mha(Q, Q, Q))
'''

data = np.array([[2, 3, 4, 5, 0], [6, 7, 8, 9, 0]])

enc = Encoder(10, 5, 4)
inp = Input(shape=(None,))
print(enc(data))