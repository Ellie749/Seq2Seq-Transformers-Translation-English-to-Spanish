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
    def __init__(self, n_heads, embedding_dim, mask=False): #add different shapes for dk and dmodel and test
        super(ScaledMultiHeadAttention, self).__init__()
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
            scaled_e_matrix = (weights[0](Q) @ tf.transpose(weights[1](K), perm=(0, 2, 1))) / self.scale

            if self.mask == True:
                scaled_e_matrix = scaled_e_matrix + mask_matrix

            temp = self.activation(scaled_e_matrix) @ weights[2](V)

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
    def __init__(self, embedding_dim, n_heads=1):
        super(Encoder, self).__init__()
        self.mha = ScaledMultiHeadAttention(n_heads, embedding_dim)
        self.norm = LayerNormalization() # should norm be different for each norm layer in enc and dec?
        self.ffn = FFN(embedding_dim)


    def call(self, input_layer):
        y = self.mha(input_layer, input_layer, input_layer)
        res2 = self.norm(y + input_layer)
        x = self.ffn(res2)
        x = self.norm(x + res2)
        return x


class Decoder(Layer):
    def __init__(self, embedding_dim, n_heads=1):
        super(Decoder, self).__init__()
        self.mmha = ScaledMultiHeadAttention(n_heads, embedding_dim, mask=True)
        self.mha = ScaledMultiHeadAttention(n_heads, embedding_dim)
        self.norm = LayerNormalization()
        self.ffn = FFN(embedding_dim)


    def call(self, input_layer, K, V):
        x = self.mmha(input_layer, input_layer, input_layer)
        res2 = self.norm(x + input_layer)
        x = self.mha(res2, K, V)
        res3 = self.norm(x + res2)
        x = self.ffn(res3)
        x = self.norm(x + res3)
        return x



class Transformers(Layer):
    def __init__(self, vocab_size, sequence_length, embedding_dim, n_encoders=1, n_decoders=1):
        super(Transformers, self).__init__()
        self.enc_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.dec_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.positional_encoding = PositionalEncoding(sequence_length, embedding_dim)

        self.encoder_layers = [Encoder(embedding_dim, n_heads=4) for _ in range(n_encoders)]
        self.decoder_layers = [Decoder(embedding_dim, n_heads=4) for _ in range(n_decoders)]


    def call(self):
        source_data = Input(shape=(None,), name="English")
        target_data = Input(shape=(None,), name="Spanish")
        source_embeddings = self.enc_embedding(source_data)
        pe = self.positional_encoding()
        enc_input = pe + source_embeddings

        for encoder in self.encoder_layers:
            enc_input = encoder(enc_input)


        target_embeddings = self.dec_embedding(target_data)
        dec_input = pe + target_embeddings

        for decoder in self.decoder_layers:
            dec_input = decoder(dec_input, enc_input, enc_input)


        result = Dense(10, activation='softmax')(dec_input)

        model = Model([source_data, target_data], result)

        return model
        


# test
# data_enc = np.array([[2, 3, 4, 5, 0], [6, 7, 8, 9, 0]])
# data_dec = np.array([[2, 2, 1, 0, 0], [5, 1, 0, 0, 0]])
# label_dec = np.array([[2, 1, 0, 0, 0], [1, 5, 0, 0, 0]])
# transformer_network = Transformers(10, 5, 4, n_encoders=4, n_decoders=2)
# result = transformer_network(data_enc, data_dec)
# print(result)