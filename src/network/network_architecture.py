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
        self.Q_enc = Dense(self.embedding)
        self.K_enc = Dense(self.embedding)
        self.V_enc = Dense(self.embedding)

    def __call__(self, Q, K, V):
        logit_matrix = None

        if self.mask == True:
            inf_number = -1e9
            mask_matrix = np.zeros(((Q.shape)[1], (Q.shape)[1]))
            for i in range((Q.shape)[1]):
                mask_matrix[i, i+1:] = inf_number
            #tf.expand_dims(mask_matrix, axis=0)
            # print(mask_matrix)
            
        def mlp_layer():
            Q_enc = Dense(self.embedding)
            K_enc = Dense(self.embedding)
            V_enc = Dense(self.embedding)
            return [Q_enc, K_enc, V_enc]

        for i in range(self.n_heads):
            #weights = mlp_layer() # every time different weights are created
            scaled_e_matrix = (self.Q_enc(Q) @ tf.transpose(self.K_enc(K), perm=(0, 2, 1))) / self.scale
            if self.mask == True:
                scaled_e_matrix = scaled_e_matrix + mask_matrix

            temp = self.activation(scaled_e_matrix) @ self.K_enc(V)
            if logit_matrix is None:
                logit_matrix = temp
            else:
                logit_matrix = tf.concat([logit_matrix, temp], axis = -1) # (batchsize, seq_length, embedding_dim  -> comcatenation should be applied on embedding_dim axis

        return self.att_weight(logit_matrix)


class FFN(Layer):
    def __init__(self, embedding_dim):
        super(FFN, self).__init__()
        self.embedding_dim = embedding_dim
        self.node = Dense(4*self.embedding_dim, activation='relu')
        self.out = Dense(self.embedding_dim)

    def call(self, x): #dense is applied to the last dimension automatically
        return self.out(self.node(x)) #according to the paper it maps to 4*emb


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


    def call(self, vocab_size, source_data, target_data):
        
        source_embeddings = self.enc_embedding(source_data)

        pe = self.positional_encoding()
        enc_input = pe + source_embeddings

        for encoder in self.encoder_layers:
            enc_input = encoder(enc_input)


        target_embeddings = self.dec_embedding(target_data)
        dec_input = pe + target_embeddings

        for decoder in self.decoder_layers:
            dec_input = decoder(dec_input, enc_input, enc_input)


        result = Dense(vocab_size, activation='softmax')(dec_input)
        

        return result
        
    def build_model(self, vocab_size, input_shape_source, input_shape_target):
        # Define input layers
        source_input = Input(shape=input_shape_source, name="English")
        target_input = Input(shape=input_shape_target, name="Spanish")

        outputs = self.call(vocab_size, source_input, target_input)

        return Model(inputs=[source_input, target_input], outputs=outputs)
    

# enc_inp = tf.convert_to_tensor(np.array([[[2, 3, 4, 5, 0], [6, 7, 8, 9, 0]]]), dtype=tf.float32) 
# dec_inp = tf.convert_to_tensor(np.array([[[2, 2, 1, 0, 0], [5, 1, 0, 0, 0]]]), dtype=tf.float32)
# transformer_network = Transformers(10, 5, 8)
# output = transformer_network()
# source_data = Input(shape=(None,), name="English")
# target_data = Input(shape=(None,), name="Spanish")
# model = Model([source_data, target_data], output)
