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
        self.d_k = embedding_dim // n_heads
        self.scale = embedding_dim**0.5
        self.inf_number = -1e9
        self.activation = Softmax(axis=-1)
        self.att_weight = Dense(embedding_dim)
        self.Q_enc = [Dense(self.d_k) for _ in range(n_heads)] # Each head should be separate 
        self.K_enc = [Dense(self.d_k) for _ in range(n_heads)]
        self.V_enc = [Dense(self.d_k) for _ in range(n_heads)] # add dk

    def __call__(self, Q, K, V, pad_mask):
        logit_matrix = None
        print(f"self.d_k: {self.d_k, self.embedding, self.n_heads}")

        
        if self.mask == True:
            mask_matrix = np.zeros(((Q.shape)[1], (Q.shape)[1]))
            for i in range((Q.shape)[1]):
                mask_matrix[i, i+1:] = self.inf_number
            # tf.expand_dims(mask_matrix, axis=0)
            
        for i in range(1):
            scaled_e_matrix = (self.Q_enc[i](Q) @ tf.transpose(self.K_enc[i](K), perm=(0, 2, 1))) / self.scale
            # don't consider batch since the Model and Keras handles it automatically

            scaled_e_matrix += (1.0 - pad_mask * tf.transpose(pad_mask, perm=[0, 2, 1])) * self.inf_number #pad_masking in rows and columns (making sure broadcasting i sapplied on columns as well)

            if self.mask == True:
                scaled_e_matrix = scaled_e_matrix + mask_matrix
            
            # Softmax doesn't 0 out the padded masks in rows. It only applies in columns -> *pad_mask
            temp = self.activation(scaled_e_matrix)*pad_mask @ self.V_enc[i](V)
           
            if logit_matrix is None:
                logit_matrix = temp
            else:
                logit_matrix = tf.concat([logit_matrix, temp], axis = -1) # (batchsize, seq_length, embedding_dim  -> comcatenation should be applied on embedding_dim axis

        # calculate each head in smaller dimension separately and then concat or concat and then convert to smaller dimensions?
        return self.att_weight(logit_matrix)


class FFN(Layer):
    def __init__(self, embedding_dim):
        super(FFN, self).__init__()
        self.embedding_dim = embedding_dim
        self.node = Dense(4*self.embedding_dim, activation='relu') #according to the paper it maps to 4*emb
        self.out = Dense(self.embedding_dim)

    def call(self, x): #dense is applied to the last dimension automatically
        return self.out(self.node(x)) 


class Encoder(Layer):
    def __init__(self, embedding_dim, n_heads=1):
        super(Encoder, self).__init__()
        self.mha = ScaledMultiHeadAttention(n_heads, embedding_dim)
        self.norm = LayerNormalization() # should norm be different for each norm layer in enc and dec?
        self.ffn = FFN(embedding_dim)


    def call(self, input_layer, pad_mask):
        y = self.mha(input_layer, input_layer, input_layer, pad_mask)
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


    def call(self, input_layer, K, V, pad_mask):
        x = self.mmha(input_layer, input_layer, input_layer, pad_mask)
        res2 = self.norm(x + input_layer)
        x = self.mha(res2, K, V, pad_mask)
        res3 = self.norm(x + res2)
        x = self.ffn(res3)
        x = self.norm(x + res3)
        return x


class Transformers(Layer):
    def __init__(self, vocab_size, sequence_length, embedding_dim, n_encoders=1, n_decoders=1):
        super(Transformers, self).__init__()
        self.vocab_size = vocab_size
        self.enc_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.dec_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.positional_encoding = PositionalEncoding(sequence_length, embedding_dim)
        self.encoder_layers = [Encoder(embedding_dim, n_heads=4) for _ in range(n_encoders)]
        self.decoder_layers = [Decoder(embedding_dim, n_heads=4) for _ in range(n_decoders)]


    def call(self, source_data, target_data):
        source_mask = tf.cast(tf.not_equal(source_data, 0), dtype=tf.float32)[:, :, None] #changes (batch, seq_length) to (batch, seq_length, 1) None adds one dimension
        target_mask = tf.cast(tf.not_equal(target_data, 0), dtype=tf.float32)[:, :, None]
        pe = self.positional_encoding()

        source_embeddings = self.enc_embedding(source_data)
        enc_input = pe*source_mask + source_embeddings
     
        for encoder in self.encoder_layers:
            enc_input = encoder(enc_input, source_mask)

        target_embeddings = self.dec_embedding(target_data)
        dec_input = pe*target_mask + target_embeddings

        for decoder in self.decoder_layers:
            dec_input = decoder(dec_input, enc_input, enc_input, target_mask)

        result = Dense(self.vocab_size, activation='softmax')(dec_input)        

        return result
        
        
    def build_model(self, vocab_size, input_shape_source, input_shape_target):
        
        source_input = Input(shape=input_shape_source, name="English")
        target_input = Input(shape=input_shape_target, name="Spanish")

        outputs = self.call(vocab_size, source_input, target_input)

        return Model(inputs=[source_input, target_input], outputs=outputs)
    

source_data = tf.convert_to_tensor([[1,4,2,6,3,3,0,0], [1,3,9,5,1,3,2,2]])
target_data = tf.convert_to_tensor([[4,2,2,2,2,6,0,0], [3,2,2,5,6,7,7,9]])
trans = Transformers(10, 8, 4)
print(trans(source_data, target_data))