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
        
    def __call__(self):
        for pos in range(self.sequence_length):
            for i in range(self.embedding_dim):
                self.embedding_matrix[pos, 2*i] = math.sin(pos / (math.pow(10000, (2*i / self.embedding_dim))))
                self.embedding_matrix[pos, 2*i+1] = math.cos(pos / (math.pow(10000, (2*i / self.embedding_dim))))

        return tf.convert_to_tensor(self.embedding_matrix)
    

class ScaledMultiHeadAttention():
    def __init__(self, n_heads, embedding_dim, **mask:bool): #add different shapes for dk and dmodel and test
        super(ScaledMultiHeadAttention).__init__()
        self.n_heads = n_heads
        self.mask = mask
        self.scale = embedding_dim**0.5
        self.activation = Softmax()
        self.Q_enc = Dense(embedding_dim)
        self.K_enc = Dense(embedding_dim)
        self.V_end = Dense(embedding_dim)
        self.att_weight = Dense(embedding_dim)

    def call(self, Q, K, V):
        temp = []
        for i in range(self.n_heads):
            temp = tf.concat(
                self.activation((self.Q_enc(Q) @ tf.transpose(self.K_enc(K))) / self.scale ) @ self.V_enc(V)
                )
        
        return self.att_weight(temp)




class Encoder(Layer):
    def __init__(self, vocab_size, embedding_dim):
        super(Encoder, self).__init__()
        self.e_input = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.positional_encoding = PositionalEncoding(20, 128)
        # self.input_matrix = self.e_input + self.positional_encoding
        self.mha = ScaledMultiHeadAttention()
        self.norm = LayerNormalization()


    def call(self, input_layer):
        x = self.e_input(input_layer)
        x = self.positional_encoding(x)
        x = self.e_input + x
        y = self.mha(x, x, x)
        x = self.norm(y + x)




class Decoder(Layer):
    def __init(self):
        super(self).__init__()
        


class Transformers():
    pass

def build_network(vocab_size, embed_dim, sequence_length):
    """
    Seq2seq model architecture for training

    Inputs:
        vocab_size: maximum number of words in the defined dictionary
        embed_dim: words embedding dimension

    Outputs:
        model: an encoder-decoder model for seq2seq prediction using transformers
    
    """
    # Encoder
    eng_input = Input(shape=(None,), name="English")
    e_input = Embedding(vocab_size, embed_dim, mask_zero=True)(eng_input)
    positional_embedding_enc = get_positional_encoding(sequence_length, embed_dim)
    mask = tf.cast(tf.not_equal(eng_input, 0), dtype=tf.float32)[:, :, None]
    embeddings_enc = e_input + positional_embedding_enc*mask # None, 20, 128
    Q_enc = Dense(128)(embeddings_enc) 
    K_enc = Dense(128)(embeddings_enc)
    V_enc = Dense(128)(embeddings_enc)
    # input to Dense is not necessarily a vector. 
    # input = N-D tensor with shape: (batch_size, ..., input_dim)
    # output = N-D tensor with shape: (batch_size, ..., units)
    attention_embedding_enc, att_scores = MultiHeadAttention(4, 128)(query=Q_enc, key=K_enc, value=V_enc, return_attention_scores=True)
    added_embeddings1 = attention_embedding_enc + embeddings_enc
    normalized_embeddings_enc = LayerNormalization()(added_embeddings1) #It normalizes across the last dimension by default
    x = Dense(256, activation='relu')(normalized_embeddings_enc)
    x = Dense(128)(x)
    added_embeddings2_enc = x + normalized_embeddings_enc
    encoder_embeddings = LayerNormalization()(added_embeddings2_enc)
    print(f"encoder output shape: {encoder_embeddings.shape}")




    # Decoder
    spa_input = Input(shape=(None,), name="Spanish")
    e_output = Embedding(vocab_size, embed_dim, mask_zero=True)(spa_input)
    positional_embedding_dec = get_positional_encoding(sequence_length, embed_dim)
    mask = tf.cast(tf.not_equal(eng_input, 0), dtype=tf.float32)[:, :, None]
    embeddings_dec = e_output + positional_embedding_dec*mask
    Q_dec = Dense(128)(embeddings_dec)
    K_dec = Dense(128)(embeddings_dec)
    V_dec = Dense(128)(embeddings_dec)
    attention_embedding_dec, _ = MultiHeadAttention(4, 128)(query=Q_dec, key=K_dec, value=V_dec, return_attention_scores=True, use_causal_mask=True)
    added_embeddings1_dec = attention_embedding_dec + embeddings_dec
    normalized_embeddings1_dec = LayerNormalization()(added_embeddings1_dec)
    cross_attention, _ = MultiHeadAttention(4, 128)(query=normalized_embeddings1_dec, key=encoder_embeddings, value=encoder_embeddings, return_attention_scores=True)
    added_embeddings2_dec = cross_attention + normalized_embeddings1_dec
    normalized_embeddings2_dec = LayerNormalization()(added_embeddings2_dec)
    y = Dense(256, activation='relu')(normalized_embeddings2_dec)
    y = Dense(128)(y)
    added_embeddings3_dec = y + normalized_embeddings2_dec
    decoder_embeddings = LayerNormalization()(added_embeddings3_dec)


    final_output = Dense(vocab_size, activation='softmax')(decoder_embeddings)

    model = Model([eng_input, spa_input], final_output)

    print(model.summary)

    return model

    

def get_positional_encoding(sequence_length, embed_dim):
    embedding_matrix = np.zeros((sequence_length, embed_dim))
    for pos in range(sequence_length):
        for i in range(embed_dim):
            if i%2==0: 
                embedding_matrix[pos,i] = math.sin(pos/(math.pow(10000, (2*i)/embed_dim )))
            elif i%2 != 0:
                embedding_matrix[pos,i] = math.cos(pos/(math.pow(10000, (2*i)/embed_dim )))
    
    # print(embedding_matrix)
    # plt.matshow(embedding_matrix)
    # plt.show()
    
    #return tf.cast(embedding_metrix, dtype=tf.float32)
    return tf.convert_to_tensor(embedding_matrix, dtype='float32')


'''
pe = get_positional_encoding(12, 10)
inp = Embedding(10, 10, mask_zero=True)(np.array([[2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0], [6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0]]))
print(pe)
print(inp)
print(pe + inp)

# positional encodings get added to 0s as well anyways. how to prevent them to be calculated in the model? 
# mask 0 is true but why they are assigned a value
'''

Encoder(vocab_size=3500, embedding_dim=128)