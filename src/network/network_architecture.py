import os
os.environ["TF_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings('ignore')
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras import Model

def build_network(vocab_size, embed_dim, sequence_length):
    """
    Seq2seq model architecture for training

    Inputs:
        vocab_size: maximum number of words in the defined dictionary
        embed_dim: words embedding dimension

    Outputs:
        model: an encoder-decoder model for seq2seq prediction
    
    """
    eng_input = Input(shape=(None,), name="English")
    e_input = Embedding(vocab_size, embed_dim, mask_zero=True)(eng_input)

    positional_embedding = get_positional_encoding(sequence_length, embed_dim)

    embeddings = e_input + positional_embedding
    # print(e_input.shape)
    # print(positional_embedding.shape)
    # print(embeddings.shape)

    

    # spa_input = Input(shape=(None,), name="Spanish")
    # e_output = Embedding(vocab_size, embed_dim, mask_zero=True)(spa_input)




    # model = Model([eng_input, spa_input], den)

    #return model

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

