import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input


def translate(inf_model, t, end_token, start_token, sequence_length):


    target_sequence = np.zeros((1, sequence_length), dtype=np.int32)
    target_sequence[0, 0] = start_token 
    t = tf.reshape(t, (1, -1))
    print(t.shape)
    print(target_sequence.shape)

    for i in range(1):

        prediction = inf_model.predict([t, target_sequence])
        print(prediction.shape)

        vocab = np.argmax(prediction[0, i-1])
        print(vocab)
        if(vocab == end_token):
            break
        
        # decoded_sentence.append(vocab)
        # target_seq = np.array([vocab])
      