import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from network import transformer_network


def infer(vocab_size: int, sequence_length: int , embedding_dim: int, test_data: tf.Tensor, model_path: str, vocabs: list) -> str:
    """
    Initializing inference model and doing the translation.

    Args:
        vocab_size: maximum number of words in the defined dictionary
        sequence_length: Maximum length of the translation sequence
        embedding_dim: words embedding dimension
        test_data: source sequence to be translated
        model_path: path to the weights of the model that is going to do the translation
        vocabs: vectorization object of the destination language to convert predictions from number space to vocabulary space

    Retunrs:
        final_translation: which is converted to string
    """

    inf_model = transformer_network.Transformers(vocab_size, sequence_length, embedding_dim)
    inf_model = inf_model.build_model(input_shape_source=(sequence_length,), input_shape_target=(sequence_length,))
    inf_model.load_weights(model_path)

    end_token = vocabs.index("[end]") #int
    start_token = vocabs.index("[start]") #int
    translated_seq = translate(inf_model, test_data, end_token, start_token, sequence_length)
    final_translation = []

    for i in range(len(translated_seq)):
        final_translation.append(vocabs[translated_seq[i]])
    
    return " ".join(final_translation)


def translate(inf_model: Model, test_data: tf.Tensor, end_token: int, start_token: int, sequence_length: int) -> list:
    """
    Predicting and making the output sequence.

    Args:
        inf_model: inference model that does the prediction
        test_data: source sequence to be translated
        end_token: identifier of the last token which the model should stop predicting upon producing it
        start_token: initial token to be given to the decoder part of the model
        sequence_length: maximum length of the translation sequence

    Retunrs:
        decoded_sentence: list of predicted tokens in their numerical/tokenized format
    """
        
    decoded_sentence = []
    target_sequence = np.zeros((1, sequence_length), dtype=np.int32)
    target_sequence[0, 0] = start_token 
    test_data = tf.reshape(test_data, (1, -1))
    print(target_sequence.shape)

    for i in range(sequence_length):

        prediction = inf_model.predict([test_data, target_sequence])
        vocab = np.argmax(prediction[0, i]) # Transformer produces 20 tokens regardless - we have to take the timestep we want and add to the input of decoder
        # You should add softmax
        if(vocab == end_token):
            break
        
        decoded_sentence.append(vocab)
        target_sequence[0, i] = np.array([vocab])

    return decoded_sentence
      