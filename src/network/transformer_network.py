import os
os.environ["TF_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from keras.layers import Dense, Input, Embedding, Layer
from keras import Model
from network.positional_encoding import PositionalEncoding
from network.encoder import Encoder
from network.decoder import Decoder



class Transformers(Layer):
    """
    Args:
        vocab_size: number of distinct tokens, dictionary size of both source and destination inputs
        sequence_length: maximum tokens in a sequence. Equal for both source and destination inputs
        embedding_dim: words embedding dimension
        n_encoders: number of encoders (layers)
        n_decoders: number of decoders (layers)
    
    Attributes:
        vocab_size: number of maximum tokens in encoder and decoder
        enc_embedding: embedding layer for encoder inputs
        dec_embedding: embedding layer for decoder inputs
        positional_encoding: positional encoding layer
        encoder_layers: a list of encoder layers
        decoder_layers: a list of decoder layers
    """

    def __init__(self, vocab_size: int, sequence_length: int, embedding_dim: int, n_encoders: int=1, n_decoders: int=1, n_heads: int=4) -> None:
        super(Transformers, self).__init__()
        self.vocab_size = vocab_size
        self.enc_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.dec_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.positional_encoding = PositionalEncoding(sequence_length, embedding_dim)
        self.encoder_layers = [Encoder(embedding_dim, n_heads=n_heads) for _ in range(n_encoders)]
        self.decoder_layers = [Decoder(embedding_dim, n_heads=n_heads) for _ in range(n_decoders)]


    def call(self, source_data: tf.Tensor, target_data: tf.Tensor) -> tf.Tensor:
        """
        Transformer architecture assembles together.

        Args:
            source_data: encoder input
            target_data: decoder input

        Returns:
            result: final output of the transformer block which are the extracted features before nonlinearity.
        """

        # find padding mask
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

        result = Dense(self.vocab_size)(dec_input) 
        # logits because of padding masking we want to calculate probabilities in a self defined loss 

        return result
        

    def build_model(self, input_shape_source: tuple, input_shape_target: tuple) -> Model:
        """
        Building transformer model with input and output layers.

        Args:
            input_shape_source: shape of the encoder input (seq_length,)
            input_shape_target: shape of the decoder input (seq_length,)

        Returns:
            Transformer model
        
        """
        
        source_input = Input(shape=input_shape_source, name="English")
        target_input = Input(shape=input_shape_target, name="Spanish")

        outputs = self.call(source_input, target_input)

        return Model(inputs=[source_input, target_input], outputs=outputs)
    
