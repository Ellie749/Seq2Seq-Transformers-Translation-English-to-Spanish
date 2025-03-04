import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


def train(seq2seq, train_dataset, validation_dataset, BATCH_SIZE, EPOCHS):
    """
    train and save model
    
    Args:
        seq2seq: model architecture
        train_dataset: train data
        validation_dataset: validation data
        BATCH_SIZE: batch size
        EPOCHS: epochs

    Returns:
        H: history of training

    """
    seq2seq.compile(optimizer='adam', loss=masked_loss_function, metrics=[masked_accuracy_function])
    checkpoint = ModelCheckpoint('src/weights/Best_Model_Transformer.h5', monitor='val_masked_accuracy_function', mode='max', save_best_only='True', save_format="h5")
    H = seq2seq.fit(train_dataset, validation_data=validation_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[checkpoint])
    
    return H


def masked_loss_function(y_true: tf.Tensor, y_pred: tf.Tensor):
    """
    Loss function for transformer model. The loss function requires masking in order to ignore padded tokens in the dataset.

    Args:
        y_true: actual label
        y_pred: model's predicted label

    Returns:
        cross entropy loss result between actual and predicted label after masking

    """

    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    masked_loss = loss * mask
   
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)


def masked_accuracy_function(y_true: tf.Tensor, y_pred: tf.Tensor):
    """
    Accuracy function for transformer model. The accuracy function requires masking in order to ignore padded tokens in the dataset.

    Args:
        y_true: actual label
        y_pred: model's predicted label

    Returns:
        categorical accuracy result between actual and predicted label after masking

    """

    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    y_pred_classes = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    matches = tf.cast(tf.equal(y_true, tf.cast(y_pred_classes, y_true.dtype)), dtype=tf.float32)
    matches *= mask
    
    return tf.reduce_sum(matches) / tf.reduce_sum(mask)
