from tensorflow.keras.callbacks import ModelCheckpoint


def train(seq2seq, train_dataset, validation_dataset, BATCH_SIZE, EPOCHS):
    """
    train and save model
    
    Inputs:
        seq2seq: model architecture
        train_dataset: train data
        validation_dataset: validation data
        BATCH_SIZE: batch size
        EPOCHS: epochs

    Outputs:
        H: history of training

    """
    seq2seq.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    checkpoint = ModelCheckpoint('Best_Model_Multi_Transformer.h5', monitor='val_accuracy', mode='max', save_best_only='True', save_format="h5")
    H = seq2seq.fit(train_dataset, validation_data=validation_dataset, batch_size=BATCH_SIZE, epochs=2, callbacks=[checkpoint])
    
    return H
