import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import save_model, load_model
from dataset import data_loader
from dataset import tokenizer
from network import transformer_network
from model import train_model
from visualization import utils
from model import inference
import tensorflow as tf 

print(tf.config.list_physical_devices("GPU"))
print(tf.config.list_physical_devices("CPU"))


DATA_PATH = "src/dataset/spa.txt"
MAX_TOKEN = 15000
SEQUENCE_LENGTH = 20
BATCH_SIZE = 64
EPOCHS = 10
EMBED_DIM = 128
MODEL_PATH = "src/weights/Best_Model_Transformer.h5"


def main():
    # load data
    doc = data_loader.load_doc(DATA_PATH)
    pairs = data_loader.make_pairs(doc)
    train, validation, test = data_loader.train_test_split_data(pairs)
    train_dataset, validation_dataset, test_dataset, spanish_tokenizer = tokenizer.vectorize(train, validation, test, MAX_TOKEN, SEQUENCE_LENGTH, BATCH_SIZE)
    vocabs = spanish_tokenizer.get_vocabulary() # to reverse predictions from numbers to words

    # Run this for train
    # seq2seq = transformer_network.Transformers(MAX_TOKEN, SEQUENCE_LENGTH, EMBED_DIM)
    # model = seq2seq.build_model(input_shape_source=(SEQUENCE_LENGTH,), input_shape_target=(SEQUENCE_LENGTH,))
    # H = train_model.train(model, train_dataset, validation_dataset, BATCH_SIZE, EPOCHS)
    # utils.plot_metrics(H) # change accuracy metric name of your choice in plot_metrics

    
    # Run this for inference
    for inputs in test_dataset.take(1):
        t = inputs[0]["English"][0]
        label = inputs[1]

    translated = inference.infer(MAX_TOKEN, SEQUENCE_LENGTH, EMBED_DIM, t, MODEL_PATH, vocabs)

    print("Correct Translation: ", label[0].numpy().decode('utf-8'))
    print(f"Model translation: {translated}")

if __name__ == "__main__":
    main()
