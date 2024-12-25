import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import save_model
from dataset import data_loader
from dataset import tokenizer
from network import transformer_network
from model import train_model
from visualization import utils
#import inference

#print(config.list_physical_devices("GPU"))
#print(config.list_physical_devices("CPU"))


DATA_PATH = "src/dataset/spa.txt"
MAX_TOKEN = 15000
SEQUENCE_LENGTH = 20
BATCH_SIZE = 64
EPOCHS = 15
EMBED_DIM = 128
#LATENT_DIM = 1024


def main():
    doc = data_loader.load_doc(DATA_PATH)
    pairs = data_loader.make_pairs(doc)
    train, validation, test = data_loader.train_test_split_data(pairs)
    train_dataset, validation_dataset, test_dataset, spanish_tokenizer = tokenizer.vectorize(train, validation, test, MAX_TOKEN, SEQUENCE_LENGTH, BATCH_SIZE)
    vocabs = spanish_tokenizer.get_vocabulary() # to reverse predictions from numbers to words

    for batch in train_dataset.take(1):
        print(batch)  # Print a sample batch

    # print(len(train))
    for inputs, targets in train_dataset.take(1):
        print(f"english input: {inputs['English'].shape}")
        print(inputs['English'][6])
        print(f"Spanish input: {inputs['Spanish'].shape}")
        print(inputs['Spanish'][6])
        print(f"Spanish output: {targets.shape}")

    seq2seq = transformer_network.build_network(MAX_TOKEN, EMBED_DIM, SEQUENCE_LENGTH)
    H = train_model.train(seq2seq, train_dataset, validation_dataset, BATCH_SIZE, EPOCHS)
    utils.plot_metrics(H)
    

if __name__ == "__main__":
    main()
