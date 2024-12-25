import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import save_model, load_model
from dataset import data_loader
from dataset import tokenizer
from network import network_architecture_multi_transformer
from model import train_model
from visualization import utils
from model import inference
#import inference

#print(config.list_physical_devices("GPU"))
#print(config.list_physical_devices("CPU"))


DATA_PATH = "src/dataset/spa.txt"
MAX_TOKEN = 15000
SEQUENCE_LENGTH = 20
BATCH_SIZE = 64
EPOCHS = 1
EMBED_DIM = 128
#LATENT_DIM = 1024


def main():
    doc = data_loader.load_doc(DATA_PATH)
    pairs = data_loader.make_pairs(doc)
    train, validation, test = data_loader.train_test_split_data(pairs)
    train_dataset, validation_dataset, test_dataset, spanish_tokenizer = tokenizer.vectorize(train, validation, test, MAX_TOKEN, SEQUENCE_LENGTH, BATCH_SIZE)
    vocabs = spanish_tokenizer.get_vocabulary() # to reverse predictions from numbers to words


    # Run this for train
    # seq2seq = network_architecture_multi_transformer.Transformers(MAX_TOKEN, SEQUENCE_LENGTH, EMBED_DIM)
    # model = seq2seq.build_model(input_shape_source=(SEQUENCE_LENGTH,), input_shape_target=(SEQUENCE_LENGTH,))
    # H = train_model.train(model, train_dataset, validation_dataset, BATCH_SIZE, EPOCHS)
    # utils.plot_metrics(H)

    
    # Run this for inference
    inf_model = network_architecture_multi_transformer.Transformers(MAX_TOKEN, SEQUENCE_LENGTH, EMBED_DIM)
    inf_model = inf_model.build_model(input_shape_source=(SEQUENCE_LENGTH,), input_shape_target=(SEQUENCE_LENGTH,))

    inf_model.load_weights("src/weights/Best_Model_Multi_Transformer.h5")
    for inputs in test_dataset.take(1):
        t = inputs[0]["English"][0]
        label = inputs[1]

    end_token = vocabs.index("[end]") #int
    start_token = vocabs.index("[start]") #int
    translated_seq = inference.translate(inf_model, t, end_token, start_token, SEQUENCE_LENGTH)
    final_translation = []

    for i in range(len(translated_seq)):
        final_translation.append(vocabs[translated_seq[i]])
    
    print("Prediction: ", " ".join(final_translation))
    print("Correct Translation: ", label[0].numpy().decode('utf-8'))

if __name__ == "__main__":
    main()
