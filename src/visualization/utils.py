import matplotlib.pyplot as plt


def plot_metrics(H):
    """
    Visualizing loss and accuracy diagrams based on history of a model
    """
    plt.subplot(121)
    plt.title("Loss")
    plt.plot(range(len(H.history['loss'])), H.history['loss'], label="Train Loss")
    plt.plot(range(len(H.history['val_loss'])), H.history['val_loss'], label="Validation Loss")
    plt.legend()

    plt.subplot(122)
    plt.title("Accuracy")
    plt.plot(range(len(H.history['masked_accuracy_function'])), H.history['masked_accuracy_function'], label="Train Accuracy")
    plt.plot(range(len(H.history['val_masked_accuracy_function'])), H.history['val_masked_accuracy_function'], label="Validation Accuracy")
    plt.legend()
    
    plt.savefig("metrics.png")
    plt.show()