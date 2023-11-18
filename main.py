from plotting import plot_images_and_labels
from constants import *
from train_model import train_model
from evaluate_model import compute_model_accuracy, display_model_predictions


def main():
    plot_images_and_labels(3, False)
    model_name = train_model(INPUT_SHAPE, FILTERS, N_CLASSES, EPOCHS, BATCH_SIZE, TRAIN_SIZE, True, True, True, True)
    compute_model_accuracy(model_name, BATCH_SIZE, TRAIN_SIZE)
    display_model_predictions(model_name, BATCH_SIZE, TRAIN_SIZE)


if __name__ == "__main__":
    main()
