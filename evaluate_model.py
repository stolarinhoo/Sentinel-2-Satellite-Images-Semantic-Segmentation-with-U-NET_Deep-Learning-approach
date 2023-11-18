import keras
from keras.models import load_model
from keras.utils import custom_object_scope

from data_preprocessing import prepare_datasets
from plotting import plot_model_predictions
from train_model import __masked_sparse_categorical_crossentropy


def compute_model_accuracy(model_filepath: str, batch_size: int, train_size: float):
    model = load_unet_model(model_filepath)

    train_dataset, test_dataset, val_dataset = prepare_datasets(batch_size, train_size)

    train_loss, train_accuracy = model.evaluate(train_dataset, batch_size=batch_size)
    test_loss, test_accuracy = model.evaluate(test_dataset, batch_size=batch_size)
    validation_loss, validation_accuracy = model.evaluate(val_dataset, batch_size=batch_size)

    print(f'Model Accuracy on the Training Dataset: {round(train_accuracy * 100, 2)}%')
    print(f'Model Accuracy on the Validation Dataset: {round(validation_accuracy * 100, 2)}%')
    print(f'Model Accuracy on the Test Dataset: {round(test_accuracy * 100, 2)}%')


def display_model_predictions(model_filepath: str, batch_size: int, test_size: int,
                              number_of_batches: int = 1, number_of_images: int = 3) -> None:
    loaded_model = load_unet_model(model_filepath)
    plot_model_predictions(loaded_model, number_of_batches, number_of_images, batch_size, test_size)


def load_unet_model(model_filepath: str) -> keras.Model:
    with custom_object_scope({'__masked_sparse_categorical_crossentropy': __masked_sparse_categorical_crossentropy}):
        model = load_model(model_filepath)
    return model
