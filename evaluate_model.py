import keras
from keras.models import load_model
from keras.utils import custom_object_scope

from data_preprocessing import prepare_datasets
from plotting import plot_model_predictions
from train_model import _masked_sparse_categorical_crossentropy


def compute_model_accuracy(model_filepath, batch_size, train_size, images_dir, labels_dir):
    model = _load_unet_model(model_filepath)

    train_dataset, test_dataset, val_dataset = prepare_datasets(batch_size, train_size, images_dir, labels_dir)

    train_loss, train_accuracy = model.evaluate(train_dataset, batch_size=batch_size)
    test_loss, test_accuracy = model.evaluate(test_dataset, batch_size=batch_size)
    validation_loss, validation_accuracy = model.evaluate(val_dataset, batch_size=batch_size)

    print(f'Model Accuracy on the Training Dataset: {round(train_accuracy * 100, 2)}%')
    print(f'Model Accuracy on the Validation Dataset: {round(validation_accuracy * 100, 2)}%')
    print(f'Model Accuracy on the Test Dataset: {round(test_accuracy * 100, 2)}%')


def display_model_predictions(model_filepath, batch_size, test_size, number_of_batches = 1, number_of_images = 3):
    loaded_model = _load_unet_model(model_filepath)
    plot_model_predictions(loaded_model, number_of_batches, number_of_images, batch_size, test_size)


def _load_unet_model(model_filepath):
    with custom_object_scope({'__masked_sparse_categorical_crossentropy': _masked_sparse_categorical_crossentropy}):
        model = load_model(model_filepath)
    return model
