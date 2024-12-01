import random
import rasterio
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.image import AxesImage

from keras.callbacks import History
from keras.models import Model
from rasterio import DatasetReader
from datetime import datetime

from data_preprocessing import prepare_datasets
from constants import LAND_COVER_CLASSES
from utils import get_file_paths_for_images_and_labels, get_number_of_images_and_labels
matplotlib.use('Qt5Agg')


def plot_images_and_labels(number_images_to_plot, save, images_dir, labels_dir):
    image_paths, label_paths = get_file_paths_for_images_and_labels(images_dir, labels_dir)
    number_of_samples, _ = get_number_of_images_and_labels(images_dir, labels_dir)

    for img in range(number_images_to_plot):
        random_number = random.randint(0, number_of_samples - 1)
        print(f"Displaying {random_number}th image and label")

        image_path = image_paths[random_number]
        label_path = label_paths[random_number]

        fig, arr = plt.subplots(1, 3, figsize=(20, 8))

        with rasterio.open(image_path) as image, rasterio.open(label_path) as label:
            _print_image_metadata(image) if img == 0 else None
            image_rgb = _s2_image_to_rgb(image)
            label = label.read(1)

            arr[0].imshow(X=image_rgb)
            arr[0].set_title('Real Image')

            im1 = arr[1].imshow(label)
            arr[1].set_title('Label')
            patches1 = _get_patches(im1, label)
            arr[1].legend(handles=patches1, bbox_to_anchor=(1.0, -0.06), loc=0, borderaxespad=0.)

            im2 = arr[2].imshow(label, cmap='Accent')
            arr[2].set_title('Label Overlay')
            patches2 = _get_patches(im2, label)
            arr[2].legend(handles=patches2, bbox_to_anchor=(1.0, -0.06), loc=0, borderaxespad=0.)

            _save_plot(plt, f"{random_number}th_image{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}") if save else None
    plt.show()


def plot_training_results(history, save = False):
    acc = [0.] + history.history['accuracy']
    val_acc = [0.] + history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    _save_plot(plt, f"unet_training_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}") if save else None
    plt.show()


def plot_model_predictions(model, number_of_batches, number_of_images, batch_size, train_size, save = False):
    train_dataset, test_dataset, val_dataset = prepare_datasets(batch_size, train_size)

    for image, label in val_dataset.take(number_of_batches):
        predictions = model.predict(image)
        predicted_classes = np.argmax(predictions, axis=-1)

        for i in range(number_of_images):
            fig, arr = plt.subplots(1, 3, figsize=(20, 8))

            image_rgb = image[i][:, :, 3:0:-1] * 10 + 0.15
            arr[0].imshow(image_rgb)
            arr[0].set_title('Real Image')

            im1 = arr[1].imshow(label[i])
            arr[1].set_title('Label')
            ptch1 = _get_patches(im1, label[i])
            arr[1].legend(handles=ptch1, bbox_to_anchor=(1.0, -0.06), loc=0, borderaxespad=0.)

            im2 = arr[2].imshow(predicted_classes[i])
            arr[2].set_title('Predicted label')
            ptch2 = _get_patches(im2, predicted_classes[i])
            arr[2].legend(handles=ptch2, bbox_to_anchor=(1.0, -0.06), loc=0, borderaxespad=0.)

            _save_plot(plt, f"{model}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}") if save else None
            plt.show()


def _s2_image_to_rgb(image):
    result_array = np.array([image.read(4), image.read(3), image.read(2)], dtype=float)
    result_array = np.transpose(result_array, (1, 2, 0))
    result_array = (result_array / 10000) + 0.15  # 0.15 added to brighten up the image
    return result_array


def _print_image_metadata(image):
    print(f"Shape: width x height x depth")
    print(f"\t\t{image.width} x {image.height} x {image.count}")
    print(f"CRS: {image.crs}")


def _get_patches(im, image):
    unique_values = _get_unique_values(image)
    cmap_colors = _get_cmap_colors(im, image)
    cmap = _get_cmap(unique_values, cmap_colors)
    classes = LAND_COVER_CLASSES
    return [mpatches.Patch(color=cmap[i], label=classes[i]) for i in cmap]


def _get_unique_values(image):
    return np.unique(image)


def _get_cmap_colors(im, image):
    return im.cmap(im.norm(_get_unique_values(image)))


def _get_cmap(values, colors):
    result = None
    if len(values) == len(colors):
        result = {values[i]: colors[i] for i in range(len(values))}
    return result


def _save_plot(_plt, title):
    image_format = "png"
    _plt.savefig(fname=f"{title}.{image_format}", format=image_format)
