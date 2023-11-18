import rasterio

from typing import Union
from rasterio.plot import reshape_as_image
from sklearn.model_selection import train_test_split
from tensorflow.python.types.data import DatasetV1, DatasetV2

from constants import INPUT_SHAPE, LABEL_SHAPE
from utils import get_file_paths_for_images_and_labels

import tensorflow as tf


def prepare_datasets(batch_size: int, test_size: float):
    (train_image_paths, val_image_paths, test_image_paths,
     train_label_paths, val_label_paths, test_label_paths) = __split_data_for_training_testing_and_validation(test_size)

    train_dataset = __create_tf_datasets(train_image_paths, train_label_paths)
    test_dataset = __create_tf_datasets(test_image_paths, test_label_paths)
    val_dataset = __create_tf_datasets(val_image_paths, val_label_paths)
    train_dataset, val_dataset, test_dataset = __preprocess_datasets(
        train_dataset, val_dataset, test_dataset, batch_size)

    return train_dataset, val_dataset, test_dataset


def __split_data_for_training_testing_and_validation(test_size: float) -> tuple[
    list[str], list[str], list[str], list[str], list[str], list[str]
]:
    image_paths, label_paths = get_file_paths_for_images_and_labels()
    train_image_paths, test_image_paths, train_label_paths, test_label_paths = __train_test_split_paths(
        image_paths, label_paths, test_size)

    val_image_paths, test_image_paths, val_label_paths, test_label_paths = __train_test_split_paths(
        test_image_paths, test_label_paths, test_size)

    print(f'There are {len(train_image_paths)} images in the Training Set')
    print(f'There are {len(val_image_paths)} images in the Validation Set')
    print(f'There are {len(test_image_paths)} images in the Test Set')
    return train_image_paths, val_image_paths, test_image_paths, train_label_paths, val_label_paths, test_label_paths


def __train_test_split_paths(image_paths: list[str], label_paths: list[str], train_size: float) -> tuple[
    list[str], list[str], list[str], list[str]
]:  # todo; add "random_state" or "shuffle" parameters
    train_image_paths, test_image_paths, train_label_paths, test_label_paths = train_test_split(
        image_paths, label_paths, train_size=train_size, random_state=0)
    return train_image_paths, test_image_paths, train_label_paths, test_label_paths


def __create_tf_datasets(image_paths: list[str], label_paths: list[str]) -> Union[DatasetV1, DatasetV2]:
    image_tf_list = tf.constant(image_paths)
    label_tf_list = tf.constant(label_paths)
    return tf.data.Dataset.from_tensor_slices((image_tf_list, label_tf_list))


# Load and preprocess an image-label pair given their paths
def __load_and_preprocess(image_path: any, label_path: any):
    with (rasterio.open(image_path.numpy().decode(), 'r') as img_src,
          rasterio.open(label_path.numpy().decode(), 'r') as lbl_src):
        img = img_src.read()
        img = reshape_as_image(img)
        img = tf.image.convert_image_dtype(img, tf.float32)

        lbl = lbl_src.read()
        lbl = reshape_as_image(lbl)
        lbl = tf.math.reduce_max(lbl, axis=-1, keepdims=True)
        lbl = tf.cast(lbl, tf.uint16)

    return img, lbl


# Wrapper function for using 'load_and_preprocess' with TensorFlow operations
def __load_and_preprocess_wrapper(image_path, label_path):
    image, label = tf.py_function(func=__load_and_preprocess, inp=[image_path, label_path],
                                  Tout=(tf.float32, tf.uint16))
    image.set_shape(INPUT_SHAPE)
    label.set_shape(LABEL_SHAPE)
    return image, label


# Preprocess and batch the training, testing, and validation datasets
def __preprocess_datasets(train_dataset, val_dataset, test_dataset, batch_size):
    train_dataset = train_dataset.map(__load_and_preprocess_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.map(__load_and_preprocess_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.map(__load_and_preprocess_wrapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset, test_dataset
