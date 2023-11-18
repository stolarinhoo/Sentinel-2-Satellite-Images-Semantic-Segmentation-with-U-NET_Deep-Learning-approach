import tensorflow as tf

from datetime import datetime
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_preprocessing import prepare_datasets
from plotting import plot_training_results
from unet import unet_model


def train_model(input_shape: tuple[int, int, int], filters: int, n_classes: int, epochs: int, batch_size: int,
                test_size: float, save_model: bool = True, print_model_summary: bool = False,
                plot_training_summary: bool = True, save_training_summary: bool = False) -> str:
    model = unet_model(input_shape, filters=filters, n_classes=n_classes, print_summary=print_model_summary)

    model.compile(optimizer='adam', loss=__masked_sparse_categorical_crossentropy, metrics=['accuracy'])
    callback = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=1e-1, patience=5, verbose=1, min_lr=2e-6)

    train_dataset, val_dataset, test_dataset = prepare_datasets(batch_size, test_size)

    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[callback, reduce_lr],
                        shuffle=True)

    plot_training_results(history, save_training_summary) if plot_training_summary else None

    model_name = f"unet_model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.keras"
    model.save(model_name) if save_model else None

    return model_name


# Masked sparse categorical cross-entropy loss function.
# This function calculates the loss only for labeled pixels in the ground truth.
def __masked_sparse_categorical_crossentropy(y_true, y_pred):
    mask = tf.math.greater(y_true, 0)
    mask_squeezed = tf.squeeze(mask, axis=-1)

    y_true_masked = tf.boolean_mask(y_true, mask_squeezed)
    y_preds_list = [tf.boolean_mask(y_pred[..., i], mask_squeezed) for i in range(y_pred.shape[-1])]
    y_pred_masked = tf.stack(y_preds_list, axis=-1)

    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked)
    return tf.reduce_mean(loss)
