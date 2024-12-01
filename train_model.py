import tensorflow as tf

from datetime import datetime
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_preprocessing import prepare_datasets
from plotting import plot_training_results
from unet import unet_model


def train_model(input_shape, filters, n_classes, epochs, batch_size, train_size, save_model,
                plot_training_summary, save_training_summary, images_dir, labels_dir):
    model = unet_model(input_shape, filters=filters, n_classes=n_classes)

    model.compile(optimizer='adam', loss=_masked_sparse_categorical_crossentropy, metrics=['accuracy'])
    callback = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=1e-1, patience=5, verbose=1, min_lr=2e-6)

    train_dataset, val_dataset, test_dataset = prepare_datasets(batch_size, train_size, images_dir, labels_dir)

    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[callback, reduce_lr],
                        shuffle=True)

    plot_training_results(history, save_training_summary) if plot_training_summary else None

    model_name = f"unet_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    model.save(model_name) if save_model else None

    return model_name


# Masked sparse categorical cross-entropy loss function.
# This function calculates the loss only for labeled pixels in the ground truth.
def _masked_sparse_categorical_crossentropy(y_true, y_pred):
    mask = tf.math.greater(y_true, 0)
    mask_squeezed = tf.squeeze(mask, axis=-1)

    y_true_masked = tf.boolean_mask(y_true, mask_squeezed)
    y_preds_list = [tf.boolean_mask(y_pred[..., i], mask_squeezed) for i in range(y_pred.shape[-1])]
    y_pred_masked = tf.stack(y_preds_list, axis=-1)

    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked)
    return tf.reduce_mean(loss)
