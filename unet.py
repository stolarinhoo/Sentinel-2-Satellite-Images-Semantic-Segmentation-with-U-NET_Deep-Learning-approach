from keras.layers import Activation, Input, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, concatenate
from keras.models import Model
from tensorflow import Tensor


def __encoding_block(inputs: Tensor, filters: int, max_pooling=True):
    C = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(inputs)
    C = BatchNormalization()(C)
    C = Activation("relu")(C)

    C = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(C)
    C = BatchNormalization()(C)
    C = Activation("relu")(C)
    skip_connection = C

    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(C)
    else:
        next_layer = C
    return next_layer, skip_connection


def __decoding_block(inputs: Tensor, skip_connection_input: int, filters: int):
    CT = Conv2DTranspose(filters, 3, strides=(2, 2), padding="same", kernel_initializer="he_normal")(inputs)
    residual_connection = concatenate([CT, skip_connection_input], axis=3)
    C = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(residual_connection)
    C = BatchNormalization()(C)
    C = Activation("relu")(C)
    C = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(C)
    C = BatchNormalization()(C)
    C = Activation("relu")(C)
    return C


def unet_model(input_size: tuple[int, int, int], filters: int, n_classes: int, print_summary: bool = False):
    inputs = Input(input_size)

    C1, S1 = __encoding_block(inputs, filters, max_pooling=True)
    C2, S2 = __encoding_block(C1, filters * 2, max_pooling=True)
    C3, S3 = __encoding_block(C2, filters * 4, max_pooling=True)
    C4, S4 = __encoding_block(C3, filters * 8, max_pooling=True)
    C5, S5 = __encoding_block(C4, filters * 16, max_pooling=False)

    U6 = __decoding_block(C5, S4, filters * 8)
    U7 = __decoding_block(U6, S3, filters * 4)
    U8 = __decoding_block(U7, S2, filters * 2)
    U9 = __decoding_block(U8, S1, filters)

    C10 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(U9)
    C11 = Conv2D(filters=n_classes, kernel_size=(1, 1), activation='softmax',  padding='same')(C10)
    model = Model(inputs=inputs, outputs=C11)

    model.summary() if print_summary else None
    return model
