from tensorflow import keras


def build_model_mobilenet(weights='imagenet', dropout=0.25):
    """
    Creates the model used in the training loop.
    Args:
        weights (str, optional): Whether to use imagenet pretraiing or no weights. Defaults to 'imagenet'.
        dropout (float, optional): Dropout in the last layer. Defaults to 0.25.
    Returns:
        [keras model]: The un-compiled model.
    """

    base = keras.applications.MobileNetV2(input_shape=(224,224,3),
                                            include_top=False, weights=weights)

    model = keras.Sequential([base,
                              keras.layers.Conv2D(filters=128, kernel_size=(3,3)),
                              keras.layers.BatchNormalization(),
                              keras.layers.Activation('relu'),
                              keras.layers.Conv2D(filters=64, kernel_size=(3,3)),
                              keras.layers.BatchNormalization(),
                              keras.layers.Activation('relu'),
                              keras.layers.Conv2D(filters=32, kernel_size=(3,3)),
                              keras.layers.Activation('relu'),
                              keras.layers.Dropout(dropout),
                              keras.layers.Conv2D(filters=8, kernel_size=(1,1)),
                              keras.layers.Dropout(dropout),
                              keras.layers.Activation('relu'),
                              keras.layers.Flatten(),
                              keras.layers.Dense(8, activation='sigmoid')
                              ])

    return model


def build_model(weights='imagenet', dropout=0.25):
    """Creates the model used in the training loop.

    Args:
        weights (str, optional): Whether to use imagenet pretraiing or no weights. Defaults to 'imagenet'.
        dropout (float, optional): Dropout in the last layer. Defaults to 0.25.

    Returns:
        keras model: The un-compiled model.
    """
    #base = keras.applications.MobileNetV2(input_shape=(224, 224, 3),
    #                                      include_top=False, weights=weights)
    def simple_block(fs, ks, inps):
        block = [keras.layers.Conv2D(fs, kernel_size=(ks, ks), padding='same',
                 input_shape = (inps, inps, 3), kernel_initializer='he_normal'),
                 keras.layers.BatchNormalization(),
                 keras.layers.LeakyReLU(),
                 keras.layers.MaxPool2D()]

        return block

    base_block = []

    for ks, fs, inps in zip([7, 5, 3, 3, 3, 3, 3],
                            [8, 16, 32, 64, 128, 128, 256],
                            [224, 112, 56, 28, 14 ,7, 3]):
        base_block.extend(simple_block(fs, ks, inps))

    model = keras.Sequential([*base_block,
                              keras.layers.Flatten(),
                              keras.layers.Dense(25, activation='relu'),
                              keras.layers.Dropout(dropout),
                              keras.layers.Dense(8, activation='sigmoid')])

    print(model.summary())
    return model
