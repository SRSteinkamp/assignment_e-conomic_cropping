from tensorflow import keras


def build_model(weights='imagenet', dropout=0.25):
    # Create the model in the training loop
    base = keras.applications.MobileNetV2(input_shape=(224,224,3),
                                      include_top=False, weights=weights)
    model = keras.Sequential([base,
                              keras.layers.GlobalAveragePooling2D(),
                              keras.layers.Dense(50, activation='relu'),
                              keras.layers.Dropout(dropout),
                              keras.layers.Dense(8, activation='relu')])

    return model