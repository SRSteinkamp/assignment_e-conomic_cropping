# %%
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from cropping_lib.model_parts import DataGenerator
# %%
train_csv = pd.read_csv('data/train.csv')
valid_csv = pd.read_csv('data/validation.csv')
# %%
train_generator = DataGenerator(train_csv, batch_size=20)
# %% If no model exists: Initiate model
base = keras.applications.MobileNetV2(input_shape=(224,224,3),
                                      include_top=False, weights='imagenet')
# %%
model = keras.Sequential([ base,
                           keras.layers.GlobalAveragePooling2D(),
                           keras.layers.Dense(30, activation='relu'),
                           keras.layers.Dropout(0.5),
                           keras.layers.Dense(8, activation='sigmoid')])
# %% Else: Load model, pass learning rate via CLI, epochs, callbacks:
# Load model

# Edit model name:
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss = 'mean_squared_error')
# %%
model.fit(train_generator, epochs=5)
# %% Print out validation score:
valid_generator = DataGenerator(valid_csv, batch_size=1, shuffle=False)
model.evaluate(valid_generator)

# %%
