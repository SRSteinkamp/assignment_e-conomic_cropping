# %%
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from cropping_lib.model_parts import DataGenerator, iou_wrapper
# %%
LR = 0.001
EPOCHS = 100
train_csv = pd.read_csv('data/train.csv')
valid_csv = pd.read_csv('data/validation.csv')

# Data generators from training
train_generator = DataGenerator(train_csv, batch_size=20)
valid_generator = DataGenerator(valid_csv, batch_size=1, shuffle=False)
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
# Callbacks:
cb_logging = tf.keras.callbacks.CSVLogger('models/log.csv', separator=",", append=False)

cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='models',
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

cb_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

cb_reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5)

# Edit model name:
optimizer = keras.optimizers.Adam(learning_rate=LR)
model.compile(optimizer=optimizer, loss = iou_wrapper)
# %%
model.fit(train_generator, validation_data = valid_generator,
          epochs=EPOCHS,
          callbacks=[cb_logging, cb_checkpoint,
                     cb_earlystopping, cb_reduceLR])

# %%