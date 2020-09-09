# %%
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from PIL import Image
from cropping_lib.model_parts import DataGenerator, IOU_wrapper, build_model
from cropping_lib.utils import make_folder
# %%
LR = 0.001
EPOCHS = 100
DATA_PATH ='data/'
MODEL_PATH = 'model/20200909_mobilenet/'
MODEL_NAME = 'model/20200909_mobilenet/'
train_csv = pd.read_csv(f'{DATA_PATH}/train.csv')
valid_csv = pd.read_csv(f'{DATA_PATH}/validation.csv')

# Data generators from training
train_generator = DataGenerator(train_csv, batch_size=25)
valid_generator = DataGenerator(valid_csv, batch_size=7, shuffle=False)
# %% If no model exists: Initiate model
if True: #not os.path.isdir('model/base/'):
    make_folder('model')
    model = build_model(weights='imagenet', dropout=0.5)
elif os.path.isdir(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH, compile=False)
else:
    # Adjust to load most recent model
    print("Did not find model path, loading base model")
    model = keras.models.load_model('model/base', compile=False)

# %% Else: Load model, pass learning rate via CLI, epochs, callbacks:
# Callbacks:
cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='model',
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

cb_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

cb_reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5)

# Edit model name:
optimizer = keras.optimizers.Adam(learning_rate=LR)
model.compile(optimizer=optimizer, loss = 'mean_absolute_error')
# %%
model.fit(train_generator, validation_data = valid_generator,
          epochs=EPOCHS,
          callbacks=[cb_checkpoint, cb_earlystopping, cb_reduceLR])

# A bit hacky
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.save(MODEL_NAME)