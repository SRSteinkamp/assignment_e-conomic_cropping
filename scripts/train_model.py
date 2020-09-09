# %%
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from PIL import Image
from cropping_lib.model_parts import DataGenerator, IOU_wrapper, build_model
from cropping_lib.utils import make_folder
# %% Training settings
LR = 0.001 # Learning Rate
EPOCHS = 100 # Number of Epochs
DATAPATH ='data/' # Data Location
MODELPATH = 'model/mobilenetv2/' # The model to load
MODELNAME = 'model/mobilenetv2/' # The model to save
train_csv = pd.read_csv(f'{DATAPATH}/train.csv')
valid_csv = pd.read_csv(f'{DATAPATH}/validation.csv')

# Data generators for training and validation
train_generator = DataGenerator(train_csv, batch_size=25)
valid_generator = DataGenerator(valid_csv, batch_size=7, shuffle=False)

# If the model does not exist: Initiate model, else reload previous model
if not os.path.isdir(MODELPATH):
    make_folder('model')
    model = build_model(weights='imagenet', dropout=0.5)
elif os.path.isdir(MODELPATH):
    model = keras.models.load_model(MODELPATH, compile=False)
else:
    raise IOError('Provide the correct model path')

# Prepare callbacks: Checkpoint to iteratively save model (just in case)
cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='model',
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
# Early stopping to prevent overfitting
cb_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# There is more fancy stuff out there, let's just leave it with this
cb_reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5)

# Compile model
optimizer = keras.optimizers.Adam(learning_rate=LR)
model.compile(optimizer=optimizer, loss = 'mean_absolute_error')
# Fit
model.fit(train_generator, validation_data = valid_generator,
          epochs=EPOCHS,
          callbacks=[cb_checkpoint, cb_earlystopping, cb_reduceLR])

# Recompile model with tf.keras.loss function, so that there are no issues
# when later loading the model without custom losses
model.compile(optimizer=optimizer, loss='mean_squared_error')
# And finally save the model
model.save(MODELNAME)