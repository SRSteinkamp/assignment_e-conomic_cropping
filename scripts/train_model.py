# %%
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from cropping_lib.utils import check_working_dir
from cropping_lib.model_parts import DataGenerator, IOU_TwoBox
from cropping_lib.model_parts import IOU_LargeBox, build_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint

# %% Training settings
LR = 0.01 # Learning Rate
EPOCHS = 5000 # Number of Epochs
BASEPATH = check_working_dir(os.path.realpath(__file__))
DATAPATH = BASEPATH + '/data/' # Data Location
MODELPATH = BASEPATH + '/model/mobilenetv2/' # The model to load
MODELNAME = BASEPATH + '/model/mobilenetv2/' # The model to save
train_csv = pd.read_csv(f'{DATAPATH}/train.csv')
valid_csv = pd.read_csv(f'{DATAPATH}/validation.csv')

# Data generators for training and validation
train_generator = DataGenerator(train_csv, batch_size=20, shuffle=True)
valid_generator = DataGenerator(valid_csv, batch_size=1, shuffle=False)

# If the model does not exist: Initiate model, else reload previous model
if not os.path.isdir(MODELPATH):
    os.makedirs(BASEPATH + '/model', exist_ok=True)
    model = build_model(weights=None, dropout=0.25)
elif os.path.isdir(MODELPATH):
    model = keras.models.load_model(MODELPATH, compile=False)
else:
    raise IOError('Provide the correct model path')

# Prepare callbacks: Checkpoint to iteratively save model (just in case)
cb_checkpoint = ModelCheckpoint(
                                filepath= MODELPATH,
                                save_weights_only=False,
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True)
# Early stopping to prevent overfitting
cb_earlystopping = EarlyStopping(monitor='val_loss', patience=20)
# There is more fancy stuff out there, let's just leave it with this
cb_reduceLR = ReduceLROnPlateau(monitor='val_loss', patience=5)

# Compile model
optimizer = keras.optimizers.Nadam(learning_rate=LR)
model.compile(optimizer=optimizer, loss=IOU_LargeBox(),
              metrics=['mean_absolute_error'])

# Fit
history = model.fit(train_generator, validation_data = valid_generator,
          epochs=EPOCHS,
          callbacks=[cb_checkpoint, cb_earlystopping, cb_reduceLR])

# Recompile model with tf.keras.loss function, so that there are no issues
# when later loading the model without custom losses
model.compile(optimizer=optimizer, loss='mean_squared_error')
# And finally save the model
model.save(MODELNAME)
# %% Create figure from model history
fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].plot(history.history['loss'])
axes[0].plot(history.history['val_loss'])
axes[0].legend(['loss', 'validation loss'])
axes[0].set_title('Loss')
axes[1].plot(history.history['mean_absolute_error'])
axes[1].plot(history.history['val_mean_absolute_error'])
axes[1].legend(['MAE', 'validation MAE'])
axes[1].set_title('Error')
plt.savefig(MODELNAME + '/loss_curve.png')