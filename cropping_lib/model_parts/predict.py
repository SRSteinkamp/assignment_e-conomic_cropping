from tensorflow import keras
from ..utils import preprocess_image, load_image, scale_coords_down
from ..utils import scale_coords_up
import numpy as np
import os
from tqdm.auto import tqdm
from glob import glob


def make_prediction(images, model):
    # Create image batch, and preprocess images
    image_batch = np.array([preprocess_image(im.copy(),
                                            target_size=(224,224))
                            for im in images])

    if image_batch.ndim == 3:
        image_batch = image_batch[np.newaxis, :, :, :]

    pred = model.predict(image_batch)

    return pred


def predict_files(FILEPATH, MODELPATH):
    # Load model
    model = keras.models.load_model(MODELPATH)

    if os.path.isfile(FILEPATH):
        images = [FILEPATH]

    elif os.path.isdir(FILEPATH):
        # Glob for .png, .jpeg, .jpg
        images = []
        for suff in ['.jpg', '.jpeg', '.png']:
            images.extend(glob(FILEPATH  + '/*' + suff))
    else:
        raise IOError(f"File or path {FILEPATH} not found!")

    # Make sure images are in alphabetical order (for testing...)
    images = np.sort(images)
    images = [load_image(im) for im in images]

    predictions = make_prediction(images, model)
    pred_rescaled = []

    # Preprocess coordinates, return images
    for im, pre in zip(images, predictions):
        tmp = scale_coords_down(pre.ravel(), 224, 224)
        pred_rescaled.append(scale_coords_up(tmp, im.width, im.height))

    return pred_rescaled, images
