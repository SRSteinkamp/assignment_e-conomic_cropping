from tensorflow import keras
from ..utils import preprocess_image, load_image, scale_coords_down
from ..utils import scale_coords_up
import numpy as np
import os
from tqdm.auto import tqdm
from glob import glob


def make_prediction(images, model):
    """Makes a prediction given a list of imags. First creates the batch by
       preprocessing the images and in the end returns the prediction.

    Args:
        images ([list, PIL.Image]): List of PIL Images files, or a single PIL Image
        model ([keras.model]): A keras model, that can be used for predictions.

    Returns:
        [list]: Predictions of the model, in this case a len(images) x 8 batch.
    """
    image_batch = np.array([preprocess_image(im.copy(),
                                            target_size=(224,224))
                            for im in images])

    if image_batch.ndim == 3:
        image_batch = image_batch[np.newaxis, :, :, :]

    pred = model.predict(image_batch)

    return pred


def predict_files(FILEPATH, MODELPATH):
    """Searches for files in FILEPATH and load the model given by MODELPATH,
       runs make_predictions and returns the rescaled predictions and images.

    Args:
        FILEPATH ([str]): Path to the file, or folder of which the images are to
                          be taken.
        MODELPATH ([str]): Path to the keras model.

    Raises:
        IOError: If FILEPATH is not found.

    Returns:
        [np.array, list]: Returns the predictions for the images which were
                          which were provided in FILEPATH
    """
    model = keras.models.load_model(MODELPATH)

    if os.path.isfile(FILEPATH):
        images = [FILEPATH]

    elif os.path.isdir(FILEPATH):
        # Glob for .png, .jpeg, .jpg
        images = []
        for suff in ['.jpg', '.jpeg', '.png']:
            images.extend(glob(FILEPATH  + '/*' + suff))
    else:
        raise IOError(f"File or path: {FILEPATH} not found!")

    # Make sure images are ordered (for testing...)
    images = np.sort(images)
    images = [load_image(im) for im in images]

    predictions = make_prediction(images, model)
    pred_rescaled = []

    # Postprocess coordinates, return images
    for im, pre in zip(images, predictions):
        tmp_scu = scale_coords_up(pre.ravel(), im.width, im.height)
        pred_rescaled.append(tmp_scu)

    return pred_rescaled, images
