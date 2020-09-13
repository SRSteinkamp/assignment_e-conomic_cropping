from ..utils import load_image, preprocess_image, scale_coords_down
from ..utils import get_bbox_names
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_addons.losses import giou_loss


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


class CenterLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        """ Loss function optimizing the center of the largest bounding box.
        """
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """Selects the largest bounding box of the predictions and the
           ground truth and calculates the iou_loss between them. And adds the
           MAE to it (to optimize the other coordinates).

        Args:
            y_true (tf.tensor): Ground truth
            y_pred (tf.tensor): Predictions

        Returns:
            [tf.tensor]: IOU loss for largest bounding box
        """
        # Bounding box for pred
        x_min_pred = tf.reduce_min(y_pred[:, ::2], axis=1)
        x_max_pred = tf.reduce_max(y_pred[:, ::2], axis=1)
        y_min_pred = tf.reduce_min(y_pred[:, 1::2], axis=1)
        y_max_pred = tf.reduce_max(y_pred[:, 1::2], axis=1)
        # Bounding box for truth
        x_min_true = tf.reduce_min(y_true[:, ::2], axis=1)
        x_max_true = tf.reduce_max(y_true[:, ::2], axis=1)
        y_min_true = tf.reduce_min(y_true[:, 1::2], axis=1)
        y_max_true = tf.reduce_max(y_true[:, 1::2], axis=1)
        # Calculate boxes
        x_mid_pred = x_min_pred + (x_max_pred - x_min_pred) / 2
        y_mid_pred = y_min_pred + (y_max_pred - y_min_pred) / 2
        x_mid_true = x_min_true + (x_max_true - x_min_true) / 2
        y_mid_true = y_min_true + (y_max_true - y_min_true) / 2
        # Approximating corners of bounding box

        center_pred = tf.stack([x_mid_pred, y_mid_pred], axis=1)
        center_true = tf.stack([x_mid_true, y_mid_true], axis=1)

        center_loss = keras.losses.MSE(center_true, center_pred)

        return center_loss


class IOU_LargeBox(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        """ Loss function optimizing the largest possible bounding box in the
            image.
        """
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """Selects the largest bounding box of the predictions and the
           ground truth and calculates the iou_loss between them. And adds the
           MAE to it (to optimize the other coordinates).

        Args:
            y_true (tf.tensor): Ground truth
            y_pred (tf.tensor): Predictions

        Returns:
            [tf.tensor]: IOU loss for largest bounding box
        """
        x_min_pred = tf.reduce_min(y_pred[:, ::2], axis=1)
        x_max_pred = tf.reduce_max(y_pred[:, ::2], axis=1)
        y_min_pred = tf.reduce_min(y_pred[:, 1::2], axis=1)
        y_max_pred = tf.reduce_max(y_pred[:, 1::2], axis=1)

        box_pred = tf.stack([y_min_pred, x_min_pred,
                             y_max_pred, x_max_pred], axis=1)
        # Bounding box for truth
        x_min_true = tf.reduce_min(y_true[:, ::2], axis=1)
        x_max_true = tf.reduce_max(y_true[:, ::2], axis=1)
        y_min_true = tf.reduce_min(y_true[:, 1::2], axis=1)
        y_max_true = tf.reduce_max(y_true[:, 1::2], axis=1)

        box_true = tf.stack([y_min_true, x_min_true,
                             y_max_true, x_max_true], axis=1)

        # Approximating corners of bounding box
        iou = giou_loss(box_true, box_pred, mode='giou')

        return iou


class IOU_TwoBox(tf.keras.losses.Loss):
    def __init__(self, box1_index=[7, 6, 3, 2], box2_index=[5, 0, 1, 4], **kwargs):
        """Uses the eight coordinates to create two bounding boxes, based on
        bottom left, and top right, as well as bottom right and top left. The
        loss for theses bounding boxes is then calculated.

        Args:
            box1_index (list, optional): Indices for bounding box 1. Defaults to [7,6,3,2].
            box2_index (list, optional): Indices for bounding box 2. Defaults to [5,0,1,4].
        """
        self.box1_index = box1_index
        self.box2_index = box2_index
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """Calculate sum of the two bounding boxes.

        Args:
            y_true (tf.tensor): Ground truth
            y_pred (tf.tensor): Predictions

        Returns:
            [tf.tensor]: The loss calculated for the two bounding boxes.
        """
        box1_true = tf.gather(y_true, self.box1_index, axis=1)
        box1_pred = tf.gather(y_pred, self.box1_index, axis=1)
        # Get box 2
        box2_true = tf.gather(y_true, self.box2_index, axis=1)
        box2_pred = tf.gather(y_pred, self.box2_index, axis=1)

        # Bounding Boxes
        box1_loss = giou_loss(box1_true, box1_pred, mode='giou')
        box2_loss = giou_loss(box2_true, box2_pred, mode='giou')

        box_loss = (box1_loss + box2_loss) / 2

        return (box_loss + IOU_LargeBox()(y_true, y_pred) +
                CenterLoss()(y_true, y_pred) +
                tf.keras.losses.MSE(y_true, y_pred))

    def get_config(self):
        # based on handson ML 2 p.386, to save loss with model
        base_config = super().get_config()
        return {**base_config, "box1_index": self.box1_index,
                               "box2_index": self.box2_index}


class DataGenerator(keras.utils.Sequence):

    def __init__(self, image_csv, image_size=(224, 224), batch_size=1, shuffle=False):
        """Data generator for validation and training.

        Args:
            image_csv (pd.DataFrame): Dataframe with absolute path to images in
                                      "Filename", and bounding boxe locations.
            image_size (tuple, optional): Target size of images. Defaults to (224, 224).
            batch_size (int, optional): Batch size. Defaults to 1.
            shuffle (bool, optional): Whether to shuffle before creating batches. Defaults to False.
        """
        self.image_csv = image_csv
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.bbox_order = get_bbox_names()
        self.on_epoch_end()

    def __len__(self):
        # Calculate the lenght of the generator (i.e. number of batches in epoch)
        return int(np.floor(self.image_csv.shape[0] / self.batch_size))

    def on_epoch_end(self):
        # Indices = number of files in there (even if it's df)
        self.indexes = np.arange(self.image_csv.shape[0])

        # shuffle, if wanted
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # Create indices (shuffled)
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        X, y = self.__data_generation(indexes)

        return X, y

    def __data_generation(self, indexes):
        X, y = [], []

        # Generate data
        for idx in indexes:
            # Load image, save dimensions and resize:
            img = load_image(self.image_csv.iloc[idx]['Filename'])
            height = img.height
            width = img.width
            img = preprocess_image(img, self.image_size)
            # Transform bbox coordinates
            coords = self.image_csv.iloc[idx][self.bbox_order].values.ravel()
            # normalize
            coords = scale_coords_down(coords, width, height)
            X.append(img)
            y.append(coords)

        return np.array(X), np.array(y)
