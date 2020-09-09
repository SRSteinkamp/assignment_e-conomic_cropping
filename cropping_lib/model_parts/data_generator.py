# %%
import numpy as np
from tensorflow.keras.utils import Sequence
from ..utils import load_image, preprocess_image, transform_coords
from tensorflow_addons.losses import giou_loss
import tensorflow as tf
# %%

def iou_wrapper(y_true, y_pred):
    # IOU needs: [y_min, x_min, y_max, x_max]
    # Input is:
    # TopLeftX,TopLeftY,TopRightX,TopRightY,BottomRightX,BottomRightY,BottomLeftX,BottomLeftY
    #    0         1        2          3        4           5           6           7
    # IOU should perform better than MSE
    box1_true = tf.gather(y_true, [7, 6, 3, 2], axis=1)
    box1_pred = tf.gather(y_pred, [7, 6, 3, 2], axis=1)
    box2_true = tf.gather(y_true, [5, 0, 1, 4], axis=1)
    box2_pred = tf.gather(y_pred, [5, 0, 1, 4], axis=1)

    # Approximating corners of bounding box
    box1_loss = giou_loss(box1_true, box1_pred)
    box2_loss = giou_loss(box2_true, box2_pred)

    return box1_loss + box2_loss


class DataGenerator(Sequence):

    def __init__(self, image_csv, image_size=(224, 224), batch_size=1, shuffle=False):
        self.image_csv = image_csv
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.bbox_order = ['TopLeftX', 'TopLeftY', 'TopRightX', 'TopRightY',
                           'BottomRightX', 'BottomRightY', 'BottomLeftX',
                           'BottomLeftY']
        self.on_epoch_end()

    def __len__(self):
        # Calculate the lenght of the generator (i.e. number of batches in epoch)
        return int(np.floor(self.image_csv.shape[0]/ self.batch_size))

    def on_epoch_end(self):
        # Indices = number of files in there (even if it's csv)
        self.indexes = np.arange(self.image_csv.shape[0])

        # shuffle, if wanted
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # Create indices (shuffled)
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

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
            coords = transform_coords(coords, width, height)

            X.append(img)
            y.append(coords)

        return np.array(X), np.array(y)