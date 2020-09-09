# %%
import numpy as np
from tensorflow.keras.utils import Sequence
from ..utils import load_image, preprocess_image, scale_coords_down
from ..utils import scale_coords_up, get_bbox_names
from tensorflow_addons.losses import giou_loss
from tensorflow.keras.losses import MSE
import tensorflow as tf
# %%

class IOU_wrapper(tf.keras.losses.Loss):
    def __init__(self, box1_index=[7,6,3,2], box2_index=[5,0,1,4], **kwargs):
        self.box1_index = box1_index
        self.box2_index = box2_index
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        # IOU needs: [y_min, x_min, y_max, x_max]
        # Input is:
        # TopLeftX,TopLeftY,TopRightX,TopRightY,BottomRightX,BottomRightY,BottomLeftX,BottomLeftY
        #    0         1        2          3        4           5           6           7
        # IOU should perform better than MSE
        box1_true = tf.gather(y_true, self.box1_index, axis=1)
        box1_pred = tf.gather(y_pred, self.box1_index, axis=1)
        box2_true = tf.gather(y_true, self.box2_index, axis=1)
        box2_pred = tf.gather(y_pred, self.box2_index, axis=1)

        # Approximating corners of bounding box
        box1_loss = giou_loss(box1_true, box1_pred, mode='iou')
        box2_loss = giou_loss(box2_true, box2_pred, mode='iou')

        overall_loss = MSE(y_true, y_pred)
        return box1_loss + box2_loss + overall_loss

    def get_config(self):
        # based on handson ML 2 p.386, to save loss with model
        base_config = super().get_config()
        return {**base_config, "box1_index" : self.box1_index,
                               "box2_index" : self.box2_index}


class DataGenerator(Sequence):

    def __init__(self, image_csv, image_size=(224, 224), batch_size=1, shuffle=False):
        self.image_csv = image_csv
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.bbox_order = get_bbox_names()
        self.on_epoch_end()

    def __len__(self):
        # Calculate the lenght of the generator (i.e. number of batches in epoch)
        return int(np.floor(self.image_csv.shape[0]/ self.batch_size))

    def on_epoch_end(self):
        # Indices = number of files in there (even if it's df)
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
            coords = scale_coords_down(coords, width, height)
            coords = scale_coords_up(coords, self.image_size[0], self.image_size[1])

            X.append(img)
            y.append(coords)

        return np.array(X), np.array(y)