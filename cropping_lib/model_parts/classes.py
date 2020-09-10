# %%
from ..utils import load_image, preprocess_image, scale_coords_down
from ..utils import scale_coords_up, get_bbox_names
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_addons.losses import giou_loss


def build_model(weights='imagenet', dropout=0.25):
    # Create the model in the training loop
    base = keras.applications.MobileNetV2(input_shape=(224,224,3),
                                      include_top=False, weights=weights)
    model = keras.Sequential([base,
                              keras.layers.Conv2D(filters=32, kernel_size=(3,3,),
                                                  activation='relu'),
                              keras.layers.Conv2D(filters=24, kernel_size=(3,3,),
                                                  activation='relu'),
                              keras.layers.BatchNormalization(),
                              keras.layers.Conv2D(filters=12, kernel_size=(3,3,),
                                                  activation='relu'),
                              keras.layers.Dropout(dropout),
                              keras.layers.Conv2D(filters=8, kernel_size=(1,1,),
                                                  activation='sigmoid'),
                              keras.layers.Flatten()])

    return model


class IOU_LargeBox(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        # IOU needs: [y_min, x_min, y_max, x_max]
        # Extract min and max bounding boxes for prediction
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

    def get_config(self):
        # based on handson ML 2 p.386, to save loss with model
        base_config = super().get_config()
        return {**base_config}


class IOU_TwoBox(tf.keras.losses.Loss):
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
        # Get box1
        box1_true = tf.gather(y_true, self.box1_index, axis=1)
        box1_pred = tf.gather(y_pred, self.box1_index, axis=1)
        # Get box 2
        box2_true = tf.gather(y_true, self.box2_index, axis=1)
        box2_pred = tf.gather(y_pred, self.box2_index, axis=1)

        # Bounding Boxes
        box1_loss = giou_loss(box1_true, box1_pred, mode='giou')
        box2_loss = giou_loss(box2_true, box2_pred, mode='giou')

        return (box1_loss + box2_loss) / 3 + IOU_LargeBox()(y_true, y_pred) / 3

    def get_config(self):
        # based on handson ML 2 p.386, to save loss with model
        base_config = super().get_config()
        return {**base_config, "box1_index" : self.box1_index,
                               "box2_index" : self.box2_index}


class DataGenerator(keras.utils.Sequence):

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
            # normalize
            coords = scale_coords_down(coords, width, height)

            X.append(img)
            y.append(coords)

        return np.array(X), np.array(y)