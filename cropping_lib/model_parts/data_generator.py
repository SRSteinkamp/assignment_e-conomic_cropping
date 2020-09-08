import numpy as np
from tensorflow.keras.utils import Sequence
from ..utils import load_image, preprocess_image, transform_coords

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