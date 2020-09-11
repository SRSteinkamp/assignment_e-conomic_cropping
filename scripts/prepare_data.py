# %%
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from cropping_lib.utils import make_set, check_working_dir
# %% Setting up Paths
BASEPATH = check_working_dir(os.path.realpath(__file__))

PATH = BASEPATH + '/cropping_data/mobile_images_cropping_ds/'

# Bad boxes:
# IMG_1518.jpg < No content, but corners
# IMG_2146.jpg < bad cut of bounding boxes
# IMG_7086.jpg < half bounding box
# IMG_7909.jpg < half bounding box
# 4f9edd7aee90d73ad1e7eec6965988a3-image.jpg # Half bounding box
# 65f85a4512fb87931a0bbaa1a9289c30-image.jpg # Half bounding box

# Medium bad boxes
# IMG_6877.jpg < Bounding box to large
# IMG_9754.jpg < bounding box slightly to small

# Two receipts
# IMG_9837.jpeg
# 78ff591eccabd016385d6eabf54a0236-image.jpg
# f1fd2a6215e73c3656ec63254c3446a3-image.png

bad_images = ['IMG_1518.jpg', 'IMG_2146.jpg', 'IMG_7086.jpg', 'IMG_7909.jpg',
              '4f9edd7aee90d73ad1e7eec6965988a3-image.jpg',
              '65f85a4512fb87931a0bbaa1a9289c30-image.jpg',
              'IMG_6877.jpg', 'IMG_9754.jpg', 'IMG_9837.jpeg',
              '78ff591eccabd016385d6eabf54a0236-image.jpg',
              'f1fd2a6215e73c3656ec63254c3446a3-image.png']


# Reading data, adding a factor for stratification (fileype)
cropped_coords = pd.read_csv(f'{PATH}cropped_coords.csv')
cropped_coords['ftype'] = cropped_coords.Filename.str.split('.', expand=True)[1]

# Index of bad images:
bad_index = cropped_coords.Filename.isin(bad_images)
# Remove bad images:
cropped_coords = cropped_coords[bad_index == 0]
# %% There's definetely a more elegant way, relabel filetypes
for n, ii in enumerate(cropped_coords['ftype'].unique()):
    cropped_coords.loc[cropped_coords.ftype == ii, 'ftype'] = n

# Split data into train and test file
train_val, test = train_test_split(cropped_coords, stratify=cropped_coords['ftype'],
                                   test_size=0.1, random_state=9)
# Train and validation data
train, validation = train_test_split(train_val, stratify=train_val['ftype'],
                                     test_size=0.1, random_state=11)

# Move stratified files to corresponding folders
for set_csv, set_name in zip([train, test, validation],
                             ['train', 'test', 'validation']):
    # Create folder
    os.makedirs(BASEPATH + f'/data/{set_name}', exist_ok=True)
    make_set(set_csv, set_name, PATH, BASEPATH + '/data/')


# Move bad images to another folder as demonstrations set:
os.makedirs(BASEPATH + '/data/demonstration', exist_ok=True)

for fname in bad_images:
    shutil.copy(f'{PATH}{fname}', BASEPATH + '/data/demonstration')
