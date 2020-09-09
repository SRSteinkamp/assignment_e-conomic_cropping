#%%
#%load_ext autoreload
#%autoreload 2
import pandas as pd
from sklearn.model_selection import train_test_split
from cropping_lib.utils import get_data, make_folder, make_set
# %% Setting up Paths
PATH = 'cropping_data/mobile_images_cropping_ds/'

make_folder('data')

# Reading data, adding a factor for stratification (fileype)
cropped_coords = pd.read_csv(f'{PATH}cropped_coords.csv')
cropped_coords['ftype'] = cropped_coords.Filename.str.split('.', expand=True)[1]

# There's definetely a more elegant way, relabel filetypes
for n, ii in enumerate(cropped_coords['ftype'].unique()):
    cropped_coords.loc[cropped_coords.ftype==ii, 'ftype'] = n

# Split data into train and test file
train_val, test = train_test_split(cropped_coords, stratify=cropped_coords['ftype'],
                                   test_size=0.1, random_state=29)
# Train and validation data
train, validation = train_test_split(train_val, stratify=train_val['ftype'],
                                     test_size=0.1, random_state=9)

# Move stratified files to corresponding folders
for set_csv, set_name in zip([train, test, validation],
                             ['train', 'test', 'validation']):

    make_folder(f'data/{set_name}')
    make_set(set_csv, PATH, f'data/{set_name}/', with_mask=True)