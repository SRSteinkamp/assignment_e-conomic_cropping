import requests
import os
import zipfile
import shutil
import numpy as np
from tqdm.auto import tqdm
from .image_utils import load_image, create_mask
from skimage.io import imsave, imread

def get_data(url, out):
    # Helper function to download data
    if not os.path.isdir(out):
        r = requests.get(url, allow_redirects=True)
        with open(out + '.zip', 'wb') as f:
            f.write(r.content)
        # https://stackoverflow.com/questions/3451111/unzipping-files-in-python/3451150
        with zipfile.ZipFile(out + '.zip', 'r') as zip_ref:
            zip_ref.extractall(out)
    else:
        print("Data already present!")


def make_folder(folder_path):
    # Make folder if it doesn't exist
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    else:
        print('Folder exists!')


def make_set(set_csv, INPATH, OUTPATH, with_mask=False):
    # Create a training set, move files from csv to the respective folders
    set_csv = set_csv.copy()

    if with_mask:
        # Remove trailing slash
        MASKPATH = OUTPATH.rstrip('/') + '_mask/'
        make_folder(MASKPATH)

    for n, fname in tqdm(enumerate(set_csv.Filename)):

        shutil.copy(f'{INPATH}{fname}', OUTPATH)

        if with_mask:
            # Reshape array
            poly = set_csv.iloc[n][get_bbox_names()].values.reshape(-1, 2)
            poly = poly.tolist()
            poly.append(poly[0])
            poly = np.array(poly)

            # Load image, but get only shape
            img_shape = imread(f'{INPATH}{fname}').shape[:2]
            # Flip y-axis
            poly[:, 1] = img_shape[0] - poly[:, 1]

            mask = create_mask(poly[:, 1], poly[:, 0], img_shape) * 255

            imsave(f'{MASKPATH}{fname}', mask)

    # Reorder:
    set_csv = set_csv[['Filename', *get_bbox_names()]]

    # Add filepath
    if with_mask:
        mask_csv = set_csv['Filename'].copy()
        mask_csv = MASKPATH + mask_csv.loc[:, 'Filename']

    set_csv.loc[:, 'Filename'] = OUTPATH + set_csv.loc[:, 'Filename']
    tmp_dir = OUTPATH.split(os.sep)[0]
    tmp_name = OUTPATH.split(os.sep)[1]
    set_csv.to_csv(f'{tmp_dir}/{tmp_name}.csv', index=False)


def get_bbox_names():
    '''
    Returns the order and names of the bounding box.
    '''
    names =['TopLeftX','TopLeftY', 'TopRightX','TopRightY',
            'BottomRightX','BottomRightY', 'BottomLeftX','BottomLeftY']
    return names
