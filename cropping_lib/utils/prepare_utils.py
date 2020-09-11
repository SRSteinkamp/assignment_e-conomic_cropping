import requests
import os
import zipfile
import shutil
import numpy as np
from tqdm.auto import tqdm
from .image_utils import load_image, create_mask
from skimage.io import imsave, imread


def get_data(url, out):
    """Download the data from url, and unzip into out

    Args:
        url (str): Dowload location
        out (str): Unzip location.
    """
    if not os.path.isdir(out):
        r = requests.get(url, allow_redirects=True)
        with open(out + '.zip', 'wb') as f:
            f.write(r.content)
        # https://stackoverflow.com/questions/3451111/unzipping-files-in-python/3451150
        with zipfile.ZipFile(out + '.zip', 'r') as zip_ref:
            zip_ref.extractall(out)
    else:
        print("Data already present!")


def make_set(set_csv, set_name, INPATH, OUTPATH):
    """Creates and transform the data into a specific format.

    Args:
        set_csv (pd.Dataframe): Dataframe with bounding box information and Filenames.
        set_name (str): The set_name (e.g. train)
        INPATH (str): Where to find original images.
        OUTPATH (str): Where to copy the images.
    """
    set_csv = set_csv.copy()

    # Copy files to designated folder
    for n, fname in tqdm(enumerate(set_csv.Filename)):
        shutil.copy(f'{INPATH}{fname}', OUTPATH + f'/{set_name}/')

    # Reorder:
    set_csv = set_csv[['Filename', *get_bbox_names()]]

    # Add filepath
    set_csv.loc[:, 'Filename'] = (f'{OUTPATH}/{set_name}/'
                                  + set_csv.loc[:, 'Filename'])
    set_csv.to_csv(f'{OUTPATH}/{set_name}.csv', index=False)


def get_bbox_names():
    """Returns the order of the bounding boxes, as defined by the assignment.

    Returns:
        list: list of size 8, with the coordinate order. TopLeft, TopRight,
        BottomRight, BottomLeft.
    """
    names =['TopLeftX','TopLeftY', 'TopRightX','TopRightY',
            'BottomRightX','BottomRightY', 'BottomLeftX','BottomLeftY']
    return names


def check_working_dir(script_location):
    """Util to check whether the script is executed in the right directory, a
    bit hacky.

    Args:
        script_location (str): Location of the current script.

    Raises:
        EnvironmentError: If not running in repository folder, an error occurs.

    Returns:
        parent_dir: Parent directory of the script.
    """

    file_dir = os.path.dirname(script_location)
    parent_dir = os.path.dirname(file_dir)

    if os.getcwd() != parent_dir:
        raise EnvironmentError(f"Please make sure you execute script in "
                                 + parent_dir + "!")
    else:
        return parent_dir