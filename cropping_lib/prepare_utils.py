import requests
import os
import zipfile
import shutil
from tqdm.auto import tqdm


def get_data(url, out):
    # Helper function to download data
    if not os.path.isfile(out):
        r = requests.get(url, allow_redirects=True)
        with open(out + '.zip', 'wb') as f:
            f.write(r.content)
        # https://stackoverflow.com/questions/3451111/unzipping-files-in-python/3451150
        with zipfile.ZipFile(out + '.zip', 'r') as zip_ref:
            zip_ref.extractall(out)

def make_folder(folder_path):
    # Make folder if it doesn't exist
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    else:
        print('Folder exists!')

def make_set(set_csv, INPATH, OUTPATH):
    # Create a training set, move files from csv to the respective folders

    for fname in tqdm(set_csv.Filename):
        shutil.copy(f'{INPATH}{fname}', OUTPATH)

    # Drop ftype column
    set_csv.drop('ftype', axis=1)
    tmp_dir = OUTPATH.split(os.sep)[0]
    tmp_name = OUTPATH.split(os.sep)[1]
    set_csv.to_csv(f'{tmp_dir}/{tmp_name}.csv')
