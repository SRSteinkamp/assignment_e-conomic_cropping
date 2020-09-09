# %%
from predict import predict_files
from cropping_lib.utils import make_folder, get_bbox_names
from cropping_data.predict import make_prediction
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tqdm.auto as tqdm

MODELPATH = 'model/20200909_mobilenet/'
EVALPATH = f'test_evaluation/{MODELPATH.split("/")[1]}'
make_folder(EVALPATH)

rescaled, images = predict_files('data/test/', MODELPATH)

test_csv = pd.read_csv('data/test.csv').sort_values('Filename')

for n, im in tqdm(enumerate(images)):
    # Reshape true coordinates
    poly = test_csv[get_bbox_names()].iloc[n].values.reshape(-1, 2).tolist()
    poly.append(poly[0])
    poly = np.array(poly)
    # Flip coordinates
    poly[:, 1] = im.height - poly[:, 1]
    draw = ImageDraw.Draw(im)
    # Draw the ground truth
    draw.line(tuple(map(tuple, poly)), fill='red',  width=5)

    # Draw the predicted box
    poly_test = rescaled[n].reshape(-1, 2).tolist()
    poly_test.append(poly_test[0])
    poly_test = np.array(poly_test)
    poly_test[:, 1] = im.height - poly_test[:, 1]
    # Draw the prediction
    draw.line(tuple(map(tuple, poly_test)), fill='green', width=3)
    # Store image in list
    images[n] = im
    # Save image
    im.save(f'{EVALPATH}{test_csv.iloc[n]["Filename"].split("/")[-1]}')