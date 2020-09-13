# %%
import os
import pandas as pd
from PIL import ImageDraw
import numpy as np
from cropping_lib.utils import get_bbox_names, create_mask, check_working_dir
from cropping_lib.model_parts import predict_files
from tqdm.auto import tqdm
from sklearn.metrics import jaccard_score
# %%
BASEPATH = check_working_dir(os.path.realpath(__file__))
MODELPATH = BASEPATH + '/model/mobilenetv2/'
EVALPATH = BASEPATH + f'/evaluation_images/{MODELPATH.split("/")[-2]}'
test_csv = pd.read_csv(BASEPATH + '/data/test.csv').sort_values('Filename')

os.makedirs(EVALPATH, exist_ok=True)

rescaled, images = predict_files(BASEPATH + '/data/test/', MODELPATH)

evaluation_dict = {'Img': [], 'IOU': []}
# %%
for n, im in tqdm(enumerate(images)):
    # Reshape true coordinates
    poly = test_csv[get_bbox_names()].iloc[n].values.reshape(-1, 2).tolist()
    poly.append(poly[0])
    poly = np.array(poly)
    # Flip coordinates
    poly[:, 1] = im.height - poly[:, 1]
    draw = ImageDraw.Draw(im)
    # Draw the ground truth
    draw.line(tuple(map(tuple, poly)), fill='red', width=5)

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

    true_mask = create_mask(poly[:, 1], poly[:, 0], (im.height, im.width))
    test_mask = create_mask(poly_test[:, 1], poly_test[:, 0], (im.height, im.width))

    score = jaccard_score(true_mask.flatten(), test_mask.flatten())

    evaluation_dict['Img'].append(test_csv.iloc[n]['Filename'])
    evaluation_dict['IOU'].append(score)
    im.save(f'{EVALPATH}/{test_csv.iloc[n]["Filename"].split("/")[-1]}')

evaluation_dict = pd.DataFrame(evaluation_dict)
evaluation_dict.to_csv(f'{EVALPATH}/scores.csv')
