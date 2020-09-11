# %%
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from cropping_lib.utils import get_bbox_names
from cropping_lib.model_parts import predict_files
from tqdm.auto import tqdm
import argparse


def main(IMAGEPATH, MODELPATH):
    rescaled, images = predict_files(IMAGEPATH, MODELPATH)
    print(rescaled)

    for n, im in tqdm(enumerate(images)):

        draw = ImageDraw.Draw(im)
        # Draw the predicted box
        poly_test = rescaled[n].reshape(-1, 2).tolist()
        poly_test.append(poly_test[0])
        poly_test = np.array(poly_test)
        poly_test[:, 1] = im.height - poly_test[:, 1]
        # Draw the prediction
        draw.line(tuple(map(tuple, poly_test)), fill='green', width=3)

        if os.path.isdir(IMAGEPATH):
            im.save(f'{IMAGEPATH}/img_{n}_bbox.png')
        elif os.path.isfile(IMAGEPATH):
            im.save(f'{os.path.dirname(IMAGEPATH)}/img{n}_bbox.png')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        dest='path_in', help='Path to the Image(s)', type=str)
    parser.add_argument(dest='path_model', help='Location of the model', type=str)


    args = parser.parse_args()
    main(args.path_in, args.path_model)