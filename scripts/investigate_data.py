# %%
import pandas as pd
from PIL import ImageDraw, Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from cropping_lib.utils import get_bbox_names, check_working_dir, create_mask
# %%
BASEPATH = check_working_dir(os.path.realpath(__file__))

# Create folders to save the results
os.makedirs(BASEPATH + '/cropping_data/check_boxes', exist_ok=True)
# Create ftype
box_data = pd.read_csv(BASEPATH + '/cropping_data/mobile_images_cropping_ds/cropped_coords.csv')
box_data['ftype'] = box_data.Filename.str.split('.', expand=True)[1]

# Run over all images, draw ground truth, and store image dimension
for idx in tqdm(box_data.index):

    im = Image.open(BASEPATH + '/cropping_data/mobile_images_cropping_ds/'
                             + box_data.loc[idx, 'Filename'])

    # XY to rows
    poly = box_data.loc[idx, get_bbox_names()].values.reshape(-1, 2).tolist()
    poly.append(poly[0])
    # Flip y-Axis
    poly = np.array(poly)
    poly[:, 1] = im.height - poly[:, 1]
    draw = ImageDraw.Draw(im)
    # Draw the ground truth
    draw.line(tuple(map(tuple, poly)), fill='red', width=5)

    # Store additional information
    mask = create_mask(poly[:, 1], poly[:, 0], (im.height, im.width))
    box_data.loc[idx, 'Height'] = im.height
    box_data.loc[idx, 'Width'] = im.width
    box_data.loc[idx, 'Area'] = mask.sum()
    im.save(BASEPATH + '/cropping_data/check_boxes/'
                     + box_data.loc[idx, 'Filename'])

# Calculate some intermediate steps
box_data['Aspect'] = box_data['Height'] / box_data['Width']
box_data['BoxProportion'] = (box_data['Area'] /
                             (box_data['Width'] * box_data['Height']))

# Save results
box_data.to_csv(BASEPATH + '/cropping_data/augmented_coords.csv')
# Plot histogram of Statistics
box_data.hist(figsize=(20, 20))
plt.savefig(BASEPATH + '/cropping_data/image_stats.png', bins=20)
