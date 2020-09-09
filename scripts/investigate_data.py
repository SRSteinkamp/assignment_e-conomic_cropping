# %%
import pandas as pd
from PIL import ImageDraw, Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from cropping_lib.utils import make_folder, create_mask, get_data
# %%
URL = 'https://github.com/e-conomic/hiring-assignments/raw/master/machinelearningteam/receipt-cropping/cropping_data.zip'

get_data(URL, 'cropping_data')

# Create folders to save the results
make_folder('cropping_data')
make_folder('cropping_data/check_boxes')
# %%
box_data = pd.read_csv('cropping_data/mobile_images_cropping_ds/cropped_coords.csv')
# %%
box_data['ftype'] = box_data.Filename.str.split('.', expand=True)[1]
# Get the tupl for corner pairs
coordinate_tuples =[['TopLeftX','TopLeftY'], ['TopRightX','TopRightY'],
                    ['BottomRightX','BottomRightY'], ['BottomLeftX','BottomLeftY']]
#%%
# Run over all images, draw ground truth, and store image dimension
for rw in tqdm(box_data.index):

    im = Image.open('cropping_data/mobile_images_cropping_ds/'
                    + box_data.loc[rw, 'Filename'])

    # Get image coordinates
    poly = []
    for cord in coordinate_tuples:
        poly.append((box_data.loc[rw, cord].values))
    poly.append(poly[0])

    # Flip y-Axis
    poly = np.array(poly)
    poly[:, 1] = im.height - poly[:, 1]
    draw = ImageDraw.Draw(im)
    # Draw the ground truth
    draw.line(tuple(map(tuple, poly)), fill='red',  width=5)

    # Store additional information
    msk = create_mask(poly[:, 1], poly[:, 0], (im.height, im.width))
    box_data.loc[rw, 'Height'] = im.height
    box_data.loc[rw, 'Width'] = im.width
    box_data.loc[rw, 'Area'] = msk.sum()
    im.save(f'cropping_data/check_boxes/' + box_data.loc[rw, 'Filename'])

# Calculate some intermediate steps
box_data['Aspect'] = box_data['Height'] / box_data['Width']
box_data['BoxProportion'] = (box_data['Area'] /
                            (box_data['Width'] * box_data['Height']))

# Save results
box_data.to_csv('cropping_data/augmented_coords.csv')
# Plot histogram of Statistics
box_data.hist(figsize=(20,20))
plt.savefig('cropping_data/image_stats.png', bins=20)