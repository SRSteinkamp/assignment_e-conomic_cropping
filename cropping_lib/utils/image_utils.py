from PIL import Image, ImageOps
import numpy as np
from skimage.draw import polygon, polygon2mask
from tensorflow import keras

def load_image(img_path):
    # Simply load a Pillow image, but convert to RGB
    return Image.open(img_path).convert('RGB')


def preprocess_image(image, target_size):
    '''
    Basic preprocessing of image, resizing and transforming to numpy array
    '''
    # For now using nearest filter for performance
    image = image.resize(target_size, Image.NEAREST)
    # doing the preprocessing for mobile net here.
    image = np.array(image) / 255
    image = -1 + (image * 2)
    return image


def scale_coords_down(coord_array, img_width, img_height):
    '''
    Transforming coordinates into proportions of the image size.
    Assuming eight coordinates in the order:
    TopLeftX, TopLeftY, TopRightX, TopRightY, BotRightX, BotRightY, BotLeftX,BotLeftY,
    the X coordinates will be devided by img_width
    the Y coordinates will be devided by img_height
    '''
    transformed_array = np.zeros(8)
    transformed_array[::2] = coord_array[::2] / img_width
    transformed_array[1::2] = coord_array[1::2] / img_height

    return transformed_array


def scale_coords_up(coords, width, height):
    '''
    Transforming proportions into coordinates of the image size.
    Assuming eight coordinates in the order:
    TopLeftX, TopLeftY, TopRightX, TopRightY, BotRightX, BotRightY, BotLeftX,BotLeftY,
    the X coordinates will be multiplied by img_width
    the Y coordinates will be multiplied by img_height
    '''
    resc_coords = np.zeros(8)
    resc_coords[::2] = coords[::2] * width
    resc_coords[1::2] = coords[1::2] * height

    return resc_coords


def create_mask(rows, cols, img_shape):
    '''
    Creates a binary mask based on bounding box
    '''
    # Draw a polygon (as not rectangular masks)
    rr, cc = polygon(rows, cols, img_shape)
    # Create mask
    mask = np.zeros(img_shape)
    mask[rr, cc] = 1

    return mask.astype('uint8')