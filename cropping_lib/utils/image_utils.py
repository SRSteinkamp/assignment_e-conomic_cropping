from PIL import Image
import numpy as np
from tensorflow import keras

def load_image(img_path):
    # Simply load a Pillow image
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

def transform_coords(coord_array, img_width, img_height):
    '''
    Transforming coordinates into proportions of the image size.
    Assuming eight coordinates in the order:
    TopLeftX,TopLeftY,TopRightX,TopRightY,BottomRightX,BottomRightY,BottomLeftX,BottomLeftY,
    the X coordinates will be devided by img_width
    the Y coordinates will be devided by img_height
    '''
    transformed_array = np.zeros(8)
    transformed_array[::2] = coord_array[::2] / img_width
    transformed_array[1::2] = coord_array[1::2] / img_height

    return transformed_array


def rescale_items(coords, width, height):
    '''
    Transforming proportions into coordinates of the image size.
    Assuming eight coordinates in the order:
    TopLeftX,TopLeftY,TopRightX,TopRightY,BottomRightX,BottomRightY,BottomLeftX,BottomLeftY,
    the X coordinates will be multiplied by img_width
    the Y coordinates will be multiplied by img_height
    '''
    resc_coords = np.zeros(8)
    resc_coords[::2] = coords[::2] * width
    resc_coords[1::2] = coords[1::2] * height

    return resc_coords