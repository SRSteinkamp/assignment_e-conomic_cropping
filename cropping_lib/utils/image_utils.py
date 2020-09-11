from PIL import Image, ImageOps
import numpy as np
from skimage.draw import polygon, polygon2mask
from tensorflow import keras

def load_image(img_path):
    """Load Pillow image and transform to RGB

    Args:
        img_path (str): Image path

    Returns:
        [PIL.Image]: The loaded image
    """
    return Image.open(img_path).convert('RGB')


def preprocess_image(image, target_size):
    """Basic preprocessing for training. Resizing and scaling the image,
    then creating numpy array of it.

    Args:
        image (PIL.Image): Unprocessed PIL.Image
        target_size (tuple): Target size of the image.

    Returns:
        np.array: Scaled and resized image as np.array
    """
    # For now using nearest filter for performance
    image = image.resize(target_size, Image.NEAREST)
    # doing the preprocessing for mobile net here.
    image = np.array(image) / 255
    image = -1 + (image * 2)
    return image


def scale_coords_down(coord_array, img_width, img_height):
    """Normalizing coordinates, by deviding x-coordinates by width, and y
    coordinates by height.

    Args:
        coord_array (np.array, list): 1D numpy array, with coordinates in
                                      the restricted order.
        img_width (int): Image width
        img_height (int): Image height

    Returns:
        np.array: np.array of size 8, with the scaled coordinates.
    """
    transformed_array = np.zeros(8)
    transformed_array[::2] = coord_array[::2] / img_width
    transformed_array[1::2] = coord_array[1::2] / img_height

    return transformed_array


def scale_coords_up(coords, width, height):
    """Scaling coordinates up, by multiplying x-coordinates by width, and y
    coordinates by height.

    Args:
        coord_array (np.array, list): 1D numpy array, with coordinates in
                                      the restricted order.
        img_width (int): Image width
        img_height (int): Image height

    Returns:
        np.array: np.array of size 8, with the scaled coordinates.
    """
    resc_coords = np.zeros(8)
    resc_coords[::2] = coords[::2] * width
    resc_coords[1::2] = coords[1::2] * height

    return resc_coords


def create_mask(rows, cols, img_shape):
    """Create a binary mask from bounding boxes, assuming that each values
    represents a corner of the bounding box.

    Args:
        rows (list, np.array): Polygon rows.
        cols (list, np.array): Polygon columns.
        img_shape (list, tuple): Target image shape.

    Returns:
        np.array: Binary mask of size img_shape.
    """
    # Draw a polygon (as not rectangular masks)
    rr, cc = polygon(rows, cols, img_shape)
    # Create mask
    mask = np.zeros(img_shape)
    mask[rr, cc] = 1

    return mask.astype('uint8')