import pytest
import os
import numpy as np
from cropping_lib.utils import scale_coords_up, preprocess_image
from cropping_lib.utils import scale_coords_down, load_image, create_mask


def test_load_image():
    # Just a smoke test
    tmp = load_image(os.path.dirname(__file__) + '/files/test_img.jpg')


def test_scale_coords_down():
    # Test if inputs are unaltered
    assert np.alltrue(np.arange(8) == scale_coords_down(np.arange(8), 1, 1))

    # Test width application
    test_array = np.ones(8)
    test_array[::2] = 0.5
    assert np.allclose(test_array, scale_coords_down(np.ones(8), 2, 1))

    # Test height application
    test_array = np.ones(8)
    test_array[1::2] = 0.5
    assert np.allclose(test_array, scale_coords_down(np.ones(8), 1, 2))


def test_scale_coords_up():
    # Test if inputs are unaltered
    assert np.alltrue(np.arange(8) == scale_coords_up(np.arange(8), 1, 1))

    # Test width application
    test_array = np.ones(8)

    test_array[::2] = 2
    assert np.allclose(test_array, scale_coords_up(np.ones(8), 2, 1))

    # Test height application
    test_array = np.ones(8)
    test_array[1::2] = 2
    assert np.allclose(test_array, scale_coords_up(np.ones(8), 1, 2))


def test_preprocess_image():
    # Check if image has been resized, is a numpy array and has the correct shape
    tmp = load_image(os.path.dirname(__file__) + '/files/test_img.jpg')
    test_img = preprocess_image(tmp, (224, 224))

    assert np.allclose(test_img.shape, [224, 224, 3])

    # Test for scaling
    assert np.max([1, np.max(test_img)]) == 1
    assert np.min([0, np.min(test_img)]) == 0


def test_create_mask():
    # Draw the mask, with a single entry in middle (drawing polygon)
    mask = create_mask([1, 1, 2, 2],
                       [1, 2, 1, 2], (3,3))

    assert np.allclose(mask.shape, (3,3))
    assert mask[1, 1] == 1
    assert mask.sum() == 1