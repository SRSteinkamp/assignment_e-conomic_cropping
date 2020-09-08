import pytest
import os
import numpy as np
from cropping_lib.utils import get_data, load_image, transform_coords, preprocess_image


def test_load_image():
    # Just a smoke test
    tmp = load_image(os.path.dirname(__file__) + '/files/test_img.jpg')


def test_transform_coords():
    # Test if inputs are unaltered
    assert np.alltrue(np.arange(8) == transform_coords(np.arange(8), 1, 1))

    # Test width application
    test_array = np.ones(8)
    test_array[::2] -= 0.5
    assert np.allclose(test_array, transform_coords(np.ones(8), 2, 1))

    # Test height application
    test_array = np.ones(8)
    test_array[1::2] -= 0.5
    assert np.allclose(test_array, transform_coords(np.ones(8), 1, 2))


def test_preprocess_image():
    # Check if image has been resized, is a numpy array and has the correct shape
    tmp = load_image(os.path.dirname(__file__) + '/files/test_img.jpg')
    test_img = preprocess_image(tmp, (224, 224))

    assert np.allclose(test_img.shape, [224, 224, 3])
