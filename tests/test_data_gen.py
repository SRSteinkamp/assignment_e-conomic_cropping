from cropping_lib.model_parts import DataGenerator, IOU_wrapper
import pandas as pd
import tensorflow as tf
import numpy as np


def test_smoke_DataGenerator():
    # check if DataGenerator can be created
    df = pd.DataFrame(np.arange(10))
    DataGenerator(df)


def test_iou_wrapper():
    # Test if iou_wrapper provides results:
    # TopLeftX,TopLeftY,TopRightX,TopRightY,BottomRightX,BottomRightY,BottomLeftX,BottomLeftY
    boxes1 = tf.constant([[0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
                          [0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1]])

    assert np.isclose(0, np.sum(IOU_wrapper()(boxes1, boxes1)))