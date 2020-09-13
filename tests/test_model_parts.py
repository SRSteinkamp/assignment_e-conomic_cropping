from cropping_lib.model_parts import DataGenerator, IOU_TwoBox, CombinedLoss
from cropping_lib.model_parts import build_model, IOU_LargeBox, CenterLoss
from cropping_lib.model_parts import build_model_mobilenet
import pandas as pd
import tensorflow as tf
import numpy as np


def test_smoke_DataGenerator():
    # check if DataGenerator can be created
    df = pd.DataFrame(np.arange(10))
    DataGenerator(df)


def test_IOU_TwoBox():
    # Test if iou_wrapper provides results:
    # TopLeftX,TopLeftY,TopRightX,TopRightY,BottomRightX,BottomRightY,BottomLeftX,BottomLeftY
    boxes1 = tf.constant([[0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
                          [0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1]])

    assert np.isclose(0, np.sum(IOU_TwoBox()(boxes1, boxes1)))


def test_IOU_LargeBox():
    # Test if iou_wrapper provides results:
    boxes1 = tf.constant([[0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
                          [0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1]])

    assert np.isclose(0, np.sum(IOU_LargeBox()(boxes1, boxes1)))


def test_CenterLoss():
    # Test if iou_wrapper provides results:
    boxes1 = tf.constant([[0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
                          [0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1]])

    assert np.isclose(0, np.sum(CenterLoss()(boxes1, boxes1)))


def test_CombinedLoss():
    # Test if iou_wrapper provides results:
    boxes1 = tf.constant([[0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
                          [0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1]])

    assert np.isclose(0, np.sum(CombinedLoss()(boxes1, boxes1)))


def test_smoke_model():
    bn = build_model(weights=None)
    bn.compile(optimizer='sgd', loss='mean_squared_error')


def test_smoke_model_mobile():
    bn = build_model_mobilenet(weights=None)
    bn.compile(optimizer='sgd', loss='mean_squared_error')