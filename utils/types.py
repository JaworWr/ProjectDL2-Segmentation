import tensorflow as tf
from dataclasses import dataclass


@dataclass
class Datapoint:
    filename: tf.Tensor
    image: tf.Tensor
    segmentation_mask: tf.Tensor
