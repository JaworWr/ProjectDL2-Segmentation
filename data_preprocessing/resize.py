from base.data_preprocessing import BaseDataPreprocessing
from utils.types import Datapoint
from utils.data import N_CLASSES
from typing import Tuple
import tensorflow as tf


class ResizePreprocessing(BaseDataPreprocessing):
    def __init__(self, config):
        super().__init__(config)
        self.n_pixels = config.data_preprocessing.image_size[0] * config.data_preprocessing.image_size[1]

    def preprocess_config(self):
        self.config.model.input_shape = (*self.config.data_preprocessing.image_size, 3)
        self.config.model.num_classes = N_CLASSES

    def preprocess_train_valid(self, datapoint: Datapoint) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        image_size = tuple(self.config.data_preprocessing.image_size)
        input_image = tf.image.resize(datapoint.image, image_size)
        input_mask = tf.image.resize(datapoint.segmentation_mask, image_size, method="nearest")

        input_mask = tf.reshape(input_mask, (-1, 22))

        mask_indices = tf.argmax(input_mask, 1)
        weights = tf.ones_like(mask_indices)
        return (input_image, input_mask, weights)

    def preprocess_test(self, datapoint: Datapoint) -> Tuple[tf.Tensor]:
        return (datapoint.image,)
