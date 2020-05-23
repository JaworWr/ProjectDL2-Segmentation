from base.data_preprocessing import BaseDataPreprocessing
from utils.types import Datapoint
from utils.data import N_CLASSES
from typing import Tuple
import tensorflow as tf


def _normalize(input_image):
    input_image = tf.cast(input_image, tf.float64) / 127.5 - 1.
    return input_image

class SimpleNormalize(BaseDataPreprocessing):
    def __init__(self, config):
        super().__init__(config)

    def preprocess_config(self):
        self.config.model.input_shape = (*self.config.data_preprocessing.image_size, 3)
        self.config.model.num_classes = N_CLASSES

    def preprocess_train_valid(self, datapoint: Datapoint) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        image_size = tuple(self.config.data_preprocessing.image_size)
        input_image = tf.image.resize(datapoint.image, image_size)
        input_mask = tf.image.resize(datapoint.segmentation_mask, image_size, method="nearest")

        input_image = _normalize(input_image)
        input_mask = tf.reshape(input_mask, (-1, 22))

        mask_indices = tf.argmax(input_mask, 1)
        # ignore last label
        input_weights = tf.where(
            mask_indices == N_CLASSES-1,
            0.,
            1.,
        )

        # optionally set background weight
        if "background_class_weight" in self.config.data_preprocessing:
            input_weights = tf.where(
                mask_indices == 0,
                self.config.data_preprocessing.background_class_weight,
                input_weights
            )

        return input_image, input_mask, input_weights

    def preprocess_test(self, datapoint: Datapoint) -> Tuple[tf.Tensor]:
        image_size = tuple(self.config.data_preprocessing.image_size)
        input_image = tf.image.resize(datapoint.image, image_size)
        input_image = _normalize(input_image)
        return (input_image,)
