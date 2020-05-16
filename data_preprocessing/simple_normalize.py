from base.data_preprocessing import BaseDataPreprocessing
import tensorflow as tf


def _normalize(input_image):
    input_image = tf.cast(input_image, tf.float64) / 127.5 - 1.
    return input_image


class SimpleNormalize(BaseDataPreprocessing):
    def __init__(self, config):
        super().__init__(config)

    def preprocess_config(self):
        self.config.model.input_shape = (*self.config.data_preprocessing.image_size, 3)
        self.config.model.num_classes = 34

    def preprocess(self, datapoint):
        image_size = tuple(self.config.data_preprocessing.image_size)
        input_image = tf.image.resize(datapoint['image_left'], image_size)
        input_mask = tf.image.resize(datapoint['segmentation_label'], image_size, method="nearest")

        input_image = _normalize(input_image)

        return input_image, input_mask