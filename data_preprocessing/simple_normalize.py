from base.data_preprocessing import BaseDataPreprocessing
import tensorflow as tf


class SimpleNormalize(BaseDataPreprocessing):
    @staticmethod
    def normalize(input_image):
        input_image = tf.cast(input_image, tf.float64) / 127.5 - 1.
        return input_image

    def preprocess(self, datapoint):
        image_size = tuple(self.config.data_preprocessing.image_size)
        input_image = tf.image.resize(datapoint['image_left'], image_size)
        input_mask = tf.image.resize(datapoint['segmentation_label'], image_size, method="nearest")

        input_image = self.normalize(input_image)

        return input_image, input_mask
