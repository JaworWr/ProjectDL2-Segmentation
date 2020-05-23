from utils.types import Datapoint
from typing import Tuple
import tensorflow as tf


class BaseDataPreprocessing:
    def __init__(self, config):
        self.config = config

    def preprocess_config(self):
        pass

    def preprocess_train_valid(self, datapoint: Datapoint) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        raise NotImplementedError

    def preprocess_train(self, datapoint: Datapoint) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return self.preprocess_train_valid(datapoint)

    def preprocess_valid(self, datapoint: Datapoint) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return self.preprocess_train_valid(datapoint)

    def preprocess_test(self, datapoint: Datapoint) -> Tuple[tf.Tensor]:
        raise NotImplementedError
