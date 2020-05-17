from base.data_preprocessing import BaseDataPreprocessing
from utils.types import Datapoint
from typing import Tuple
import tensorflow as tf


class NoOpPreprocessing(BaseDataPreprocessing):
    def preprocess(self, datapoint: Datapoint) -> Tuple[tf.Tensor, tf.Tensor]:
        return datapoint.image, datapoint.segmentation_mask
