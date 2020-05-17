from base.data_preprocessing import BaseDataPreprocessing
import tensorflow as tf
import os
import numpy as np
from utils.types import Datapoint
from dataclasses import dataclass
from typing import Sequence, Dict, Callable, Tuple

DATASET_SIZE = 2913

"""
Splits: 70%/15%/15%
"""
SUBSET_SIZES = {
    "train": int(0.7 * DATASET_SIZE),
    "valid": int(0.15 * DATASET_SIZE),
    "test": DATASET_SIZE - int(0.7 * DATASET_SIZE) - int(0.15 * DATASET_SIZE)
}


def color_map(n):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    indices = set(range(n)) | {255}
    cmap = np.zeros((len(indices), 3), dtype=np.uint8)
    for i in indices:
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        if i == 255:
            i = -1
        cmap[i] = np.array([r, g, b])
    return cmap


N_CLASSES = 21
CMAP = color_map(N_CLASSES)


def cmap_to_one_hot(img):
    label = tf.equal(img[:, :, None, :], CMAP[None, None, :, :])
    label = tf.reduce_all(label, axis=3)
    label = tf.cast(label, tf.uint8)
    return label


def indices_to_cmap(indices):
    return tf.gather(CMAP, indices, axis=0)


@dataclass
class Split:
    split: Callable[[tf.data.Dataset], tf.data.Dataset]
    preprocessing: Callable[[Datapoint], Tuple[tf.Tensor, tf.Tensor]]


def get_train_valid_data(config, preprocessing: BaseDataPreprocessing) -> Dict[str, tf.data.Dataset]:
    root = os.path.join(config.data.get("data_dir", "data"), "VOCdevkit", "VOC2012")
    splits = {
        "train": Split(
            split=lambda ds: ds.take(SUBSET_SIZES["train"]),
            preprocessing=lambda datapoint: preprocessing.preprocess_train(datapoint),
        ),
        "valid": Split(
            split=lambda ds: ds.skip(SUBSET_SIZES["train"]).take(SUBSET_SIZES["valid"]),
            preprocessing=lambda datapoint: preprocessing.preprocess_test(datapoint),
        ),
    }
    dataset = _create_dataset(
        root,
        _get_filenames(root, "trainval"),
        splits,
        config.data.get("workers", None),
    )
    if config.data.shuffle:
        dataset["train"] = dataset["train"].shuffle(config.data.get("shuffle_buffer_size", SUBSET_SIZES["train"]))
    if config.data.prefetch:
        dataset = {k: ds.prefetch(config.data.get("prefetch_buffer_size", 50)) for k, ds in dataset.items()}
    return dataset


def get_test_data(config, preprocessing: BaseDataPreprocessing) -> Dict[str, tf.data.Dataset]:
    root = os.path.join(config.data.get("data_dir", "data"), "VOCdevkit", "VOC2012")
    splits = {"test": Split(
        split=lambda ds: ds.skip(SUBSET_SIZES["train"] + SUBSET_SIZES["valid"]),
        preprocessing=lambda datapoint: preprocessing.preprocess_test(datapoint),
    )}
    dataset = _create_dataset(
        root,
        _get_filenames(root, "trainval"),
        splits,
        config.data.get("workers"),
    )
    if config.data.prefetch:
        dataset["test"] = dataset["test"].prefetch(config.data.get("prefetch_buffer_size", 10))
    return dataset


def _get_filenames(root, split):
    path = os.path.join(root, "ImageSets", "Segmentation", split + ".txt")
    with open(path) as f:
        filenames = [line.strip() for line in f.readlines()]
    return filenames


def _create_dataset(
        root: str,
        filenames: Sequence[str],
        splits: Dict[str, Split],
        workers: int
) -> Dict[str, tf.data.Dataset]:
    def gen():
        yield from filenames

    dataset = tf.data.Dataset.from_generator(gen, output_types=tf.string)
    split_datasets = {}
    for name, s in splits.items():
        split_datasets[name] = s.split(dataset)

        def load_and_preprocess(filename):
            datapoint = _load_sample(root, filename)
            return s.preprocessing(datapoint)

        split_datasets[name] = split_datasets[name].map(load_and_preprocess, num_parallel_calls=workers)
    return split_datasets


def _load_sample(root: str, filename: tf.Tensor) -> Datapoint:
    image_path = tf.strings.join([root, "JPEGImages", filename + ".jpg"], separator=os.sep)
    label_path = tf.strings.join([root, "SegmentationClass", filename + ".png"], separator=os.sep)
    image = tf.io.read_file(image_path)
    label = tf.io.read_file(label_path)
    image = tf.image.decode_jpeg(image, channels=3)
    label = tf.image.decode_png(label, channels=3)

    label = cmap_to_one_hot(label)

    return Datapoint(filename, image, label)
