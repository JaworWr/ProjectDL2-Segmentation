from base.data_preprocessing import BaseDataPreprocessing
import tensorflow as tf
import os
import numpy as np


N_CLASSES=21


def get_train_valid_data(config, preprocessing: BaseDataPreprocessing):
    pass


def _get_filenames(root, split):
    path = os.path.join(root, "ImageSets", "Segmentation", split + ".txt")
    with open(path) as f:
        filenames = [l.strip() for l in f.readlines()]
    return filenames


def _create_dataset(root, filenames):
    def gen():
        for filename in filenames:
            yield _load_sample(root, filename)

    return tf.data.Dataset.from_generator(
        gen,
        {
            "filename": tf.string,
            "image": tf.uint8,
            "segmentation_mask": tf.uint8,
        },
    )


def _load_sample(root, filename):
    image_path = os.path.join(root, "JPEGImages", filename + ".jpg")
    label_path = os.path.join(root, "SegmentationClass", filename + ".png")
    image = tf.io.read_file(image_path)
    label = tf.io.read_file(label_path)
    image = tf.image.decode_jpeg(image, channels=3)
    label = tf.image.decode_png(label, channels=3)

    cmap = color_map()
    label = tf.equal(label[:, :, None, :], cmap[None, None, :, :])
    label = tf.reduce_all(label, axis=3)

    return {
        "filename": filename,
        "image": image,
        "segmentation_mask": label,
    }


def color_map(n_classes=N_CLASSES, N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])
    return cmap[list(range(n_classes)) + [-1], :]
