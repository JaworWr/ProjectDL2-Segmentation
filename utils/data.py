from base.data_preprocessing import BaseDataPreprocessing
import tensorflow as tf
import os


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
    image = tf.image.decode_jpeg(image)
    label = tf.image.decode_png(label)

    # TODO: colors into labels
    return {
        "filename": filename,
        "image": image,
        "segmentation_mask": label,
    }