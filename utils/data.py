from base.data_preprocessing import BaseDataPreprocessing
import os


def _get_filenames(path, split):
    path = os.path.join(path, "ImageSets", "Segmentation", split + ".txt")
    with open(path) as f:
        filenames = [l.strip() for l in f.readlines()]
    return filenames
