import tensorflow_datasets as tfds
from base.data_preprocessing import BaseDataPreprocessing


def get_train_valid_data(config, preprocessing: BaseDataPreprocessing):
    d = _get_data(config, ["train", "validation"])
    d = {
        "train": d[0].map(preprocessing.preprocess_train),
        "valid": d[1].map(preprocessing.preprocess_test),
    }
    if config.data.shuffle:
        d["train"] = d["train"].shuffle(config.data.get("shuffle_buffer_size", 6000))
    return d


def get_test_data(config, preprocessing: BaseDataPreprocessing):
    return _get_data(config, "test").map(preprocessing.preprocess_test)


def _get_data(config, splits):
    data_dir = config.data.get("data_dir", "data")

    splits = tfds.load(
        "caltech_birds2011",
        data_dir=data_dir,
        batch_size=config.data.batch_size,
        split=splits,
        download=False
    )
    return splits
