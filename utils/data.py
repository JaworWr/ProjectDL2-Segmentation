import tensorflow_datasets as tfds
from base.data_preprocessing import BaseDataPreprocessing


def get_train_valid_data(config, preprocessing: BaseDataPreprocessing):

    def preprocess_train(datapoint):
        return preprocessing.preprocess_train(datapoint)

    def preprocess_test(datapoint):
        return preprocessing.preprocess_test(datapoint)

    d = _get_data(config, ["train[:90%]", "train[90%:]"],
                  shuffle_files=config.data.get("shuffle", False))
    d = [d[0].map(preprocess_train), d[1].map(preprocess_test)]
    d = [s.prefetch(config.data.get("prefetch_buffer", 1)) for s in d]
    return {"train": d[0], "valid": d[1]}


def get_test_data(config, preprocessing: BaseDataPreprocessing):

    def preprocess_test(datapoint):
        return preprocessing.preprocess_test(datapoint)

    return _get_data(config, "validation") \
        .map(preprocess_test) \
        .prefetch(config.data.get("prefetch_buffer", 1))


def _get_data(config, splits, shuffle_files=False):
    data_dir = config.data.get("data_dir", "data")

    splits = tfds.load(
        "cityscapes",
        data_dir=data_dir,
        batch_size=config.data.batch_size,
        split=splits,
        download=False,
        as_dataset_kwargs=dict(
            shuffle_files=shuffle_files
        ),
    )
    return splits
