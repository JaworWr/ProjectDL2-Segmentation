import tensorflow_datasets as tfds


def get_train_valid_data(config):
    d = _get_data(config, ["train", "validation"])
    d = {
        "train": d[0],
        "valid": d[1],
    }
    if config.data.shuffle:
        d["train"] = d["train"].shuffle(config.data.get("shuffle_buffer_size", 6000))
    return d


def get_test_data(config):
    return _get_data(config, "test")


def _get_data(config, splits):
    data_dir = config.data.get("data_dir", "data")
    download_dir = config.data.get("download_dir", None)
    extract_dir = config.data.get("extract_dir", None)

    splits = tfds.load(
        "caltech_birds2011",
        data_dir=data_dir,
        batch_size=config.data.batch_size,
        split=splits,
        download_and_prepare_kwargs=dict(
            download_dir=extract_dir,
            download_config=tfds.download.DownloadConfig(
                manual_dir=download_dir
            )
        )
    )
    return splits
