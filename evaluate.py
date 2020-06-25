from utils import config, data, factory, devices
import tensorflow as tf
import base
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description="Run model evaluation.")
    parser.add_argument(
        "-c", "--config",
        dest="config_path",
        type=str,
        help="Path to the configuration file",
        required=True
    )
    parser.add_argument(
        "-l", "--load-checkpoint",
        dest="load_checkpoint",
        default=None,
        type=str,
        help="Name of the checkpoint to load. Overwrites model.load_checkpoint in configuration file"
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        default=None,
        type=str,
        help="Location of the prepared data. Overwrites data.data_dir in configuration file"
    )
    parser.add_argument(
        "--cpu",
        dest="cpu",
        action="store_true",
        help="Use CPU regardless of GPU availability"
    )
    args = parser.parse_args()
    return args


def process_config(args, cfg):
    if args.load_checkpoint is not None:
        cfg.model.load_checkpoint = args.load_checkpoint

    if args.data_dir is not None:
        cfg.data.data_dir = args.data_dir
    elif "DATA_DIR" in os.environ:
        cfg.data.data_dir = os.environ["DATA_DIR"]


def evaluate(cfg):
    classes = {}
    for k in ["data_preprocessing", "model"]:
        module_name = cfg[k].module_name
        class_name = cfg[k].class_name
        classes[k] = factory.get_class(module_name, class_name)

    preprocessing: base.data_preprocessing.BaseDataPreprocessing = classes["data_preprocessing"](cfg)
    preprocessing.preprocess_config()

    print("Loading data...")
    data_loader = data.get_test_data(cfg, preprocessing)["test"]

    print("Building model...")
    model: base.model.BaseModel = classes["model"](cfg)
    model.build()
    model.summary()
    if not cfg.model.load_checkpoint:
        print("No model checkpoint specified")
        exit(0)
    model.load()
    model.compile()

    print("Evaluating...")
    result = model.evaluate(
        data_loader.batch(cfg.data.get("batch_size", 1)),
        workers=cfg.trainer.get("workers", 1),
    )
    print("\n" + "-" * 80)
    for metric, x in zip(model.metrics_names, result):
        print(f"{metric}: {x:.4f}")
    print("-" * 80)


def main(args):
    cfg = config.get_config(args.config_path)
    process_config(args, cfg)

    if args.cpu:
        print("Evaluating using CPU")
        with tf.device("CPU"):
            evaluate(cfg)
    else:
        devices.device_setup(cfg)
        evaluate(cfg)


if __name__ == '__main__':
    try:
        args = get_args()
        main(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
