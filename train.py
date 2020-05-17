from utils import config, data, devices, factory
import base
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description="Run model training.")
    parser.add_argument(
        "-c", "--config",
        dest="config_path",
        type=str,
        help="Path to the configuration file",
        required=True
    )
    parser.add_argument(
        "-s", "--save-checkpoint",
        dest="save_checkpoint",
        default=None,
        type=str,
        help="Where to save the model checkpoint. Overwrites model.save_checkpoint in configuration file"
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
        help="Location of the prepared data"
    )
    args = parser.parse_args()
    return args


def process_config(args, cfg):
    if args.save_checkpoint is not None:
        cfg.model.save_checkpoint = args.save_checkpoint
    if args.load_checkpoint is not None:
        cfg.model.load_checkpoint = args.load_checkpoint

    if args.data_dir is not None:
        cfg.data.data_dir = args.data_dir
    elif "DATA_DIR" in os.environ:
        cfg.data.data_dir = os.environ["DATA_DIR"]


def main(args):
    cfg = config.get_config(args.config_path)
    process_config(args, cfg)

    devices.device_setup(cfg)

    classes = {}
    for k in ["data_preprocessing", "model", "trainer"]:
        module_name = cfg[k].module_name
        class_name = cfg[k].class_name
        classes[k] = factory.get_class(module_name, class_name)

    preprocessing: base.data_preprocessing.BaseDataPreprocessing = classes["data_preprocessing"](cfg)
    preprocessing.preprocess_config()

    print("Loading data...")
    data_loaders = data.get_train_valid_data(cfg, preprocessing)

    print("Building model...")
    model: base.model.BaseModel = classes["model"](cfg)
    model.build()
    model.compile()
    model.summary()
    model.load()

    trainer: base.trainer.BaseTrainer = classes["trainer"](cfg, model, data_loaders)
    print("Training...")
    try:
        trainer.train()
    finally:
        model.save()


if __name__ == '__main__':
    try:
        args = get_args()
        main(args)
    except KeyboardInterrupt:
        print("Interrupted by user.")
