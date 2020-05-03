import tensorflow_datasets as tfds
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Download COCO data")
    parser.add_argument("-d", "--dir", type=str, default="data", dest="dir",
                        help="Directory to save the data")
    return parser.parse_args()


def main():
    args = parse_args()
    coco_builder = tfds.builder("coco/2017_panoptic", data_dir=args.dir)
    coco_builder.download_and_prepare()


if __name__ == '__main__':
    main()
