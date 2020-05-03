import tensorflow_datasets as tfds
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Download calltech_birds2011 data")
    parser.add_argument("--data-dir", type=str, default="data", dest="data_dir",
                        help="Directory to save the data")
    parser.add_argument("--download-dir", type=str, default=None, dest="download_dir",
                        help="Directory to download the data")
    return parser.parse_args()


def main():
    args = parse_args()
    birds_builder = tfds.builder("caltech_birds2011", data_dir=args.data_dir)
    birds_builder.download_and_prepare(download_dir=args.download_dir)


if __name__ == '__main__':
    main()
