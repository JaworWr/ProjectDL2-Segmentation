import tensorflow_datasets as tfds
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Download cityscapes data")
    parser.add_argument("--data-dir", type=str, default="data", dest="data_dir",
                        help="Directory to save the data")
    parser.add_argument("--download-dir", type=str, default=None, dest="download_dir",
                        help="Directory to download the data or location of previously downloaded data")
    parser.add_argument("--extract-dir", type=str, default=None, dest="extract_dir",
                        help="Directory to save the extracted data")
    return parser.parse_args()


def main():
    args = parse_args()
    birds_builder = tfds.builder("cityscapes", data_dir=args.data_dir)
    download_config = tfds.download.DownloadConfig(
        manual_dir=args.download_dir
    )
    birds_builder.download_and_prepare(download_dir=args.extract_dir, download_config=download_config)


if __name__ == '__main__':
    main()
