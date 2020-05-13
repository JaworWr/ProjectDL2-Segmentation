import tensorflow_datasets as tfds
import argparse
import getpass
import requests
import os
import progressbar


def parse_args():
    parser = argparse.ArgumentParser(description="Download cityscapes data")
    parser.add_argument("--data-dir", type=str, default="data", dest="data_dir",
                        help="Directory to save the data")
    parser.add_argument("--download-dir", type=str, default=None, dest="download_dir",
                        help="Directory to download the data or location of previously downloaded data")
    parser.add_argument("--extract-dir", type=str, default=None, dest="extract_dir",
                        help="Directory to save the extracted data")
    parser.add_argument("--download", action="store_true", help="Download the zip data")
    return parser.parse_args()


def download_zip_data(download_dir):
    with requests.Session() as session:
        while True:
            username = input("Cityscapes username or email address: ")
            password = getpass.getpass("Cityscapes password: ")

            credentials = {
                "username": username,
                "password": password
            }
            r = session.get("https://www.cityscapes-dataset.com/login",
                            allow_redirects=False)
            r.raise_for_status()
            credentials["submit"] = "Login"
            r = session.post("https://www.cityscapes-dataset.com/login",
                             data=credentials, allow_redirects=False)
            r.raise_for_status()

            if r.status_code == requests.codes.found:
                break
            else:
                print("\nIncorrect credentials\n")

        packages = {
            "gtFine_trainvaltest.zip": "https://www.cityscapes-dataset.com/file-handling/?packageID=1",
            "leftImg8bit_trainvaltest.zip": "https://www.cityscapes-dataset.com/file-handling/?packageID=3",
        }

        for filename, url in packages.items():
            path = os.path.join(download_dir, filename)
            print(f"Downloading to {os.path.abspath(path)}")
            with session.get(url, allow_redirects=False, stream=True) as r:
                r.raise_for_status()
                assert r.status_code == requests.codes.ok
                content_len = r.headers.get("content-length")
                content_len = int(content_len)

                d = 0
                widgets = [
                    progressbar.Percentage(),
                    " ", progressbar.DataSize(), " of ", progressbar.DataSize(variable="max_value"),
                    " ", progressbar.Bar(),
                    " ", progressbar.FileTransferSpeed(),
                    " ", progressbar.ETA(),
                ]
                with open(path, "wb") as f, progressbar.ProgressBar(max_value=content_len, widgets=widgets) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        d += len(chunk)
                        bar.update(d)
                        f.write(chunk)


def main():
    args = parse_args()
    if args.download_dir is not None:
        download_dir = args.download_dir
    else:
        download_dir = os.path.join(args.data_dir, "download")
    if args.download:
        download_zip_data(download_dir)
    birds_builder = tfds.builder("cityscapes", data_dir=args.data_dir)
    download_config = tfds.download.DownloadConfig(
        manual_dir=args.download_dir
    )
    birds_builder.download_and_prepare(download_dir=args.extract_dir, download_config=download_config)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except requests.HTTPError as e:
        print("Http error: " + e.strerror)
