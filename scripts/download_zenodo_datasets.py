"""
Helper script to download official multi-tab and defended WF datasets from Zenodo:
Record: https://zenodo.org/records/13732130
"""

import os
import zipfile
import argparse
import urllib.request

ZENODO_RECORD = "13732130"
DATASETS = {
    "Closed_2tab": "closed_2tab.npz.zip",
    "Closed_3tab": "closed_3tab.npz.zip",
    "Closed_4tab": "closed_4tab.npz.zip",
    "Closed_5tab": "closed_5tab.npz.zip",
    "Open_2tab": "open_2tab.npz.zip",
    "Open_3tab": "open_3tab.npz.zip",
    "Open_4tab": "open_4tab.npz.zip",
    "Open_5tab": "open_5tab.npz.zip",
    "CW": "CW.npz.zip",
    "OW": "OW.npz.zip",
    "WTF-PAD": "WTF-PAD.npz.zip",
    "Front": "Front.npz.zip",
    "Walkie-Talkie": "Walkie-Talkie.npz.zip",
    "TrafficSliver": "TrafficSliver.npz.zip",
    "NCDrift_sup": "NCDrift_sup.npz.zip",
    "NCDrift_inf": "NCDrift_inf.npz.zip",
}


def download_dataset(name, out_dir="./datasets"):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name}. Available: {list(DATASETS.keys())}")
    os.makedirs(out_dir, exist_ok=True)
    zip_filename = DATASETS[name]
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{zip_filename}/content"
    zip_path = os.path.join(out_dir, zip_filename)

    print(f"Downloading {name} from {url} -> {zip_path} ...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print(f"Done extracting {name} to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download official WFlib / ARES datasets from Zenodo")
    parser.add_argument("--dataset", type=str, default="Closed_2tab", choices=list(DATASETS.keys()))
    parser.add_argument("--out_dir", type=str, default="./datasets")
    args = parser.parse_args()
    download_dataset(args.dataset, args.out_dir)
