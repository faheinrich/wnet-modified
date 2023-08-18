from pathlib import Path
from datasets import Image, Dataset
import os
from data.split_image import split_image
import cv2
import tqdm
import random
import shutil
import sys
import numpy as np
import wget
from wget import bar_adaptive
import tarfile





def split_ocid_images(ocid_folder_path, destination_path, region_size, limit_number_samples, limit_number_each):

    try:
        shutil.rmtree(destination_path)
    except:
        print('Error deleting directory')

    ocid = load_ocid(ocid_folder_path, limit_number_samples)

    image_path = destination_path + "/train/rgb"
    label_path = destination_path + "/train/label"
    depth_path = destination_path + "/train/depth"

    os.makedirs(image_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)

    do_print = True
    for data_idx, d in tqdm.tqdm(enumerate(ocid), total=len(ocid), desc="Splitting and saving images."):
        image = d["image"]
        subregions, rows, cols = split_image(image, region_size)

        if do_print:
            print("Number per split:", rows * cols)
            do_print = False

        limit_number = np.min([limit_number_each, len(subregions)])

        random_indices = np.random.choice(np.arange(limit_number), limit_number)
        # random_indices = range(limit_number)

        for split_idx, list_idx in enumerate(random_indices):
            img = subregions[list_idx]
            cv2.imwrite(image_path + f"/{data_idx}_{split_idx}.png", img[:,:,::-1])

        label = d["label"]
        subregions, rows, cols = split_image(label, region_size)
        for split_idx, list_idx in enumerate(random_indices):
            img = subregions[list_idx]
            cv2.imwrite(label_path + f"/{data_idx}_{split_idx}.png", img)

        depth = d["depth"].astype(np.uint16)
        subregions, rows, cols = split_image(depth, region_size)
        for split_idx, list_idx in enumerate(random_indices):
            img = subregions[list_idx]
            cv2.imwrite(depth_path + f"/{data_idx}_{split_idx}.png", img)


def download_ocid(ocid_folder_path):
    url = "https://data.acin.tuwien.ac.at/index.php/s/g3EkcgcPioolQmJ/download"

    def bar_progress(current, total, width=80):
        progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
        # Don't use print() as it will print in new line every time.
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()

    filename = wget.download(url, out=ocid_folder_path, bar=bar_progress)
    print(f"Downloaded data to {filename}.")


def load_ocid(ocid_folder_path: str, limit_number_samples):
    root = Path(ocid_folder_path)

    image_paths_orig = list(root.rglob("rgb/*.png"))

    if limit_number_samples != "all":
        limit_number = limit_number_samples
        random.shuffle(image_paths_orig)
        image_paths_orig = image_paths_orig[:limit_number]

    depth_paths = list(map(lambda x: x.parent.parent / "depth" / x.name, image_paths_orig))
    label_paths = list(map(lambda x: x.parent.parent / "label" / x.name, image_paths_orig))
    dataset = Dataset.from_dict({"image": list(map(lambda x: str(x), image_paths_orig)),
                                 "label": list(map(lambda x: str(x), label_paths)),
                                 "depth": list(map(lambda x: str(x), depth_paths))})
    dataset = dataset.cast_column("image", Image()).cast_column("depth", Image()).cast_column("label", Image())
    return dataset.with_format("numpy")


if __name__ == "__main__":
    download_filename = "data/OCID-dataset.tar.gz"
    # download_ocid(download_filename)
    with tarfile.open(download_filename, "r:gz") as tar:
        tar.extractall("data")
    print("Extracted.")