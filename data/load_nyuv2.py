import os
import sys
import shutil
from pathlib import Path
from typing import Union
import tqdm
import wget
import h5py

import numpy as np
import cv2
from PIL import Image

from datasets import Dataset, Image
from data.split_image import split_image, stitch_image

# DOWNLOAD_FILENAME = "nyu_depth_v2_labeled.mat"


def nyuv2_matfile_to_folder(mat_file: str, target_folder, split_images=False, region_size=64):
    print("Loading dataset,", end="")
    p = Path(mat_file)

    f = h5py.File(p, mode='r')

    rgb = f["images"]  # uint8
    depths = f["depths"]  # float32
    # rawDepth = f['rawDepths']
    labels = f["labels"]  # uint16

    target_path = Path(target_folder)
    rgb_path = target_path / Path("imgs")
    depth_path = target_path / Path("depths")
    labels_path = target_path / Path("labels")
    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)


    for idx in tqdm.tqdm(range(len(rgb)), desc="Extracting and writing images."):
        filename = Path(f"{idx}.png")

        img = rgb[idx].transpose((2, 1, 0))
        if split_images:
            pass
        else:
            pass
        cv2.imwrite(str(rgb_path / filename), img[:,:,::-1])

        depth = depths[idx].T
        # print(depth.dtype, np.min(depth), np.max(depth))
        # depth = depth.astype(np.uint16)
        # print(depth.dtype, np.min(depth), np.max(depth))
        depth = (depth * 1000).astype(np.uint16)
        cv2.imwrite(str(depth_path / Path(f"{idx}.png")), depth)

        label = labels[idx].T.astype(np.uint16)
        cv2.imwrite(str(labels_path / filename), label)


def load_nyuv2_data(mat_file: str, extract_data_to_folder: False):
    print("Loading dataset,", end="")
    data_folder = Path(mat_file).parent / "nyuv2"
    if extract_data_to_folder:
        nyuv2_matfile_to_folder(mat_file, data_folder)

    root = Path(data_folder)
    image_paths_orig = list(root.rglob("imgs/*.png"))
    depth_paths = list(map(lambda x: x.parent.parent / "depths" / (x.stem+".png"), image_paths_orig))
    label_paths = list(map(lambda x: x.parent.parent / "labels" / x.name, image_paths_orig))
    dataset = Dataset.from_dict({"image": list(map(lambda x: str(x), image_paths_orig)),
                                 "label": list(map(lambda x: str(x), label_paths)),
                                 "depth": list(map(lambda x: str(x), depth_paths))})
    dataset = dataset.cast_column("image", Image()).cast_column("depth", Image()).cast_column("label", Image())
    return dataset.with_format("numpy")


def process_dataset(data, num_imgs: Union[str, int], square_region_size):

    rgb = data["rgb"]
    labels = data["labels"]
    depths = data["depths"]

    print(rgb[0].shape)
    print(labels[0].shape)
    print(depths[0].shape)

    if isinstance(num_imgs, str):
        num_imgs = len(rgb)
    elif isinstance(num_imgs, int):
        if num_imgs > len(rgb) or num_imgs <= 0:
            num_imgs = len(rgb)

    dummy_img = rgb[0]
    img_width = dummy_img.shape[1]
    img_height = dummy_img.shape[2]
    print("W", img_width, "x H", img_height)

    os.makedirs("data_custom/imgs_subregions", exist_ok=True)
    os.makedirs("data_custom/masks_subregions", exist_ok=True)
    os.makedirs("data_custom/depths_subregions", exist_ok=True)
    os.makedirs("data_custom/imgs_whole", exist_ok=True)
    os.makedirs("data_custom/masks_whole", exist_ok=True)
    os.makedirs("data_custom/depths_whole", exist_ok=True)

    for img_idx in tqdm.tqdm(range(num_imgs), desc="Generating data."):

        color = rgb[img_idx].transpose((2, 1, 0)) # uint8
        label = labels[img_idx].transpose((1, 0)) # uint16
        depth = depths[img_idx].transpose((1, 0)) # float32

        Image.fromarray(color).save(f"data_custom/imgs_whole/{img_idx}.jpg")
        Image.fromarray(label).save(f"data_custom/masks_whole/{img_idx}.gif")

        # np.save(f"data_custom/depths_whole/{img_idx}.npy", depth)
        # Image.fromarray(depth).save(f"data_custom/depths_whole/{img_idx}.exr")

        color_subregions, rows, cols = split_image(color, square_region_size)
        for idx, sub in enumerate(color_subregions):
            Image.fromarray(sub).save(f"data_custom/imgs_subregions/{img_idx}_{idx}.jpg")

        label_subregions, rows, cols = split_image(label, square_region_size)
        for idx, sub in enumerate(label_subregions):
            Image.fromarray(sub).save(f"data_custom/masks_subregions/{img_idx}_{idx}.gif")

        # depths_subregions, rows, cols = split_image(depths, square_region_size)
        # for idx, sub in enumerate(depths_subregions):
        #     np.save(f"data_custom/depths_subregions/{img_idx}_{idx}.npy", sub)
        #     Image.fromarray(sub).save(f"data_custom/depths_subregions/{img_idx}_{idx}.exr")


def download_NYU_Depth_Dataset_V2(mat_file: str):
    url = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"

    def bar_progress(current, total, width=80):
        progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
        # Don't use print() as it will print in new line every time.
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()

    filename = wget.download(url, out=mat_file, bar=bar_progress)
    print(f"Downloaded data to {filename}.")


if __name__ == "__main__":
    # download_NYU_Depth_Dataset_V2()

    try:
        shutil.rmtree("data_custom")
    except:
        pass

    data = load_nyuv2_data()

    rgb = data["rgb"][:]
    label = data["label"][:]
    depth = data["depth"][:]
    print(type(rgb))
    print(rgb.dtype)
    print(type(label))
    print(label.dtype)
    print(type(depth))
    print(depth.dtype)

    # process_dataset(data, num_imgs=10, square_region_size=200)

    # test_img = np.array(Image.open("data_custom/imgs_whole/0.jpg"))
    # test_img = np.array(Image.open("data_custom/masks_whole/0.gif"))
    # subregions, rows, cols = split_image(test_img, 128)
    # img = stitch_image(subregions, rows, cols, test_img.shape[0], test_img.shape[1])

    # print(img.shape)
    # test = Image.fromarray(img)
    # test.show()
