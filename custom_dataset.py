
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import glob
import time
from config import Config

from data.split_image import stitch_image, split_image

import rgbd_seg
import cv2
radius = 2
dummy_img = np.zeros((480, 640, 3))
median_filter = 5
averaging = 1 # unsused
gaussian_kernel = 5
gaussian_sigma = 4
calc = rgbd_seg.NormalCalculator(dummy_img.shape[1], dummy_img.shape[0], radius, median_filter, averaging, gaussian_kernel, gaussian_sigma)
#  int avgFilterN, int gaussianKernelSizeA,float gaussianSigmaX


config = Config()

file_ext = ".jpg"

randomCrop = transforms.RandomCrop(config.input_size)
centerCrop = transforms.CenterCrop(config.input_size)
toTensor   = transforms.ToTensor()
toPIL      = transforms.ToPILImage()

# Assumes given data directory (train, val, etc) has a directory called "images"
# Loads image as both inputs and outputs
# Applies different transforms to both input and output
class DepthDataset(Dataset):
    def __init__(self, hf_dataset, mode, input_transforms, transform_no_tensor, config):
        self.data = hf_dataset
        # self.split_data = self.data.map(lambda x: {"image": split_image(x["image"], img_size)}) # 'image', 'label', 'depth'
        # print(self.split_data)
        # exit()
        self.mode = mode
        self.transforms = input_transforms
        self.transform_no_tensor = transform_no_tensor

        self.config = config
        # self.pad = torch.nn.ZeroPad2d(radius)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # Get the ith item of the dataset

        input = self.data[i]["image"]

        depth_data = self.data[i]["depth"]
        # print(np.min(depth_data), np.max(depth_data))
        normal_img = calc.processImagePython(depth_data)
        img = Image.fromarray(input)

        stack_input = []
        stack_output = []
        input = self.transforms(img)

        input = toPIL(input)
        output = input.copy()
        if self.mode == "train" and config.variationalTranslation > 0:
            output = randomCrop(input)
        # input = toTensor(centerCrop(input))
        input = toTensor(input)
        output = toTensor(output)

        if self.config.use_rgb:
            stack_input.append(input)
            stack_output.append(output)

        if self.config.use_depth:
            depth = depth_data[np.newaxis, ...].copy()
            depth_tensor = self.transform_no_tensor(torch.from_numpy(depth))
            if self.config.normalize_depth:
                depth_tensor = depth_tensor / torch.max(depth_tensor)
            stack_input.append(depth_tensor)
            stack_output.append(depth_tensor)

        if self.config.use_normals:
            normals = self.transform_no_tensor(torch.from_numpy(normal_img.transpose(2, 0, 1)))
            if self.config.normalize_normals:
                normals = (normals + 1) / 2
            stack_input.append(normals)
            stack_output.append(normals)

        input = torch.concat(stack_input, axis=0)
        output = torch.concat(stack_output, axis=0)
        # print(torch.min(input), torch.max(output))
        return input, output



class DepthEvaluationDataset(Dataset):
    def __init__(self, hf_dataset, mode, eval_folderpath, input_transforms, transform_no_tensor, config):
        self.data = hf_dataset
        # self.split_data = self.data.map(lambda x: {"image": split_image(x["image"], img_size)}) # 'image', 'label', 'depth'
        # print(self.split_data)
        # exit()
        self.filepaths = [eval_folderpath + f"/{i}.jpg" for i in range(len(self.data))]

        self.mode = mode
        self.transforms = input_transforms
        self.transform_no_tensor = transform_no_tensor
        self.pad = torch.nn.ZeroPad2d(radius)
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        input = self.data[i]["image"]
        label = self.data[i]["label"]
        depth_data = self.data[i]["depth"]

        # print(np.min(depth_data), np.max(depth_data))
        normal_img = calc.processImagePython(depth_data)
        img = Image.fromarray(input)

        stack_input = []
        input = self.transforms(img)

        input = toPIL(input)
        input = toTensor(input)

        if self.config.use_rgb:
            stack_input.append(input)

        if self.config.use_depth:
            depth = depth_data[np.newaxis, ...].copy()
            depth_tensor = self.transform_no_tensor(torch.from_numpy(depth))
            if self.config.normalize_depth:
                depth_tensor = depth_tensor / torch.max(depth_tensor)
            stack_input.append(depth_tensor)

        if self.config.use_normals:
            normals = self.transform_no_tensor(torch.from_numpy(normal_img.transpose(2, 0, 1)))
            if self.config.normalize_normals:
                normals = (normals + 1) / 2
            stack_input.append(normals)

        input = torch.concat(stack_input, axis=0)
        # print("inputs", input.shape)
        segmentation = toTensor(label)
        segmentation = self.transform_no_tensor(segmentation)
        # print("segmentations", segmentation.shape)
        return input, segmentation, self.filepaths[i]



# # Assumes data directory/mode has a directory called "images" and one called "segmentations"
# # Loads image as input, segmentation as output
# # Transforms are specified in this file
# class EvaluationDataset(Dataset):
#     def __init__(self, mode, type):
#         self.mode = mode # The "test" directory name
#         self.data_path  = os.path.join(config.data_dir, mode)
#         self.images_dir = os.path.join(self.data_path, 'images_'+type)
#         self.seg_dir    = os.path.join(self.data_path, 'segmentations_npy_'+type)
#         self.image_list = self.get_image_list()
#
#     def __len__(self):
#         return len(self.image_list)
#
#     def __getitem__(self, i):
#         # Get the ith item of the dataset
#         image_filepath, segmentation_filepath = self.image_list[i]
#         image        = self.load_pil_image(image_filepath)
#         segmentation = self.load_segmentation(segmentation_filepath)
#         #print(image_filepath)
#
#         return toTensor(image), toTensor(segmentation), image_filepath
#
#     def get_image_list(self):
#         image_list = []
#         for file in os.listdir(self.images_dir):
#             if file.endswith(file_ext):
#                 image_path = os.path.join(self.images_dir, file)
#                 seg_path   = os.path.join(self.seg_dir,    file.split('.')[0]+'.npy')
#                 image_list.append((image_path, seg_path))
#         return image_list
#
#     def load_pil_image(self, path):
#     # open path as file to avoid ResourceWarning
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('RGB')
#
#     def load_segmentation(self, path):
#         return np.load(path, allow_pickle=True)
