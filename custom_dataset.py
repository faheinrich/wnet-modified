

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import glob
import time
from config import Config
import cv2

from data.split_image import stitch_image, split_image

config = Config()

import cupy as cp
normal_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_normals(const float* depthImg, float* normalImg, int depthWidth, int depthHeight, int radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx < (depthWidth - radius)) && (idx >= radius) &&
        (idy < (depthHeight - radius)) && (idy >= radius)) {

        int a_idx = (idy - radius) * depthWidth + (idx + radius);
        int b_idx = (idy + radius) * depthWidth + (idx + radius);
        int c_idx = idy * depthWidth + (idx - radius);

        float depth_a = depthImg[a_idx];
        float depth_b = depthImg[b_idx];
        float depth_c = depthImg[c_idx];

        float a1 = -(float)radius;
        float a3 = depth_a - depth_b;

        float b1 = (float)-radius;
        float b2 = (float)(-radius - radius);
        float b3 = depth_c - depth_a;

        float v1 = -(a3 * b2);
        float v2 = (a3 * b1) - (a1 * b3);
        float v3 = (a1 * b2);

        float norm = sqrt(v1 * v1 + v2 * v2 + v3 * v3);

        int normal_flat_pos = idy * depthWidth * 3 + (idx * 3);
        normalImg[normal_flat_pos + 0] = v1 / norm;
        normalImg[normal_flat_pos + 1] = v2 / norm;
        normalImg[normal_flat_pos + 2] = v3 / norm;
    }
}
''', 'compute_normals')


dummy_img = np.zeros((480, 640, 3))
width, height = dummy_img.shape[1], dummy_img.shape[0]

radius = 1
block_dim = (16, 16)
grid_dim = ((width + block_dim[0] - 1) // block_dim[0],
            (height + block_dim[1] - 1) // block_dim[1])
normal_buffer = cp.zeros((height, width, 3), dtype=cp.float32)
print("Works here?", normal_buffer.shape)

# import rgbd_seg
# radius = 2
# median_filter = 5
# averaging = 1 # unsused
# gaussian_kernel = 5
# gaussian_sigma = 4
# calc = rgbd_seg.NormalCalculator(dummy_img.shape[1], dummy_img.shape[0], radius, median_filter, averaging, gaussian_kernel, gaussian_sigma)
# #  int avgFilterN, int gaussianKernelSizeA,float gaussianSigmaX


file_ext = ".jpg"

randomCrop = transforms.RandomCrop(config.input_size)
centerCrop = transforms.CenterCrop(config.input_size)
toTensor   = transforms.ToTensor()
toPIL      = transforms.ToPILImage()

# Assumes given data directory (train, val, etc) has a directory called "images"
# Loads image as both inputs and outputs
# Applies different transforms to both input and output


def calc_normal(depth):

    # print("Depth", depth.dtype)
    # print("Depth", depth.shape)
    # print("Depth", np.min(depth), np.max(depth))
    kernel_input = cp.array(depth).astype(cp.float32)
    # cv2.imshow("Depth", depth.astype(np.float)/np.max(depth))

    normal_kernel(grid_dim, block_dim, (kernel_input, normal_buffer, width, height, radius))

    normal_img = normal_buffer.get()
    # print("Normal", normal_img.dtype)
    # print("Normal", normal_img.shape)
    # print("Normal", np.min(normal_img), np.max(normal_img))
    # cv2.imshow("Normal", (normal_img+1)/2)
    # cv2.waitKey()


    return normal_img


class DepthDataset(Dataset):
    def __init__(self, hf_dataset, mode, input_transforms, transform_no_tensor, config):
        self.data = hf_dataset

        print("CALC NORMALS")
        self.normals = []
        import tqdm
        for d in tqdm.tqdm(self.data, total=len(self.data)):
            self.normals.append(calc_normal(d["depth"]))
        print("DONE")

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
        global kernel_input

        input = self.data[i]["image"]

        depth_data = self.data[i]["depth"].astype(float)
        # print(np.min(depth_data), np.max(depth_data))
        # print("depth data", depth_data.shape)
        # kernel_input = cp.array(np.array(depth_data)).astype(cp.float32)
        # print("kernel input", kernel_input.shape)
        # normal_kernel(grid_dim, block_dim, (kernel_input, normal_buffer, width, height, radius))
        # normal_img = normal_buffer.get()

        # normal_img = calc_normal(depth_data)
        normal_img = self.normals[i]

        # normal_img = calc.processImagePython(depth_data)
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
            stack_input.append(depth_tensor.to(torch.float32))
            stack_output.append(depth_tensor.to(torch.float32))

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
