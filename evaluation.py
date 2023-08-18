from __future__ import print_function
from __future__ import division

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from matplotlib import gridspec


from PIL import Image
import scipy.io as sc

from config import Config
import util
from model import WNet
from evaluation_dataset import EvaluationDataset

from data.load_ocid import load_ocid, split_ocid_images
from custom_dataset import DepthDataset, DepthEvaluationDataset
import tqdm

from Evaluator.python.metrics import Results
import pickle

def main():
    results = Results()
    results.initialization()
    vis = True

    print("PyTorch Version: ", torch.__version__)
    if torch.cuda.is_available():
        print("Cuda is available. Using GPU")



    config = Config()

    if config.use_custom_data:

        ###########################################################################
        train_xform = transforms.Compose([
            # transforms.Resize((config.input_size, config.input_size)),
            transforms.ToTensor()
        ])
        val_xform = transforms.Compose([
            transforms.Resize((config.input_size, config.input_size)),
            transforms.ToTensor()
        ])
        transform_no_tensor = transforms.Compose([
            transforms.Resize((config.input_size, config.input_size)),
        ])
        number_all = 2390
        # split_ocid_images(ocid_folder_path="data/OCID-dataset", region_size=config.input_size, limit_number_samples=20,
        #                   limit_number_each=20)  # 2390="all"

        # ocid = load_ocid(ocid_folder_path="data/OCID-dataset/splits", limit_number_samples="all")
        ocid = load_ocid(ocid_folder_path="data/OCID-dataset", limit_number_samples="all")
        print(ocid)

        print("OCID len", len(ocid))
        # together = concatenate_datasets([nyuv2, ocid]).shuffle(seed=123)
        together = ocid.shuffle(seed=123)  #
        together = together.train_test_split(test_size=0.1)
        train_hf = together["train"]
        val_hf = together["test"]
        # train_dataset = DepthDataset(train_hf, "train", train_xform, transform_no_tensor, config.rgb_only,
        #                              config.depth_only, config.both, config.normalize_normals, config.input_size)

        eval_folder = "data/eval"

        val_dataset = DepthEvaluationDataset(val_hf, eval_folder, "test", val_xform, transform_no_tensor, config)
        # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4,
        #                                                shuffle=True)
        evaluation_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.test_batch_size, num_workers=4, shuffle=False)

        ###########################################################################
    else:
        data_type = "ocid"
        evaluation_dataset = EvaluationDataset("test", data_type)

        evaluation_dataloader = torch.utils.data.DataLoader(evaluation_dataset,
                                                            batch_size=config.test_batch_size, num_workers=4, shuffle=False)

    ###################################
    #          Model Setup            #
    ###################################

    # model_name = "guru"
    model_name = config.model_name

    vis_folder = f"eval_imgs/{model_name}"
    os.makedirs(vis_folder, exist_ok=True)

    # We will only use .forward_encoder()
    if torch.cuda.is_available():
        autoencoder = torch.load(f"./models/{model_name}")
    else:
        autoencoder = torch.load(f"./models/{model_name}", map_location=torch.device('cpu'))
    util.enumerate_params([autoencoder])

    ###################################
    #          Testing Loop           #
    ###################################

    autoencoder.eval()

    def combine_patches(image, patches):
        w, h = image[0].shape
        segmentation = torch.zeros(w, h)
        x, y = (0, 0)  # Start of next patch
        for patch in patches:
            if y + size > h:
                y = 0
                x += size
            segmentation[x:x + size, y:y + size] = patch
            y += size
        return segmentation

    # Because this model is unsupervised, our predicted segment labels do not
    # correspond to the actual segment labels.
    # We need to figure out what the best mapping is.
    # To do this, we will just count, for each of our predicted labels,
    # The number of pixels in each class of actual labels, and take the max in that image
    def count_predicted_pixels(predicted, actual):
        pixel_count = torch.zeros(config.k, config.k)
        for k in range(config.k):
            mask = (predicted == k)
            masked_actual = actual[mask]
            for i in range(config.k):
                pixel_count[k][i] += torch.sum(masked_actual == i)
        return pixel_count

    # Converts the predicted segmentation, based on the pixel counts
    def convert_prediction(pixel_count, predicted):
        map = torch.argmax(pixel_count, dim=1)

        for x in range(predicted.shape[0]):
            for y in range(predicted.shape[1]):
                predicted[x, y] = map[predicted[x, y]]
        return predicted

    def compute_iou(predicted, actual):
        intersection = 0
        union = 0
        for k in range(config.k):
            a = (predicted == k).int()
            b = (actual == k).int()
            # if torch.sum(a) < 100:
            #    continue # Don't count if the channel doesn't cover enough
            intersection += torch.sum(torch.mul(a, b))
            union += torch.sum(((a + b) > 0).int())
        return intersection.float() / union.float()

    def pixel_accuracy(predicted, actual):
        return torch.mean((predicted == actual).float())

    iou_sum = 0
    pixel_accuracy_sum = 0
    n = 0
    # Currently, we produce the most generous prediction looking at a single image
    for i, [images, segmentations, image_path] in tqdm.tqdm(enumerate(evaluation_dataloader, 0),
                                                            total=len(evaluation_dataloader), desc="Testing"):
        # print(image_path)
        size = config.input_size
        # Assuming batch size of 1 right now
        image = images[0]
        target_segmentation = segmentations[0]



        # # NOTE: We cut the images down to a multiple of the patch size
        # cut_w = (image[0].shape[0] // size) * size
        # cut_h = (image[0].shape[1] // size) * size
        #
        # image = image[:, 0:cut_w, 0:cut_h]
        # # print(image.shape)
        # target_segmentation = target_segmentation[:, 0:cut_w, 0:cut_h]
        #
        # patches = image.unfold(0, 3, 3).unfold(1, size, size).unfold(2, size, size)
        # # c = 3
        # c = config.inputChannels
        # patch_batch = patches.reshape(-1, c, size, size)


        patch_batch = torch.unsqueeze(image, 0)

        if torch.cuda.is_available():
            patch_batch = patch_batch.cuda()

        seg_batch = autoencoder.forward_encoder(patch_batch)
        seg_batch = torch.argmax(seg_batch, axis=1).float()

        predicted_segmentation = combine_patches(image, seg_batch)
        prediction = predicted_segmentation.int()

        actual = target_segmentation[0].int()

        pixel_count = count_predicted_pixels(prediction, actual)
        prediction = convert_prediction(pixel_count, prediction)









        gt = target_segmentation.numpy()[0]
        pred = prediction.numpy()

        gt = gt[0:pred.shape[0], 0:pred.shape[1]]
        # print(gt.shape, pred.shape)

        # visualization
        if vis:
            plt.figure(figsize=(20, 4))
            grid_spec = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 3])

            plt.subplot(grid_spec[0])
            plt.axis('off')
            plt.title('RGB image')
            plt.imshow(image.numpy()[:3].transpose((1, 2, 0)))

            plt.subplot(grid_spec[1])
            plt.axis('off')
            plt.title('prediction')
            plt.imshow(pred)

            plt.subplot(grid_spec[2])
            plt.axis('off')
            plt.title('ground Truth')
            plt.imshow(gt)

            # plt.show()
            plt.savefig(f"{vis_folder}/{i}.png", dpi=200)
        try:
            ri, vi, R = results.update(pred, gt)
        except Exception as e:
            print("Something went wrong", type(e), e)

        # print('SC is: ', R)
        # print('RI is: ', ri)
        # print('VI is: ', vi)

    meanSC, meanPRI, meanVI = results.get_results()

    print("==================================")
    print('mean SC is: ', meanSC)
    print('mean PRI is: ', meanPRI)
    print('mean VI is: ', meanVI)



if __name__ == "__main__":
    main()
