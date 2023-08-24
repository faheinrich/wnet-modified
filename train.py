# Implementation of W-Net: A Deep Model for Fully Unsupervised Image Segmentation
# in Pytorch.
# Author: Griffin Bishop
# Adapted and changed by: Guru Deep Singh

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os, shutil
import copy
import pickle

from config import Config
import util
from model import WNet
from autoencoder_dataset import AutoencoderDataset
from soft_n_cut_loss import NCutLoss2D
from plot_loss import plot_loss

import tqdm
from data.load_ocid import load_ocid, split_ocid_images
from data.load_nyuv2 import load_nyuv2_data
from custom_dataset import DepthDataset
from datasets import concatenate_datasets




def main():
    print("PyTorch Version: ",torch.__version__)
    if torch.cuda.is_available():
        print("Cuda is available. Using GPU")

    config = Config()
    os.makedirs("latent_images", exist_ok=True)
    os.makedirs("models", exist_ok=True)


    ###################################
    # Image loading and preprocessing #
    ###################################

    train_xform = transforms.Compose([
        transforms.Resize((config.training_region_size,config.training_region_size)),
        transforms.ToTensor()
    ])
    val_xform = transforms.Compose([
        transforms.Resize((config.training_region_size,config.training_region_size)),
        transforms.ToTensor()
    ])

    if config.use_custom_data:
        ###########################################################################
        transform_no_tensor = transforms.Compose([
            transforms.Resize((config.input_size, config.input_size)),
        ])

        datasets = []
        if config.use_nyu_data:
            nyu = load_nyuv2_data("data/nyuv2/nyu_depth_v2_labeled.mat", extract_data_to_folder=False)
            print("NYU len", len(nyu))
            datasets.append(nyu)
        if config.use_ocid_data:
            if config.split_ocid_images:
                number_all = 2390
                split_ocid_images(ocid_folder_path="data/OCID-dataset", destination_path="data/OCID_splits",
                                  region_size=config.training_region_size, limit_number_samples=config.limit_number_samples,
                                  limit_number_each=config.limit_number_each)  # 2390="all"
                ocid = load_ocid(ocid_folder_path="data/OCID_splits", limit_number_samples="all")
            else:

                ocid = load_ocid(ocid_folder_path="data/OCID-dataset", limit_number_samples="all")
            print(ocid)

            print("OCID len", len(ocid))
            datasets.append(ocid)

        together = concatenate_datasets(datasets)
        together = together.shuffle(seed=123)
        together = together.train_test_split(test_size=0.1)
        train_hf = together["train"]
        val_hf = together["test"]
        train_dataset = DepthDataset(train_hf, "train", train_xform, transform_no_tensor, config)
        val_dataset = DepthDataset(val_hf, "test", val_xform, transform_no_tensor, config)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4,shuffle=True, drop_last=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, num_workers=4, shuffle=False, drop_last=True)

        # rgb_tensor = torch.zeros((len(train_hf), config.inputChannels, config.input_size, config.input_size))


        ###########################################################################
    else:
        train_dataset = AutoencoderDataset("train", train_xform)
        val_dataset   = AutoencoderDataset("val", val_xform)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, shuffle=True)
        val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=4, num_workers=4, shuffle=False)


    util.clear_progress_dir()

    ###################################
    #          Model Setup            #
    ###################################



    autoencoder =WNet()#torch.load('./models/2021-04-05_21_07_54_090637') #WNet()#


    ncutloss_layer = NCutLoss2D()
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()

    optimizerE = torch.optim.Adam(autoencoder.U_encoder.parameters(), lr=0.003)
    optimizerW = torch.optim.Adam(autoencoder.parameters(), lr=0.003)

    if config.debug:
        print(autoencoder)
    util.enumerate_params([autoencoder])

    # Use the current time to save the model at end of each epoch
    modelName = config.model_name



    ###################################
    #          Loss Criterion         #
    ###################################

    def reconstruction_loss(x, x_prime):
        reconloss = F.mse_loss(x, x_prime, reduction='sum')
        return reconloss


    ###################################
    #          Training Loop          #
    ###################################

    autoencoder.train()

    progress_images, progress_expected = next(iter(val_dataloader))

    #schedulerE = torch.optim.lr_scheduler.StepLR(optimizerE, step_size=1480, gamma=0.1)
    #schedulerW = torch.optim.lr_scheduler.StepLR(optimizerW, step_size=1480, gamma=0.1)
    ncutloss = []
    reconloss = []
    for epoch in range(config.num_epochs):
        running_loss = 0.0

        for i, [inputs, outputs] in tqdm.tqdm(enumerate(train_dataloader, 0), desc=f"Epoch {epoch+1} of {config.num_epochs}", total=len(train_dataloader)):


            if config.showdata:
                print(inputs.shape)
                print(outputs.shape)
                print(inputs[0])
                plt.imshow(inputs[0].permute(1, 2, 0))
                plt.show()

            if torch.cuda.is_available():
                inputs  = inputs.cuda()
                outputs = outputs.cuda()

            optimizerE.zero_grad()

            segmentations = autoencoder.forward_encoder(inputs)
            l_soft_n_cut     = ncutloss_layer (segmentations, inputs)
            l_soft_n_cut.backward(retain_graph=False)
            optimizerE.step()
            ncutloss.append(l_soft_n_cut.detach().cpu().numpy())

            optimizerW.zero_grad()

            segmentations, reconstructions = autoencoder.forward(inputs)

            l_reconstruction = reconstruction_loss(
                inputs if config.variationalTranslation == 0 else outputs,
                reconstructions
            )

            reconloss.append(l_reconstruction.detach().cpu().numpy())

            l_reconstruction.backward(
                retain_graph=False)  # We only need to do retain graph =true if we're backpropping from multiple heads
            optimizerW.step()


            # Decrease learining rate by factor 10 every 1000 iterations.
            #schedulerE.step()
            #schedulerW.step()

            # print statistics
            running_loss += l_reconstruction + l_soft_n_cut#loss.item()


            if config.showSegmentationProgress and i == 0: # If first batch in epoch
                util.save_progress_image(autoencoder, progress_images, epoch)

        plot_loss(epoch, reconloss, ncutloss)

        epoch_loss = running_loss / len(train_dataloader.dataset)
        print(f"Epoch {epoch} loss: {epoch_loss:.6f}")

        if config.saveModel:
            util.save_model(autoencoder, modelName)

        # with open('n_cut_loss.pkl','ab') as f:
        #   pickle.dump(ncutloss, f)
        #
        # with open('reconstruction_loss.pkl','ab') as fp:
        #   pickle.dump(reconloss, fp)




if __name__ == "__main__":
    main()
