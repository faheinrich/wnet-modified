import os, shutil
from config import Config
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

config = Config()

# Clear progress images directory
def clear_progress_dir(): # Or make the dir if it does not exist
    if not os.path.isdir(config.segmentationProgressDir):
        os.mkdir(config.segmentationProgressDir)
    else: # Clear the directory
        for filename in os.listdir(config.segmentationProgressDir):
            filepath = os.path.join(config.segmentationProgressDir, filename)
            os.remove(filepath)

def enumerate_params(models):
    num_params = 0
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                num_params += param.numel()
    print(f"Total trainable model parameters: {num_params}")

def save_model(autoencoder, modelName):
    path = os.path.join("./models/", modelName.replace(":", " ").replace(".", " ").replace(" ", "_"))
    torch.save(autoencoder, path)
    with open(path+".config", "a+") as f:
        f.write(str(config))
        f.close()

def save_progress_image(autoencoder, progress_images, epoch):
    if not torch.cuda.is_available():
        segmentations, reconstructions = autoencoder(progress_images)
    else:
        segmentations, reconstructions = autoencoder(progress_images.cuda())

    channels = reconstructions.shape[1]
    if channels == 3:
        n_rows = 3
    elif channels == 6:
        n_rows = 5
    elif channels == 7:
        n_rows = 7

    f, axes = plt.subplots(n_rows, config.val_batch_size, figsize=(12, 10))
    for i in range(config.val_batch_size):
        segmentation = segmentations[i]
        pixels = torch.argmax(segmentation, axis=0).float() / config.k # to [0,1]

        channels_cut = 3


        axes[0, i].imshow(pixels.detach().cpu()) # segmentation prediction
        axes[1, i].imshow(progress_images[i].permute(1, 2, 0)[:,:,:channels_cut]) # rgb
        axes[2, i].imshow(reconstructions[i].detach().cpu().permute(1, 2, 0)[:,:,:channels_cut]) # rgb reconstruction

        if channels == 6:
            axes[3, i].imshow(progress_images[i].detach().cpu().permute(1, 2, 0)[:, :, channels_cut:])
            axes[4, i].imshow(reconstructions[i].detach().cpu().permute(1, 2, 0)[:, :, channels_cut:])
        if channels == 7:
            axes[3, i].imshow(progress_images[i].detach().cpu().permute(1, 2, 0)[:, :, 3:4])
            axes[4, i].imshow(reconstructions[i].detach().cpu().permute(1, 2, 0)[:, :, 3:4])
            axes[5, i].imshow(progress_images[i].detach().cpu().permute(1, 2, 0)[:, :, channels_cut+1:])
            axes[6, i].imshow(reconstructions[i].detach().cpu().permute(1, 2, 0)[:, :, channels_cut+1:])
        # if config.variationalTranslation:
        #     axes[3, i].imshow(progress_expected[i].detach().cpu().permute(1, 2, 0))
    plt.savefig(os.path.join(config.segmentationProgressDir, str(epoch)+".png"), dpi=300)
    plt.close(f)
