from datetime import datetime
import os

class Config():
    def __init__(self):
        # self.model_name = str(datetime.now())
        self.model_name = "ocid_all"

        self.debug = False
        self.input_size = 32 # Side length of square image patch
        self.batch_size = 10
        self.val_batch_size = 4
        self.test_batch_size = 1 
        self.verbose_testing = True

        self.use_custom_data = True
        self.use_nyu_data = False # depth in meter originally
        self.use_ocid_data = True # depth in milimeter
        self.split_ocid_images = False
        self.training_region_size = 32
        self.limit_number_samples = 1000
        self.limit_number_each = 10

        self.inputChannels = 7
        self.inputChannels = 7
        self.use_rgb = True
        self.use_depth = True
        self.normalize_depth = False
        self.use_normals = True
        self.normalize_normals = False

        self.k = 64 # Number of classes
        self.num_epochs = 50 #250 for real
        self.data_dir = "data/BSDS500val"  # Directory of images
        self.showdata = False # Debug the data augmentation by showing the data we're training on.

        self.useInstanceNorm = False # Instance Normalization
        self.useBatchNorm = True # Only use one of either instance or batch norm
        self.useDropout = True
        self.drop = 0.65

        # Each item in the following list specifies a module.
        # Each item is the number of input channels to the module.
        # The number of output channels is 2x in the encoder, x/2 in the decoder.
        self.encoderLayerSizes = [64, 128, 256, 512]
        self.decoderLayerSizes = [1024,512, 256]

        self.showSegmentationProgress = True
        self.segmentationProgressDir = f'./latent_images/{self.model_name}'

        self.variationalTranslation = 0 # Pixels, 0 for off. 1 works fine

        self.saveModel = True


