# import libraries
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.image as image
from tqdm import tqdm
import pydicom
import sys
import os
from functools import partial
"====================================="
import torch
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
import torchvision.models as models
import torchvision.transforms as T
import albumentations as A
from torch.autograd import Variable
"====================================="
from dataset import MaskDataset
from utils import get_infor, Visualize_image
from glob import glob
from ipdb import set_trace
"================================"
from model import Unet
from PIL import Image
from utils import parallel_func


if __name__ == '__main__':

    # read data:
    train_imgs = sorted(glob('raw_data/dicom-images-train/**/*.dcm', recursive=True))
    test_imgs = sorted(glob('raw_data/dicom-images-test/**/*.dcm', recursive=True))
    print(f'Number of train files: {len(train_imgs)}')
    print(f'Number of test files : {len(test_imgs)}')
    train_df = pd.read_csv('raw_data/train-rle.csv')

    print("Loading information for training set \n")
    # set_trace()
    parallel_func = partial(parallel_func, df=train_df, file_paths=train_imgs)
    train_infor = get_infor(train_df, parallel_func)
    print("information has been loaded ! \n")

    # Visualize image and mask:
    Visualize_image(train_df, train_imgs)

    # create transforms:
    train_transform = A.Compose([
        A.HorizontalFlip(),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            ], p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.3),
        A.ShiftScaleRotate(),
    ])

    # create train datatset and dataloader:
    train_dataset = MaskDataset(train_df, train_infor, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, drop_last=True)

    number_visualize = 1
    for img, mask in train_dataset:
        if number_visualize > 15:
            break
        img = img[:, :, 0]
        mask = mask[:, :, 0]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        ax1.imshow(img, cmap=plt.cm.bone)
        ax2.imshow(img, cmap=plt.cm.bone)
        ax2.imshow(mask, alpha=0.3, cmap="Blues")
        number_visualize += 1
        plt.show()

    # val_dataset = MaskDataset(val_df, val_infor)
    # val_loader = DataLoader(val_dataset, batch_size=20, shuffle=True, drop_last=True)

    model_ft = models.resnet50(pretrained=True)
    model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
    model = Unet(model_ft)

