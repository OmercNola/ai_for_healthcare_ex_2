import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import cv2
from mask_functions import *
from torch.utils.data import Dataset
import torch
from ipdb import set_trace

# print("Loading information for validation set \n")
# val_infor = get_infor(val_df)

def toTensor(np_array, axis=(0,3,1,2)):
    return np_array.permute(axis)

def toNumpy(tensor, axis=(0,2,3,1)):
    return tensor.detach().cpu().permute(axis).numpy()

class MaskDataset(Dataset):

    def __init__(self, df, img_info, transforms=None):
        self.df = df
        self.img_info = img_info
        self.transforms = transforms

    def __getitem__(self, idx):

        img_path = self.img_info[idx]["file_path"]
        key = self.img_info[idx]["key"]

        # load image data
        dataset = pydicom.dcmread(img_path)
        img = dataset.pixel_array
        img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

        mask_arr = self.img_info[idx]["mask"]

        mask = np.zeros((512, 512))

        for item in mask_arr:
            if item != "-1":
                mask_ = rle2mask(item, 1024, 1024).T
                mask_ = cv2.resize(mask_, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
                mask[mask_ == 255] = 255

        if self.transforms:
            sample = {
                "image": img,
                "mask": mask
            }
            sample = self.transforms(**sample)
            img = sample["image"]
            mask = sample["mask"]

        # norm:
        mask = np.expand_dims(mask, axis=-1) / 255.0
        img = np.expand_dims(img, axis=-1) / 255.0

        return img, mask

    def __len__(self):
        return len(self.img_info)