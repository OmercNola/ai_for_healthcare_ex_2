import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import cv2
from mask_functions import *
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from ipdb import set_trace

# print("Loading information for validation set \n")
# val_infor = get_infor(val_df)

def toTensor(np_array, axis=(0,3,1,2)):
    return np_array.permute(axis)

def toNumpy(tensor, axis=(0,2,3,1)):
    return tensor.detach().cpu().permute(axis).numpy()


def balance_the_data(img_info):

    empty_masks = []
    non_empty_masks = []

    for index in tqdm(range(len(img_info))):

        mask_arr = img_info[index]["mask"]
        mask = np.zeros((512, 512))

        for item in mask_arr:
            if item != "-1":
                mask_ = rle2mask(item, 1024, 1024).T
                mask_ = cv2.resize(mask_, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
                mask[mask_ == 255] = 255

        if np.all(mask == 0):
            empty_masks.append(img_info[index])
        else:
            non_empty_masks.append(img_info[index])

    empty_masks = empty_masks[:int(len(non_empty_masks))]

    return non_empty_masks #+ empty_masks


class MaskDataset(Dataset):

    def __init__(self, df, img_info, transforms=None):
        self.df = df
        self.img_info = balance_the_data(img_info)
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
            sample = {"image": img, "mask": mask}
            sample = self.transforms(**sample)
            img = sample["image"]
            mask = sample["mask"]

        # norm:
        mask = np.expand_dims(mask, axis=-1) / 255.0
        img = np.expand_dims(img, axis=-1) / 255.0

        img = (img - 0.4794 * 1) / (0.2443 * 1)

        return img, mask

    def __len__(self):
        return len(self.img_info)


