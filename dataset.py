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
from glob import glob
import warnings
import ast
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


class Covid_Dataset(Dataset):

    def __init__(self):

        self.train_imgs = sorted(glob('raw_data/covid-19/train/**/*.dcm', recursive=True))
        self.test_imgs = sorted(glob('raw_data/covid-19/test/**/*.dcm', recursive=True))
        self.df_train_img = pd.read_csv('raw_data/covid-19/_image_level.csv')

        print(f'Number of train files: {len(self.train_imgs)}')
        print(f'Number of test files : {len(self.test_imgs)}')
        print(f'shape of df_train_img : {self.df_train_img.shape}')

        missing = 0
        patients_data = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            for paths in tqdm(self.train_imgs):

                patient = {}
                data = pydicom.dcmread(paths)
                try:
                    image = self.df_train_img[self.df_train_img['StudyInstanceUID'] == data.StudyInstanceUID]
                    if image.shape[0] == 1:
                        patient["Findings"] = image['label']
                        patient["Name"] = data.PatientName
                        patient["StudyInstanceUID"] = data.StudyInstanceUID
                        patient['Pixels'] = data.pixel_array
                        patient["Sex"] = data.PatientSex
                        patient["Modality"] = data.Modality
                        patient["BodyPart"] = data.BodyPartExamined
                        patient["filepath"] = paths
                        patients_data.append(patient)
                except:
                    missing += 1

        df_patients = pd.DataFrame(patients_data, columns=["Findings", "Name", "StudyInstanceUID", "Pixels",
                                                           "Sex", "Modality", "BodyPart", "filepath"])

        print("images with labels: ", df_patients.shape[0])
        box_list, masks = [], []
        self.train_set = []

        for x in df_patients["StudyInstanceUID"].values:  # all data
            example2_meta = self.df_train_img[self.df_train_img['StudyInstanceUID'] == x]
            if "none" not in str(df_patients[df_patients["StudyInstanceUID"] == x]["Findings"].values[0]):
                bbox = example2_meta['boxes'].item()
                # fig, axes = plt.subplots(1, 2, figsize=(20, 15))
                img = df_patients.loc[df_patients['StudyInstanceUID'] == x]["Pixels"].values[0]

                # for box in ast.literal_eval(bbox):
                #     x1, y1, x2, y2 = box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]
                #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 80, 255), 15)
                #     # axes[0].imshow(img, cmap='bone')
                #     # axes[0].set_title('Image+box')
                #     box_list.append(img)

                masks = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

                for box in ast.literal_eval(bbox):
                    x1, y1, x2, y2 = box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    cv2.rectangle(masks, (x1, y1), (x2, y2), (255, 255, 255), -1)

                # axes[1].imshow(masks, cmap='bone')
                # axes[1].set_title('mask')

                self.train_set.append(
                    [
                        cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC),
                        cv2.resize(masks, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
                    ]
                )

        self.train_set = self.train_set[:2000]
        # fig, axes = plt.subplots(1, 2, figsize=(20, 15))
        # for img, mask in self.train_set:
        #     axes[0].imshow(img, cmap='bone')
        #     axes[0].set_title('Image')
        #     axes[1].imshow(mask, cmap='bone')
        #     axes[1].set_title('mask')
        #     plt.show()
        #     set_trace()

    def __getitem__(self, idx):

        img, mask = self.train_set[idx]

        # norm:
        mask = np.expand_dims(mask, axis=-1) / 255.0
        img = np.expand_dims(img, axis=-1) / 255.0

        img = (img - 0.4794 * 1) / (0.2443 * 1)

        return img, mask

    def __len__(self):
        return len(self.train_set)

class Covid_Dataset_test(Dataset):

    def __init__(self):

        self.train_imgs = sorted(glob('raw_data/covid-19/train/**/*.dcm', recursive=True))
        self.test_imgs = sorted(glob('raw_data/covid-19/test/**/*.dcm', recursive=True))
        self.df_train_img = pd.read_csv('raw_data/covid-19/_image_level.csv')

        print(f'Number of train files: {len(self.train_imgs)}')
        print(f'Number of test files : {len(self.test_imgs)}')
        print(f'shape of df_train_img : {self.df_train_img.shape}')

        missing = 0
        patients_data = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            for paths in tqdm(self.train_imgs):

                patient = {}
                data = pydicom.dcmread(paths)
                try:
                    image = self.df_train_img[self.df_train_img['StudyInstanceUID'] == data.StudyInstanceUID]
                    if image.shape[0] == 1:
                        patient["Findings"] = image['label']
                        patient["Name"] = data.PatientName
                        patient["StudyInstanceUID"] = data.StudyInstanceUID
                        patient['Pixels'] = data.pixel_array
                        patient["Sex"] = data.PatientSex
                        patient["Modality"] = data.Modality
                        patient["BodyPart"] = data.BodyPartExamined
                        patient["filepath"] = paths
                        patients_data.append(patient)
                except:
                    missing += 1

        df_patients = pd.DataFrame(patients_data, columns=["Findings", "Name", "StudyInstanceUID", "Pixels",
                                                           "Sex", "Modality", "BodyPart", "filepath"])

        print("images with labels: ", df_patients.shape[0])
        box_list, masks = [], []
        self.train_set = []

        for x in df_patients["StudyInstanceUID"].values:  # all data
            example2_meta = self.df_train_img[self.df_train_img['StudyInstanceUID'] == x]
            if "none" not in str(df_patients[df_patients["StudyInstanceUID"] == x]["Findings"].values[0]):
                bbox = example2_meta['boxes'].item()
                # fig, axes = plt.subplots(1, 2, figsize=(20, 15))
                img = df_patients.loc[df_patients['StudyInstanceUID'] == x]["Pixels"].values[0]

                # for box in ast.literal_eval(bbox):
                #     x1, y1, x2, y2 = box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]
                #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 80, 255), 15)
                #     # axes[0].imshow(img, cmap='bone')
                #     # axes[0].set_title('Image+box')
                #     box_list.append(img)

                masks = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

                for box in ast.literal_eval(bbox):
                    x1, y1, x2, y2 = box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    cv2.rectangle(masks, (x1, y1), (x2, y2), (255, 255, 255), -1)

                # axes[1].imshow(masks, cmap='bone')
                # axes[1].set_title('mask')

                self.train_set.append(
                    [
                        cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC),
                        cv2.resize(masks, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
                    ]
                )

        self.train_set = self.train_set[2000:]
        # fig, axes = plt.subplots(1, 2, figsize=(20, 15))
        # for img, mask in self.train_set:
        #     axes[0].imshow(img, cmap='bone')
        #     axes[0].set_title('Image')
        #     axes[1].imshow(mask, cmap='bone')
        #     axes[1].set_title('mask')
        #     plt.show()
        #     set_trace()

    def __getitem__(self, idx):

        img, mask = self.train_set[idx]

        # norm:
        mask = np.expand_dims(mask, axis=-1) / 255.0
        img = np.expand_dims(img, axis=-1) / 255.0

        img = (img - 0.4794 * 1) / (0.2443 * 1)

        return img, mask

    def __len__(self):
        return len(self.train_set)