from tqdm import tqdm
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from mask_functions import *
from multiprocessing import Pool
from ipdb import set_trace
import torch
from pathlib import Path
import os

def parallel_func(image_id, df, file_paths):

    index_ = list(filter(lambda x: image_id in file_paths[x], range(len(file_paths))))
    full_image_path = file_paths[index_[0]]

    # Get all segment encode
    record_arr = df[df["ImageId"] == image_id]
    encode_pixels = []
    for _, row in record_arr.iterrows():
        encode_pixels.append(row[" EncodedPixels"])

    return {
        "key": image_id,
        "file_path": full_image_path,
        "mask": encode_pixels
    }


def get_infor(df, parallel_func):

    image_id_arr = df["ImageId"].unique()
    with Pool(20) as p:
        res = p.map(parallel_func, image_id_arr)
    return res


def Visualize_image(train_df, train_imgs):

    image_id_arr = train_df["ImageId"].unique()
    for index, image_id in enumerate(image_id_arr):
        index_ = list(filter(lambda x: image_id in train_imgs[x], range(len(train_imgs))))
        dataset = pydicom.dcmread(train_imgs[index_[0]])
        image_data = dataset.pixel_array

        record_arr = train_df[train_df["ImageId"] == image_id]
        # Visualize patient has multi segment
        if len(record_arr) >= 2:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_figheight(15)
            fig.set_figwidth(15)
            ax1.imshow(image_data, cmap=plt.cm.bone)
            ax2.imshow(image_data, cmap=plt.cm.bone)
            mask = np.zeros((1024, 1024))

            for _, row in record_arr.iterrows():

                if row[" EncodedPixels"] != ' -1':
                    mask_ = rle2mask(row[" EncodedPixels"], 1024, 1024).T
                    mask[mask_ == 255] = 255
            ax2.imshow(mask, alpha=0.3, cmap="Blues")
            plt.show()
            break


def visualize_dataset(train_dataset, parallel_visualize):

    temp = []
    for counter, (img, mask) in enumerate(train_dataset):
        if counter == 10:
            break
        temp.append((img, mask))

    with Pool(10) as p:
        p.map(parallel_visualize, temp)


def parallel_visualize(img_and_mask):

    img, mask = img_and_mask

    img = img[:, :, 0]
    mask = mask[:, :, 0]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    ax1.imshow(img, cmap=plt.cm.bone)
    ax2.imshow(img, cmap=plt.cm.bone)
    ax2.imshow(mask, alpha=0.3, cmap="Blues")
    plt.show()


def plot_image_during_training(outputs, masks, imgs_gpu, batch_counter, train_or_test_image):

    pred_ = torch.round(outputs[0]) * 255
    pred_ = pred_.detach().cpu().permute(1, 2, 0).numpy()
    pred_ = np.reshape(pred_, (512, 512))

    mask = torch.round(masks[0]) * 255
    mask = mask.detach().cpu().permute(1, 2, 0).numpy()
    mask_ = np.reshape(mask , (512, 512))

    img_origin = imgs_gpu[0]
    img_origin = np.reshape(img_origin.cpu().numpy(), (512, 512))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    ax1.imshow(img_origin, cmap=plt.cm.bone)
    ax1.set_title('img_origin')
    ax2.imshow(img_origin, cmap=plt.cm.bone)
    ax2.imshow(pred_, alpha=0.3, cmap="Blues")
    ax2.set_title('pred')
    ax3.imshow(img_origin, cmap=plt.cm.bone)
    ax3.imshow(mask_, alpha=0.3, cmap="Blues")
    ax3.set_title('img_origin + mask')


    path_to_image_folder = Path('./eval_images')
    os.makedirs(str(path_to_image_folder), exist_ok=True)

    path_to_image_file = Path(str(path_to_image_folder) + f'/_{batch_counter}_{train_or_test_image}')

    fig.savefig(f'{str(path_to_image_file)}')  # save the figure to file
    print(f'saved image at {str(path_to_image_folder)}/_{batch_counter}_{train_or_test_image}')
    plt.close(fig)  # close the figure window
    # plt.show()

