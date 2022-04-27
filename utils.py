from tqdm import tqdm
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from mask_functions import *

def get_infor(df, file_paths):

    infor = []
    image_id_arr = df["ImageId"].unique()
    for index, image_id in tqdm(enumerate(image_id_arr), total=12089):
        index_ = list(filter(lambda x: image_id in file_paths[x], range(len(file_paths))))
        full_image_path = file_paths[index_[0]]

        # Get all segment encode
        record_arr = df[df["ImageId"]==image_id]
        encode_pixels = []
        for _, row in record_arr.iterrows():
            encode_pixels.append(row[" EncodedPixels"])

        infor.append({
            "key": image_id,
            "file_path": full_image_path,
            "mask": encode_pixels
        })
    return infor

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