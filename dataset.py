from glob import glob
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import numpy as np

train_imgs = sorted(glob('raw_data/dicom-images-train/**/*.dcm', recursive = True))
test_imgs = sorted(glob('raw_data/dicom-images-test/**/*.dcm', recursive = True))

print(f'Number of train files: {len(train_imgs)}')
print(f'Number of test files : {len(test_imgs)}')

train_df = pd.read_csv('raw_data/train-rle.csv')
print(train_df.shape)

# Visualize image mask for file .dcm
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
        break