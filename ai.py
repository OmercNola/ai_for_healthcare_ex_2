import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import pydicom
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
import warnings
warnings.simplefilter("ignore")
# %matplotlib inline
from glob import glob
train_img = sorted(glob('/content/drive/MyDrive/ai_for_healthcare_ex2/dicom-images-train/*.dcm'))
test_img = sorted(glob('/content/drive/MyDrive/ai_for_healthcare_ex2/dicom-images-test/*.dcm'))

print(f'Number of train files:{len(train_img)}')
print(f'Number of test files :{len(test_img)}')

df = pd.read_csv('/content/drive/MyDrive/ai_for_healthcare_ex2/train-rle.csv')
print(df.shape)

def show_imageInfo(dataset):
    print("Filename :", file_path)

    print("Patient's name :", dataset.PatientName.family_name + ", " + dataset.PatientName.given_name)
    print("Patient id :", dataset.PatientID)
    print("Patient's Age :", dataset.PatientAge)
    print("Patient's Sex :", dataset.PatientSex)
    print("Modality :", dataset.Modality)
    print("Body Part Examined :", dataset.BodyPartExamined)
    print("View Position :", dataset.ViewPosition)
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)

def plot_pixels(dataset, figsize=(8,8)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()

file_path = train_img[3]
data = pydicom.dcmread(file_path)
print(data)
show_imageInfo(data)
plot_pixels(data)

missing = 0
multiple = 0
patients_data = []

for k,paths in enumerate(train_img):
    patient = {}
    img_id = paths.split('/')[-1]
    data = pydicom.dcmread(paths)
    try:
        image = df[df['ImageId'] == data.file_meta.MediaStorageSOPInstanceUID]
        
        if image.shape[0] > 1: 
            multiple += 1
        rle = image[' EncodedPixels'].values
        if rle[0] == '-1':
            pixels = rle[0]
        else:    
            pixels = [i for i in rle]
        
        
        patient["UID"] = data.SOPInstanceUID
        patient['EncodedPixels'] = pixels
        patient["Age"] = data.PatientAge
        patient["Sex"] = data.PatientSex
        patient["Modality"] = data.Modality
        patient["BodyPart"] = data.BodyPartExamined
        patient["ViewPosition"] = data.ViewPosition
        patient["filepath"] = paths
        patients_data.append(patient)
    except:
        missing += 1

df_patients = pd.DataFrame(patients_data, columns=["UID", "EncodedPixels", "Age", 
                            "Sex", "Modality", "BodyPart", "ViewPosition", "filepath"])

df_patients['Pneumothorax'] = df_patients['EncodedPixels'].apply(lambda x:0 if x == '-1' else 1)
df_patients['Pneumothorax'] = df_patients['Pneumothorax'].astype('int')
print("images with labels: ", df_patients.shape[0])
df_patients



non_healthy = df_patients[df_patients['Pneumothorax'] == 1] 
healthy = df_patients[df_patients['Pneumothorax'] == 0]

df_patients['Age'] = df_patients['Age'].astype('int') 
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(20,8))
sns.distplot(df_patients['Age'],ax=axes[0])
axes[0].title.set_text('Distribution of Age')
sns.distplot(healthy['Age'],ax=axes[1])
sns.distplot(non_healthy['Age'],ax=axes[1],color='#B71C1C')
axes[1].title.set_text('Distribution of Age(healthy vs unhealthy)')
plt.show()

viewpos = df_patients['ViewPosition'].values

pa =viewpos[viewpos =="PA"].shape[0]
ap= viewpos[viewpos =="AP"].shape[0]

basic_palette = sns.color_palette()
plt.pie([pa, ap], labels = ["PA", "AP"], colors=[basic_palette[-2], basic_palette[4]], autopct='%1.1f%%', startangle=70)
plt.title("Occurrences of View positions", fontsize=16)

male = df_patients[df_patients['Sex'] == 'M'] 
female = df_patients[df_patients['Sex'] == 'F'] 

non_healthy_M = male[male['Pneumothorax'] == 1] 
healthy_M = male[male['Pneumothorax'] == 0]
non_healthy_F = female[female['Pneumothorax'] == 1] 
healthy_F = female[female['Pneumothorax'] == 0]

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(20,8))
sns.distplot(healthy_M['Age'],ax=axes[0])
sns.distplot(non_healthy_M['Age'],ax=axes[0],color='#B71C1C')
axes[0].set_title('Male Patient vs Age')
sns.distplot(healthy_F['Age'],ax=axes[1])
sns.distplot(non_healthy_F['Age'],ax=axes[1],color='#B71C1C')
axes[1].set_title('Female Patient vs Age')
plt.suptitle(f'Distribution of Age base on Sex(healthy vs unhealthy)')
plt.show()






for file in train_img[0:20]:
    data = pydicom.dcmread(file)
    image = data.pixel_array
    id_ = '.'.join(file.split('/')[-1].split('.')[:-1])
    fig,axes = plt.subplots(1,2,figsize=(10,5))
    axes[0].imshow(image,cmap='bone')
    axes[1].imshow(image,cmap='bone')
    plt.suptitle(f'Image id: {id_}')
    plt.show()

import pydicom,os,cv2
def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return rmin, cmin, rmax, cmax 


def plot_imgs(n,df):
    for i in range(n):
        idx = np.random.randint(0,df.shape[0])
        tmp = df.iloc[idx]
        path = tmp['filepath']
        encoding = tmp['EncodedPixels']
        image = pydicom.dcmread(path).pixel_array
        fig, axes = plt.subplots(1,4, figsize=(20,15))
        axes[0].imshow(image, cmap='bone')
        axes[0].set_title('Image')
        mask = rles2mask(encoding,image.shape[0],image.shape[1])
        axes[1].imshow(mask,cmap='gray')
        axes[1].set_title('Mask')
        axes[2].imshow(image,cmap='bone')
        axes[2].imshow(mask,alpha=0.3,cmap='Reds')
        axes[2].set_title('Image + mask')
        rmin, cmin, rmax, cmax = bounding_box(mask)
        image_rgb = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        cv2.rectangle(image_rgb, (cmin,rmin),(cmax,rmax), (255,255,0), 5)
        axes[3].imshow(image_rgb)
        axes[3].imshow(mask,alpha=0.3,cmap='Reds')
        axes[3].set_title('Image Box Annoted')
        plt.show()

tmp = df_patients[df_patients['Pneumothorax'] == 1].reset_index(drop=True)
plot_imgs(20,tmp)