
import argparse
import os
from torchvision import transforms
import torchvision
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import RandomApply
from tqdm import tqdm
from Dataset_blur import ImageDataset as ID
from Augmentations import blur, DrawHair, randomblur

#For now the variables are hard coded to the code. Need to change this.
data_dir = '/Volumes/J_Bac/Falses/QA/data'
target_dir = '/Volumes/J_Bac/2021/results'
if not os.path.exists(target_dir): #Folder containing all experiment folders 
    os.mkdir(target_dir)

## Transform parameters
crop_size = int(256)
resize_size = int(320)
blurSigma = 0.2

## Data Transformer(s) definition
train_transforms= transforms.Compose([
    transforms.Resize(resize_size),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)], p=0.2), # p=0.8
    transforms.RandomCrop(crop_size),
    transforms.RandomApply([DrawHair()],p=1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_transforms_blur= transforms.Compose([
    transforms.Resize(resize_size),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)], p=0.2), # p=0.8
    transforms.RandomCrop(crop_size),
    transforms.RandomApply([transforms.Lambda(lambda x:randomblur(x))],p=1),
    transforms.RandomApply([DrawHair()],p=1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

## Main Code Starts here

data_df = pd.read_csv('/Volumes/J_Bac/Falses/QA_Multi_Temp_Final.csv') #ADD CSV PATH FOR IMAGE METADATA
dataset = ID(data_df,data_dir,transform = train_transforms, transform_blur = train_transforms_blur)
data_loader = DataLoader(dataset, batch_size = 128,shuffle=True)
batch = next(iter(data_loader))
grid = torchvision.utils.make_grid(batch[0], nrow=8,padding=20)
torchvision.utils.save_image(grid,'/Volumes/J_Bac/2021/Augmented_Images.png')

print('Processing Done')