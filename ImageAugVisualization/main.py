import argparse
import os
from torchvision import transforms
import torchvision

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from Dataset import ImageDataset as ID
from Augmentations import blur, DrawHair

#For now the variables are hard coded to the code. Need to change this.
data_dir = './dataset'
target_dir = './results'
if not os.path.exists(target_dir): #Folder containing all experiment folders 
    os.mkdir(target_dir)

## Transform parameters
crop_size = int(256)
resize_size = int(320)
blurSigma = 0.2

## Data Transformer definition
train_transforms= transforms.Compose([
    transforms.Resize(resize_size),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)], p=0.2), # p=0.8
    transforms.RandomCrop(crop_size),
    transforms.RandomApply([transforms.Lambda(lambda x:blur(x,blurSigma))],p=1),
    transforms.RandomApply([DrawHair()],p=1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

## Main Code Starts here

data_df = pd.read_csv(csv_path)
dataset = ID(data_df,data_dir,transform = train_transforms)
data_loader = DataLoader(dataset, batch_size = 128,shuffle=True)

batch = next(iter(data_loader))
grid = torchvision.utils.make_grid(batch[0], nrow=8,padding=20)
torchvision.utils.save_image(grid,'Augmented_Images.png')

print('Processing Done')