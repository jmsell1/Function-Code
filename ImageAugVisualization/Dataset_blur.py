
import torch
import torch.nn as nn 
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np 
import pandas as pd 
import os
import random

class ImageDataset(Dataset):
    
    def __init__(self, df: pd.DataFrame, im_folder: str, transform = None, transform_blur = None, include_src = False):
        
        self.df = df
        self.transform = transform
        self.transform_blur = transform_blur
        self.im_folder = im_folder

        #### Changed target --> fine_grain_target (smh)
        self.classes = self.df['blur'].unique() ###### Should have been like this the whole time ! 
        self.targets = list(df["blur"])

        self.include_src = include_src
        if self.include_src:
            self.src     = list(df['src'])

    def __getitem__(self,index):
        im_path = os.path.join(self.im_folder, self.df.iloc[index]['ImageID'])
        img = Image.open(im_path) #Image.fromarray(im_path)#cv2.imread(im_path)
        target = self.df.iloc[index]['blur']
        if target == 1:
            img = self.transform(img)
        else:
            rand_decimal = random.randint(0, 100)/100
            if rand_decimal >= 0.2:
                img = self.transform(img)
            elif rand_decimal < 0.2:
                target = 1
                img = self.transform_blur(img)
        # return target
        if self.include_src:
            src = self.df.iloc[index]['src']
            return img, src
        else:
            return img,target

    
    def __len__(self):
        return len(self.df)
    
    def getLabel(self, index):
        return self.df.iloc[index]['blur']
    
    def getFname(self,index):
        return self.df.iloc[index]['ImageID']