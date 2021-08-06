
from PIL import Image
import glob
from tqdm import tqdm
import random
import os
import pandas as pd
import shutil

#gathering random images
files = glob.glob('/Volumes/J_Bac/2021/Datasets/*/*',recursive = True) #Dataset of images

folderlist = ['TestImages','ResizedTest'] #Clear previous test of images
for folder in folderlist:
    dir = '/Volumes/J_Bac/2021/'+folder
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

print('Gathering Random Sample of Test Images:')
for i in tqdm(range(100)): #100 random sample images
    random_file = random.choice(files) #Choosing random image from total list
    folder = random_file.split("/")
    name = folder[6] #name of image is 7th term in pathname (in this case)
    nm = name.split(".")
    name = nm[0]+".jpg"
    img = Image.open(random_file)
    img = img.save("/Volumes/J_Bac/2021/TestImages/"+name) #Destination folder

#resizing and moving to new folder
files = glob.glob("/Volumes/J_Bac/2021/TestImages/*") #Random sample images

print('Resizing Images:')
for pic in tqdm(files):
    image = Image.open(pic)

    folders = pic.split("/")
    a = folders[5]
    b = a.split(".")
    c = b[0] #Isolating image name without extension
    
    image.thumbnail([512, 512])
    newpath = c + '_thumbnail.jpg'
    image.save(newpath) #new name

    source = "/Users/jacob/"+newpath #Source and Destination folders for moving resized images
    destination = "/Volumes/J_Bac/2021/ResizedTest"
    shutil.move(source, destination)

#isolating meta for only test images
imglist = []
DSlist = []
Dialist = []
LIDlist = []
Parlist = []

files = glob.glob('/Volumes/J_Bac/2021/TestImages/*', recursive=True)
df = pd.read_excel("/Volumes/J_Bac/2021/Datasets.xlsx", sheet_name='Images', engine='openpyxl', usecols='A:E')

print('Isolating Metadata for Images:')
for i in tqdm(range(len(df.index))): #For each image name, find corresponding metadata in Datasets file  
    fname = df.loc[i, 'Filename']
    fname1 = fname.lower()
    for f in files:
        imname = f.split('/')[-1]
        imname = imname.lower()
        if fname1 == imname:
            imglist.append(fname)

            DS = df.loc[i, 'Dataset Source']
            DSlist.append(DS)
            
            Dia = df.loc[i, 'Diagnosis']
            Dialist.append(Dia)
            
            LID = df.loc[i, 'Lesion ID']
            LIDlist.append(LID)
            
            Par = df.loc[i, 'Partition']
            Parlist.append(Par)

#Create new dataframe with gathered metadata, and save as csv to be used in main.py
newmeta = pd.DataFrame(list(zip(imglist, DSlist, Dialist, LIDlist, Parlist)), columns = ['Filename', 'Dataset Source', 'Diagnosis', 'Lesion ID', 'Partition'])
newmeta.to_csv('/Volumes/J_Bac/2021/TestMeta.csv', index = False)