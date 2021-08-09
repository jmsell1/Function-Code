
from BlurFunctions import *
import glob
import os
import cv2
from tqdm import tqdm
import Augmentor
import shutil
from PIL import Image

path = "/Volumes/J_Bac/2021/ResizedTest/output"
if os.path.isdir(path) == True:
    shutil.rmtree(path)
else:
    pass
folder = "/Volumes/J_Bac/2021/ResizedTest"
files = glob.glob(folder+'/*', recursive=True)

#Parameters
qual = 10 #Image quality for compression (original = 95)
gb = 5 #Radius of blur for gaussian
mb = 20 #Radius of blur for motionblur
ss = 30 #Radius of blur for singleside
factor = 1.5 #Degree of color for colorshift (original = 1)
side = 'left' #Desired side of image to be blurred for singleside (left, right, top, bottom)
width, height = 256, 256 #Desired width and height of image for resize

folderlist = ['BitDepth', 'ColorShift', 'Compress', 'Gaussian', 'MotionBlur', 'Radial', 'Resize', 'SingleSide', 'Normalize']
for folder in folderlist:
    dir = '/Volumes/J_Bac/2021/Test1/'+folder
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

print('Running Functions:')
for file in tqdm(files):
    opened = Image.open(file)
    openedcv2 = cv2.imread(file)
    file = file.split('/')[5]

    new = compress(opened, qual)
    filename = file.split('thumbnail')
    name = filename[0]+'C_'+str(qual)+filename[1]
    new.save('/Volumes/J_Bac/2021/Test1/Compress/'+name)
    
    new = gaussian(opened, gb)
    filename = file.split('thumbnail')
    name = filename[0]+'gb_'+str(gb)+filename[1]
    new.save('/Volumes/J_Bac/2021/Test1/Gaussian/'+name)
    
    new = bitdepth(opened)
    filename = file.split('thumbnail')
    name = filename[0]+'bd'+filename[1]
    new.save('/Volumes/J_Bac/2021/Test1/BitDepth/'+name)

    new = motionblur(openedcv2, mb)
    filename = file.split('thumbnail')
    name = filename[0]+'mb_'+str(mb)+filename[1]
    cv2.imwrite('/Volumes/J_Bac/2021/Test1/MotionBlur/'+name, new)

    new = radial(openedcv2)
    filename = file.split('thumbnail')
    name = filename[0]+'rb'+filename[1]
    cv2.imwrite('/Volumes/J_Bac/2021/Test1/Radial/'+name, new)

    new = colorshift(opened, factor)
    filename = file.split('thumbnail')
    name = filename[0]+'cs_'+str(factor)+filename[1]
    new.save('/Volumes/J_Bac/2021/Test1/ColorShift/'+name)

    new = singleside(opened, ss, side)
    filename = file.split('thumbnail')
    name = filename[0]+'ss_'+str(ss)+'_'+str(side)+filename[1]
    new.save('/Volumes/J_Bac/2021/Test1/SingleSide/'+name)

    new = resize(opened, width, height)
    nw, nh = new.size
    filename = file.split('thumbnail')
    name = filename[0]+'R_'+str(nw)+','+str(nh)+filename[1]
    new.save('/Volumes/J_Bac/2021/Test1/Resize/'+name)

    new = normalize(openedcv2)
    filename = file.split('thumbnail')
    name = filename[0]+'N'+filename[1]
    cv2.imwrite('/Volumes/J_Bac/2021/Test1/Normalize/'+name, new)

p = Augmentor.Pipeline("/Volumes/J_Bac/2021/ResizedTest")
p.skew_tilt(probability = 1)
p.sample(len(files)) #"output" folder is saved within folder of images (i.e. ResizedTest)
