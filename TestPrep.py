import PIL
import os
import shutil
from PIL import Image
import glob
import random
'''
#gathering random images
files = glob.glob('/Volumes/J_Bac/2021/Datasets/*/*',recursive = True) #Dataset of images
print(len(files)) #Total number of images
for i in tqdm(range(100)): #100 random sample images (in this case)
    random_file = random.choice(files) #Choosing random dataset, and then random image from said dataset
    folder = random_file.split("/")
    name = folder[6] #name of image is 7th term in pathname (in this case)
    img = Image.open(random_file)
    img = img.save("/Volumes/J_Bac/2021/TestImages/"+name) #Destination folder
'''
#resizing and moving to new folder
files = glob.glob("/Volumes/J_Bac/2021/TestImages/*") #Random sample images
print(len(files))
for pic in files:
    image = Image.open(pic)
    folders = pic.split("/")
    a = folders[5]
    b = a.split(".")
    c = b[0] #Isolating image name without extension
    print("old: ", image.size)
    image.thumbnail([512, 512])
    newpath = c + '_thumbnail.jpg'
    image.save(newpath) #new name
    print("new: ", image.size)
    source = "/Users/jacob/"+newpath #Source and Destination folders for moving resized images
    destination = "/Volumes/J_Bac/2021/ResizedTest"
    shutil.move(source, destination)
