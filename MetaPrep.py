
#Isolate metadata for images within TestImages folder
#Read all images in folder
#For each image name, find corresponding metadata in Datasets excel through pd  
#Copy entire row (loop?) into separate pd dataframe
#Save second pd as csv to be used in Main.py

import pandas as pd
from tqdm import tqdm
import glob

imglist = []
DSlist = []
Dialist = []
LIDlist = []
Parlist = []

files = glob.glob('/Volumes/J_Bac/2021/TestImages/*', recursive=True)
df = pd.read_excel("/Volumes/J_Bac/2021/Datasets.xlsx", sheet_name='Images', engine='openpyxl', usecols='A:E')

for i in tqdm(range(len(df.index))):
    fname = df.loc[i, 'Filename']
    for f in files:
        imname = f.split('/')[-1]
        if fname == imname:
            imglist.append(fname)

            DS = df.loc[i, 'Dataset Source']
            DSlist.append(DS)
            
            Dia = df.loc[i, 'Diagnosis']
            Dialist.append(Dia)
            
            LID = df.loc[i, 'Lesion ID']
            LIDlist.append(LID)
            
            Par = df.loc[i, 'Partition']
            Parlist.append(Par)

newmeta = pd.DataFrame(list(zip(imglist, DSlist, Dialist, LIDlist, Parlist)), columns = ['Filename', 'Dataset Source', 'Diagnosis', 'Lesion ID', 'Partition'])
newmeta.to_csv('/Volumes/J_Bac/2021/TestMeta.csv', index = False)