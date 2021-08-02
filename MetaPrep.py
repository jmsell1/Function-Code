
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

for i in tqdm(range(len(df.index))): #For each image name, find corresponding metadata in Datasets file  
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

#Create new dataframe with gathered metadata, and save as csv to be used in main.py
newmeta = pd.DataFrame(list(zip(imglist, DSlist, Dialist, LIDlist, Parlist)), columns = ['Filename', 'Dataset Source', 'Diagnosis', 'Lesion ID', 'Partition'])
newmeta.to_csv('/Volumes/J_Bac/2021/TestMeta.csv', index = False)