#Import Necessary Libraries
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchtoolbox.transform as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from PIL import Image, ImageFilter, ImageEnhance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy  import signal
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import gc, os, cv2, time, datetime, warnings, random
from torchsummary import summary
import datetime
today = datetime.date.today()
from numpy import asarray
import io
from random import randrange

# At least fixing some random seeds. 
# It is still impossible to make results 100% reproducible when using GPU
warnings.simplefilter('ignore')
torch.manual_seed(47)
np.random.seed(47)

#Select the GPU that will be used for the experiments
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Dataset class for the Dataloaders. 
class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, im_folder: str, train: bool = True, transform = None, transform_blur = None):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            im_folder (str): folder with images
            train (bool): flag of whether a training dataset is being initialized or testing one
            transforms: image transformation method to be applied
            
        """
        self.df = df
        self.transform = transform
        self.transform_blur = transform_blur
        self.train = train
        self.im_folder = im_folder
        
    def __getitem__(self, index):
        im_path = os.path.join(self.im_folder, self.df.iloc[index]['ImageID'])
        #x = Image.open(im_path)
        x = cv2.imread(im_path)
        
        if self.train:
            y = self.df.iloc[index][1:].astype(int).to_numpy()
            if y[5] == 1 or self.transform_blur is None:
                x = self.transform(x)
            else:
                rand_decimal = random.randint(0, 100)/100
                if rand_decimal >= 0.4:
                    x = self.transform(x)
                else:
                    y[5] = 1
                    x = self.transform_blur(x)
            return x, y
        else:
            return x
    
    def __len__(self):
        return len(self.df)
    
    def getLabel (self,index):
        return self.df.iloc[index]['target']
    
    def getFname (self,index):
        return self.df.iloc[index]['image_name']
    
## Helper Model modules 

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
    
class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

#Auxilary functions for Data Augmentation

#Apply histogram equalization
def histEq(x):
    clahe = cv2.createCLAHE(clipLimit=0.5,tileGridSize=(8,8))
    lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    x = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return x

#Image Sharpen
def sharpen(x):
    # Create kernel
    kernel = np.array([[0, -0.5, 0], 
                       [-0.5, 3,-0.5], 
                       [0, -0.5, 0]])
    return cv2.filter2D(x, -1, kernel)

#Add noise to the image
def jitter(x,NoiseLevel):
    NoiseLevel = np.random.rand(1)*NoiseLevel
    img = np.array(x)
    img = img+(np.random.randn(*img.shape)*NoiseLevel)
    #x = Image.fromarray(np.uint8(np.clip(img,0,255)))
    x = np.uint8(np.clip(img,0,255))
    return x

#Blockout a rectangular region in the image (set pixel values to zero)
def blockout(x,blockSize=16):
    if np.random.rand(1)>0.5:
        return x
    blockSize = np.random.randint(blockSize,blockSize*3)
    img = np.array(x)
    shp = img.shape
    blockLoc = np.random.randint(0,np.min(shp[0:2])-blockSize,2)
    img[blockLoc[0]:blockLoc[0]+blockSize,blockLoc[1]:blockLoc[1]+blockSize] = 0
    #x = Image.fromarray(np.uint8(np.clip(img,0,255)))
    x = np.uint8(np.clip(img,0,255))
    return x

#Random blur the image
def blur(x,blurSigma=1.):
    if np.random.rand(1)>0.8:
        return x
    blurSigma = np.max([0.001,np.random.rand(1)*blurSigma])
    img = np.array(x)
    kx, ky = np.meshgrid(np.linspace(-1,1,7), np.linspace(-1,1,7))
    d = np.sqrt(kx*kx+ky*ky)
    g = np.exp(-( (d)**2 / ( 2.0 * (blurSigma**2) ) ))
    g /= np.sum(g)
    for i in range(3):
        img[:,:,i] = signal.fftconvolve(img[:,:,i], g[:, :], mode='same')
    #x = Image.fromarray(np.uint8(np.clip(img,0,255)))
    x = np.uint8(np.clip(img,0,255))
    return x

#Add hair artifact to the image
class DrawHair:
    """
    Draw a random number of pseudo hairs
    Args:
        hairs (int): maximum number of hairs to draw
        width (tuple): possible width of the hair in pixels
    """

    def __init__(self, hairs:int = 4, width:tuple = (1, 2)):
        self.hairs = hairs
        self.width = width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on.
        Returns:
            PIL Image: Image with drawn hairs.
        """
        PIL_flag = 0
        if not self.hairs:
            return img

        if type(img) is Image.Image:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            PIL_flag = 1

        width, height, _= img.shape
        
        for _ in range(random.randint(0, self.hairs)):
            # The origin point of the line will always be at the top half of the image
            origin = (random.randint(0, width), random.randint(0, height // 2))
            # The end of the line 
            end = (random.randint(0, width), random.randint(0, height))
            color = (0, 0, 0)  # color of the hair. Black.
            cv2.line(img, origin, end, color, random.randint(self.width[0], self.width[1]))
        
        if PIL_flag == 1:
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, width={self.width})'

class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        y = torch.rot90(x, angle,[1,2])
        return y
    

#Jacob's Functions
def compress(x, qual, newfile = ''):
    img = io.BytesIO()
    x.save(img, format = 'JPEG',quality = qual)
    if newfile != '':
        newimg = np.array(Image.open(img))
        newimg = Image.fromarray(newimg)
        newimg.save(newfile)
    return Image.open(img)

def gaussian(x, rad, newfile = ''): #original filename, radius of blur, new filename
    x = Image.fromarray(x.astype('uint8'))
    img = x.filter(ImageFilter.GaussianBlur(radius = rad))
    if newfile != '':
        img.save(newfile)
    img = asarray(img)
    return img

def bitdepth(x, newfile = ''): #original filename, new filename
    data = asarray(x)
    newdata = data/32
    newdata = np.clip((np.around(newdata))*32, 0, 255)
    img = Image.fromarray(newdata.astype('uint8'))
    if newfile != '':
        img.save(newfile)
    return img

def motionblur(x, rad, newfile = ''): #original filename, radius of blur, new filename
    #x = asarray(x)
    kernel_motion_blur = np.zeros((rad, rad))
    kernel_motion_blur[int((rad-1)/2), :] = np.ones(rad)
    kernel_motion_blur = kernel_motion_blur / rad
    prob = random.random()
    if prob <0.5:
        kernel_motion_blur = np.transpose(kernel_motion_blur)
    img = cv2.filter2D(x, -1, kernel_motion_blur)
    if newfile != '':
        cv2.imwrite(newfile, img)
    return img

def radial(x, newfile = ''): #original filename, new filename
    #x = asarray(x)
    w, h = x.shape[:2]
    center_x = w / 2
    center_y = h / 2
    blur = 0.015
    growMapx = np.tile(np.arange(h) + ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
    shrinkMapx = np.tile(np.arange(h) - ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
    growMapy = np.tile(np.arange(w) + ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)
    shrinkMapy = np.tile(np.arange(w) - ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)
    for i in range(5):
        tmp1 = cv2.remap(x, growMapx, growMapy, cv2.INTER_LINEAR)
        tmp2 = cv2.remap(x, shrinkMapx, shrinkMapy, cv2.INTER_LINEAR)
        img = cv2.addWeighted(tmp1, 0.5, tmp2, 0.5, 0)
    #img = Image.fromarray(img.astype('uint8'))
    if newfile != '':
        cv2.imwrite(newfile,x)
    return img

def colorshift(x, factor, newfile = ''): #original filename, degree of color (default = 1), new filename
    enhancer = ImageEnhance.Color(x)
    img = enhancer.enhance(factor)
    if newfile != '':
        img.save(newfile)
    return img

def singleside(x, rad, side, newfile = ''): #original filename, radius of blur, side to be blurred, new filename
    width, height,_ = x.shape
    blurimg = gaussian(x, rad)
    src1 = np.array(x)
    src2 = np.array(blurimg)
    maskx, masky = np.meshgrid(np.linspace(0, width, width), np.linspace(0, height, height))
    maskx = maskx / width
    masky = masky / height
    if side == 'left':
        mask = maskx
    if side == 'right':
        mask = 1 - maskx
    if side == 'top':
        mask = masky
    if side == 'bottom':
        mask = 1 - masky
    mask =  np.tile(np.expand_dims(mask, 2), (3))
    img = src1 * mask + src2 * (1 - mask)
    #img = Image.fromarray(dst.astype(np.uint8))
    if newfile != '':
        img.save(newfile)
    return img

def resize(x, width, height, newfile = ''): #original filename, desired width, desired height, new filename
    size = (width, height)
    img = x.thumbnail(size)
    if newfile != '':
        img.save(newfile)
    return img

def normalize(x, newfile = ''): #original filename, new filename
    img = cv2.normalize(x, x, 0, 255, cv2.NORM_MINMAX)
    if newfile != '':
        cv2.imwrite(newfile, img)
    return img

def randomblur(x, newfile = ''): #original filename, new filename
    blurlist = ['gaussian', 'motion', 'radial', 'singleside']
    blurtype = random.choice(blurlist)
    blurmount = int(randrange(30))
    blurmount = np.max([blurmount,5])
    if blurtype == 'gaussian':
        img = gaussian(x, blurmount, newfile)
    elif blurtype == 'motion':
        img = motionblur(x, blurmount, newfile)
    elif blurtype == 'radial':
        img = radial(x, newfile)
    elif blurtype == 'singleside':
        sidelist = ['left', 'right', 'top', 'bottom']
        randside = random.choice(sidelist)
        img = singleside(x, blurmount, randside, newfile)
    img = np.array(img)
    return img
    
#Setup the dataloaders

#Augmentation Parameters
NoiseLevel = 2
blockSize = 32
blurSigma = 0.2
InputSize = (256,256)
crop_size = int(256)
resize_size = int(256)

#Train data loader
train_transform= transforms.Compose([
    transforms.Resize(InputSize),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)], p=0.2), # p=0.8
    #transforms.RandomCrop(crop_size),
    #transforms.RandomApply([DrawHair()],p=1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_transform_blur= transforms.Compose([
    transforms.Resize(InputSize),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)], p=0.2), # p=0.8
    #transforms.RandomCrop(crop_size),
    transforms.RandomApply([transforms.Lambda(lambda x:randomblur(x))],p=1),
    #transforms.RandomApply([DrawHair()],p=1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

#Validation data loader
val_transform = transforms.Compose([
    transforms.Resize(size=InputSize),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

#Test data loader
test_transform = transforms.Compose([
    transforms.Resize(size=InputSize),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])


#Set the variables that show the location of the training (validation) data and the labels.
train_df = pd.read_csv('/media/Proc/Data/QA/QA_Multi_Temp_Final.csv')
train_imdir = '/media/Proc/Data/QA/data'

#Set the variables that show the location of the test (validation) data.
test_df = pd.read_csv('/media/Proc/Data/QA/QA_Multi_Temp_Final.csv')
test = ImageDataset(df=test_df, 
                       im_folder='/media/Proc/Data/QA/data',
                       transform=test_transform)
## In this current version both training and test data points to the same

class Net(nn.Module):
    def __init__(self, arch, numClasses):
        super(Net, self).__init__()
        self.arch = arch
        self.arch._fc = nn.Linear(in_features=1280, out_features=640, bias=False)
        self.arch._fcOut = nn.Linear(in_features=640, out_features=numClasses, bias=True)
        self.swish = MemoryEfficientSwish()
                
    def forward(self, x):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        x = self.arch(x)
        x = self.swish(x)
        x_out = self.arch._fcOut(x)
        return x_out

#Set the variables that show the location of the test (validation) data.
test_df = pd.read_csv('/media/Proc/Data/QA/QA_Multi_Temp_Final.csv')
test = ImageDataset(df=test_df, 
                       im_folder='/media/Proc/Data/QA/data',
                       transform=test_transform)

skf = KFold(n_splits=5, random_state=990, shuffle=True)
epochs = 100  # Number of epochs to run
es_patience = 15  # Early Stopping patience - for how many epochs with no improvements to wait
tta = 3
batchSize = 128 #64 #32
lr = 0.1
numClasses = 8
labelBalance = 0
cont = 0 #Flag for continuing from a former checkpoint

oof = np.zeros((len(train_df), numClasses))  # Out Of Fold predictions
preds = torch.zeros((len(test), numClasses), dtype=torch.float32, device=device)  # Predictions for test test

# We stratify by target value, thus, according to sklearn StratifiedKFold documentation
# We can fill `X` with zeroes of corresponding length to use it as a placeholder since we only need `y` to stratify the data
for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.zeros(len(train_df))), 1):
    print('=' * 20, 'Fold', fold, '=' * 20)
    model_path = ("CLASSIFIER_MODEL_MultiLabelv2_{}_fold{}.pth".format(today, fold))  # Path and filename to save model to
    print(model_path)
    
    best_val = None  # Best validation score within this fold
    patience = es_patience  # Current patience counter
    
    #Setup the model 
    #(i) load the pretrained efficient net 
    #(ii) add the final fully connected layers (classification layers)
    arch = EfficientNet.from_pretrained('efficientnet-b1')  # Going to use efficientnet-b2 NN architecture
    model = Net(arch,numClasses)
    
    #move the model to the respective device (in our case the selected GPU)
    model = model.to(device)
    
    #Shows the summary of the model layers (good for initial development and debugging)
    #summary(model,(3,InputSize,InputSize)) 
    if cont:
        model = torch.load(model_path)
    
    #Setup the optimizer 
    #optim = torch.optim.Adam(model.parameters(), lr=lr)
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov = True)
    
    # Reducing learning rate if no progress for `patience` epochs
    scheduler = ReduceLROnPlateau(optimizer=optim, mode='max', patience=5, verbose=True, )
    
    #Define the loss function
    #criterion = nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    
    ##### Training loops starts here #####
    for epoch in range(epochs):
        
        #During every epoch we reinitialize the data loader after randomization
        tmp = train_idx
        tmp_val = val_idx
        
        # I prefer to manually randomize the order of the data to be fed into the model. Hoping, this will be handy for future development
        random.shuffle(tmp)
        train     = ImageDataset(df=train_df.iloc[tmp].reset_index(drop=True), im_folder=train_imdir, transform=train_transform, transform_blur = train_transform_blur)
        val       = ImageDataset(df=train_df.iloc[val_idx].reset_index(drop=True), im_folder=train_imdir, transform=val_transform)  
        
        #Setup the dataloaders after randomization.
        train_loader     = DataLoader(dataset=train, batch_size=batchSize, shuffle=False, num_workers=1)
        val_loader       = DataLoader(dataset=val, batch_size=batchSize, shuffle=False, num_workers=1)
        test_loader      = DataLoader(dataset=test, batch_size=batchSize, shuffle=False, num_workers=1)
        
        ## Start Processing the data
        start_time = time.time()
        correct = 0
        epoch_loss = 0
        val_loss = 0
        #Model is on the training mode, so that batchnorms are updated or dropouts are applied
        model.train()
            
        #Batch Processing for training data starts here
        train_preds = torch.zeros((len(train_idx), numClasses), dtype=torch.float32, device=device)
        GT        = torch.zeros((len(train_idx), numClasses), dtype=torch.float32, device=device)
        for j, (x, y) in tqdm(enumerate(train_loader)):#tqdm(train_loader):
            #load the batch
            x = torch.tensor(x, device=device, dtype=torch.float32)
            y = torch.tensor(y, device=device, dtype=torch.float32)
            
            #Zero out the gradients
            optim.zero_grad()
            #batch through the model
            z = model(x)
            #Calculate the loss
            loss = criterion(z,y)
    
            loss.backward()
            optim.step()
            pred = torch.round(torch.sigmoid(z))  #Since this is a multiclass classification problem, argmax gives the final classification label
            # tracking number of correctly predicted samples and the amount of loss
            train_preds[j*x.shape[0]:j*x.shape[0] + x.shape[0],:] = pred
            GT         [j*x.shape[0]:j*x.shape[0] + x.shape[0],:] = y 
            epoch_loss += loss.item()
        train_roc = roc_auc_score(torch.reshape(GT[:],(-1,)).cpu(), torch.reshape(train_preds,(-1,)).detach().cpu())
        #train_acc = accuracy_score(GT.cpu(), torch.round(train_preds.detach().cpu()))

        
        #EVALUATE THE TRAINED MODELS FOR THE CURRENT EPOCH AND CHECK EARLY STOPING CONDITION
        model.eval()  # switch model to the evaluation mode so that batchnorms are not updated or dropouts are not applied
        val_preds = torch.zeros((len(val_idx), numClasses), dtype=torch.float32, device=device)
        GT        = torch.zeros((len(val_idx), numClasses), dtype=torch.float32, device=device)
        with torch.no_grad():  # Do not calculate gradient since we are only predicting
            # Predicting on validation set
            for j, (x_val, y_val) in enumerate(val_loader):
                x_val = torch.tensor(x_val, device=device, dtype=torch.float32)
                y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
                z_val = model(x_val)
                val_pred = torch.round(torch.sigmoid(z_val))
                val_preds[j*x_val.shape[0]:j*x_val.shape[0] + x_val.shape[0],:] = val_pred
                GT       [j*x_val.shape[0]:j*x_val.shape[0] + x_val.shape[0],:] = y_val
            val_roc = roc_auc_score(torch.reshape(GT[:],(-1,)).cpu(), torch.reshape(val_preds,(-1,)).cpu())
            #val_acc = accuracy_score(GT.cpu(), torch.round(val_preds.cpu()))
            
            #Print out the results
            print('=' * 40)
            print('Epoch {:03}: | Loss: {:.3f} | Train roc: {:.3f} | Val roc: {:.3f} | Training time: {}'.format(
            epoch + 1, 
            epoch_loss, 
            train_roc, 
            val_roc, 
            str(datetime.timedelta(seconds=time.time() - start_time))))
            print('=' * 40)
                
            #Check the early stoping criteria     
            scheduler.step(val_roc)
            # During the first iteration (first epoch) best validation is set to None
            if not best_val:
                best_val = val_roc  # So any validation roc_auc we have is the best one for now
                torch.save(model, model_path)  # Saving the model
                continue  
                
            if val_roc >= best_val:
                best_val = val_roc
                patience = es_patience  # Resetting patience since we have new best validation accuracy
                torch.save(model, model_path)  # Saving current best model
            else:
                patience -= 1
                if patience == 0:
                    print('Early stopping. Best Val Accuracy: {:.3f}'.format(best_val))
                    break

    #PREDICT OOF WITH THE BEST MODEL.
    model = torch.load(model_path)  # Loading best model of this fold
    model.eval()  # switch model to the evaluation mode
    val_preds = torch.zeros((len(val_idx), numClasses), dtype=torch.float32, device=device)
    with torch.no_grad():
        # Predicting on validation set once again to obtain data for OOF
        for j, (x_val, y_val) in enumerate(val_loader):
            x_val = torch.tensor(x_val, device=device, dtype=torch.float32)
            y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
            z_val = model(x_val)
            val_pred = torch.sigmoid(z_val)
            val_preds[j*x_val.shape[0]:j*x_val.shape[0] + x_val.shape[0]] = val_pred
        oof[val_idx,:] = val_preds.cpu().numpy()
    
    # Cleaning variables to fix memory issue
    del train, val, train_loader, val_loader, x, y, x_val, y_val
    gc.collect()
    np.savetxt("oofv2_2021_{}_batch{}_lr{}_patience{}_fold{}.csv".format(today, batchSize, lr, es_patience, fold), oof, delimiter=",")
np.savetxt("oofv2_2021_{}_batch{}_lr{}_patience{}_Final.csv".format(today, batchSize, lr, es_patience, fold), oof, delimiter=",")