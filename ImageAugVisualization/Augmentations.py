
import numpy as np
from scipy import signal
import random
import cv2

from PIL import Image, ImageFilter, ImageEnhance
from numpy import asarray
import io
import random
from random import randrange

#Random blur the image
def blur(x,blurSigma=1.):
    #if np.random.rand(1)>0.8:
    #    return x
    blurSigma = np.max([0.001,np.random.rand(1)*blurSigma])
    img = np.array(x)
    kx, ky = np.meshgrid(np.linspace(-1,1,25), np.linspace(-1,1,25))
    d = np.sqrt(kx*kx+ky*ky)
    g = np.exp(-( (d)**2 / ( 2.0 * (blurSigma**2) ) ))
    g /= np.sum(g)
    for i in range(3):
        img[:,:,i] = signal.fftconvolve(img[:,:,i], g[:, :], mode='same')
    #x = Image.fromarray(np.uint8(np.clip(img,0,255)))
    x = np.uint8(np.clip(img,0,255))
    return x

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
    img = x.filter(ImageFilter.GaussianBlur(radius = rad))
    if newfile != '':
        img.save(newfile)
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
    x = asarray(x)
    kernel_motion_blur = np.zeros((rad, rad))
    kernel_motion_blur[int((rad-1)/2), :] = np.ones(rad)
    kernel_motion_blur = kernel_motion_blur / rad
    img = cv2.filter2D(x, -1, kernel_motion_blur)
    if newfile != '':
        cv2.imwrite(newfile, img)
    return img

def radial(x, newfile = ''): #original filename, new filename
    x = asarray(x)
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
    img = Image.fromarray(img.astype('uint8'))
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
    width, height = x.size
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
    dst = src1 * mask + src2 * (1 - mask)
    img = Image.fromarray(dst.astype(np.uint8))
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
    