import numpy as np
from scipy import signal
import random
import cv2

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
        if not self.hairs:
            return img
        
        width, height, _ = img.shape
        
        for _ in range(random.randint(0, self.hairs)):
            # The origin point of the line will always be at the top half of the image
            origin = (random.randint(0, width), random.randint(0, height // 2))
            # The end of the line 
            end = (random.randint(0, width), random.randint(0, height))
            color = (0, 0, 0)  # color of the hair. Black.
            cv2.line(img, origin, end, color, random.randint(self.width[0], self.width[1]))
        
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, width={self.width})'