
from PIL import Image, ImageFilter#, ImageEnhance
import numpy as np
from numpy import asarray
import cv2

#Sample Image
Mole = "/Users/jacob/Desktop/Sci_Res/Sample.jpeg"


def compress(file, qual, newfile): #original filename, desired quality (default = 95), new filename
    image = Image.open(file)
    image.save(newfile, quality = qual)
    image.show()

#compress(Mole, 0, "compress.jpg")

def gaussian(file, rad, newfile): #original filename, radius of blur, new filename
    image = Image.open(file)
    image = image.filter(ImageFilter.GaussianBlur(radius = rad))
    image.save(newfile)
    image.show()

#gaussian(Mole, 50, "guassian.jpg")

def bitdepth(file, newfile): #original filename, new filename
    image = Image.open(file)
    data = asarray(image)
    newdata = data/32
    newdata = np.clip((np.around(newdata))*32, 0, 255)
    imdata = Image.fromarray(newdata.astype('uint8'))
    imdata.save(newfile)
    imdata.show()
    image.show()

#bitdepth(Mole, 'bitdepth.jpg')


#Motion Blur
'''
def motionblur(file, size, newfile):
    img = cv2.imread(file)
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    output = cv2.filter2D(img, -1, kernel_motion_blur)
    cv2.imwrite(newfile, output)
#motionblur(Mole, 20, 'motionblur.jpg')
'''


def radial(file, newfile): #original filename, new filename
    img = cv2.imread(file)
    w, h = img.shape[:2]

    center_x = w / 2
    center_y = h / 2
    blur = 0.01
    iterations = 5

    growMapx = np.tile(np.arange(h) + ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
    shrinkMapx = np.tile(np.arange(h) - ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
    growMapy = np.tile(np.arange(w) + ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)
    shrinkMapy = np.tile(np.arange(w) - ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)

    for i in range(iterations):
        tmp1 = cv2.remap(img, growMapx, growMapy, cv2.INTER_LINEAR)
        tmp2 = cv2.remap(img, shrinkMapx, shrinkMapy, cv2.INTER_LINEAR)
        img = cv2.addWeighted(tmp1, 0.5, tmp2, 0.5, 0)
    cv2.imwrite(newfile,img)

#radial(Mole, 'radial.jpg')
'''
def colorshift(file, factor, newfile): #original filename, degree of color (default = 1), new filename
    enhancer = ImageEnhance.Color(file)
    enhancer.enhance(factor).save(newfile)

colorshift(Mole, 2, '/Users/jacob/Desktop/newmole.jpg')


def singleside(file, side, rad): #original filename, side of image to be blurred (left, right, top, bottom), radius of blur, new filename
    img = Image.open(file)
    width, height = img.size
    if side == 'left':
        w1 = 0
        h1 = height
        w2 = width/2
        h2 = 0
    elif side == 'right':
        w1 = width/2
        h1 = height
        w2 = width
        h2 = 0
    elif side == 'top':
        w1 = 0
        h1 = height
        w2 = width
        h2 = height/2
    elif side == 'bottom':
        w1 = 0
        h1 = height/2
        w2 = width
        h2 = 0
    cropimg = img.crop((w1, h1, w2, h2))
    blurimg = cropimg.filter(ImageFilter.GaussianBlur(radius = rad))
    img.paste(blurimg,(w1, h1, w2, h2))
    img.show()

singleside(Mole, 'left', 20)
'''
