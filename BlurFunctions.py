
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from numpy import asarray
import cv2
import io

#Sample Image
Mole = "/Users/jacob/Desktop/Sci_Res/Sample.jpeg"

def compress(img, qual, newfile = ''):
    output = io.BytesIO()
    img.save(output, format = 'JPEG',quality = qual)
    if newfile != '':
        newimg = np.array(Image.open(output))
        newimg = Image.fromarray(newimg)
        newimg.save(newfile)
    return Image.open(output)
img = Image.open(Mole)
compress(img, 10)

def gaussian(image, rad, newfile = ''): #original filename, radius of blur, new filename
    img = image.filter(ImageFilter.GaussianBlur(radius = rad))
    if newfile != '':
        img.save(newfile)
    return img
image = Image.open(Mole)
gaussian(image, 10)

def bitdepth(image, newfile = ''): #original filename, new filename
    data = asarray(image)
    newdata = data/32
    newdata = np.clip((np.around(newdata))*32, 0, 255)
    imdata = Image.fromarray(newdata.astype('uint8'))
    if newfile != '':
        imdata.save(newfile)
    return imdata
image = Image.open(Mole)
bitdepth(image)

def motionblur(img, rad, newfile = ''): #original filename, radius of blur, new filename
    kernel_motion_blur = np.zeros((rad, rad))
    kernel_motion_blur[int((rad-1)/2), :] = np.ones(rad)
    kernel_motion_blur = kernel_motion_blur / rad
    output = cv2.filter2D(img, -1, kernel_motion_blur)
    if newfile != '':
        cv2.imwrite(newfile, output)
    return output
img = cv2.imread(Mole)
motionblur(img, 20)

def radial(img, newfile = ''): #original filename, new filename
    w, h = img.shape[:2]
    center_x = w / 2
    center_y = h / 2
    blur = 0.015
    iterations = 5
    growMapx = np.tile(np.arange(h) + ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
    shrinkMapx = np.tile(np.arange(h) - ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
    growMapy = np.tile(np.arange(w) + ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)
    shrinkMapy = np.tile(np.arange(w) - ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)
    for i in range(iterations):
        tmp1 = cv2.remap(img, growMapx, growMapy, cv2.INTER_LINEAR)
        tmp2 = cv2.remap(img, shrinkMapx, shrinkMapy, cv2.INTER_LINEAR)
        img = cv2.addWeighted(tmp1, 0.5, tmp2, 0.5, 0)
    if newfile != '':
        cv2.imwrite(newfile,img)
    return img
img = cv2.imread(Mole)
radial(img)

def colorshift(image, factor, newfile = ''): #original filename, degree of color (default = 1), new filename
    enhancer = ImageEnhance.Color(image)
    enhanced = enhancer.enhance(factor)
    if newfile != '':
        enhanced.save(newfile)
    return enhanced
image = Image.open(Mole)
colorshift(image, 2)

def singleside(img, rad, side, newfile = ''): #original filename, radius of blur, side to be blurred, new filename
    width, height = img.size
    blurimg = gaussian(img, rad)
    src1 = np.array(img)
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
    finalimg = Image.fromarray(dst.astype(np.uint8))
    if newfile != '':
        finalimg.save(newfile)
    return finalimg
img = Image.open(Mole)
singleside(img, 20, 'left')
