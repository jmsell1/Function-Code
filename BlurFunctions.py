
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from numpy import asarray
import cv2

#Sample Image
Mole = "/Users/jacob/Desktop/Sci_Res/Sample.jpeg"

#FIX (quality and save are together)
def compress(file, qual, newfile): #original filename, desired quality (default = 95), new filename
    image = Image.open(file)
    image.save(newfile, quality = qual)
    image.show()
#compress(Mole, 0, "compress.jpg")

def gaussian(image, rad, newfile = ''): #original filename, radius of blur, new filename
    image = image.filter(ImageFilter.GaussianBlur(radius = rad))
    if newfile != '':
        image.save(newfile)
    return image
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

def motionblur(img, size, newfile = ''): #original filename, size of blur, new filename
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
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

def singleside(img, rad, newfile = ''):
    width = img.size[0]
    blurimg = gaussian(img, rad)
    src1 = np.array(img)
    src2 = np.array(blurimg)
    mask1 = np.linspace(0, width, int(width+1))#.resize(src1.shape[1::-1]), Image.BILINEAR
    dst = src1 * mask1 + src2 * int(1 - mask1)
    print(dst.shape)
    finalimg = Image.fromarray(dst.astype(np.uint8))
    if newfile != '':
        finalimg.save(newfile)
    return finalimg
img = Image.open(Mole)
singleside(img, 20)
