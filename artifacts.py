import os
import PIL
from PIL import Image
import cv2
import numpy as np
import random
from PIL import ImageFilter
from PIL import ImageEnhance
from numpy import asarray,matlib
from tqdm import tqdm

def gaussian_blur(img, kernel_ratio):
    read=Image.open(img)
    height = read.height

    kernel_size = int(height * kernel_ratio)
    blurred = read.filter(ImageFilter.GaussianBlur(radius=kernel_size))
    save_path=""
    try:
        os.makedirs("gaus/"+str(kernel_ratio*100)+"/")
        save_path="gaus/"+str(kernel_ratio*100)+"/"
    except:
        pass

    save_path = "gaus/"+str(kernel_ratio*100)+"/"+"guas_"+str(kernel_ratio)+"_"

    print(save_path)
    file_name_path=(save_path+os.path.split(img)[1])
    blurred=blurred.save(file_name_path)


def motion_blur(img, kernel_ratio):

    read =cv2.imread(img)
    height = read.shape[0]

    kernel_size=int(height*kernel_ratio)

    kernel_v = np.zeros((kernel_size, kernel_size))
    kernel_h = np.copy(kernel_v)

    kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

    kernel_v /= kernel_size
    kernel_h /= kernel_size

    vertical_mb = cv2.filter2D(read, -1, kernel_v)
    horizonal_mb = cv2.filter2D(read, -1, kernel_h)

    save_path=""
    try:
        os.makedirs("motBlur/"+str(kernel_ratio*100)+"/")
        save_path="motBlur/"+str(kernel_ratio*100)+"/"
    except:
        pass

    save_path = "motBlur/"+str(kernel_ratio*100)+"/"+"motblur_"+str(kernel_ratio)+"_"
    # print(img)
    save_choice=random.choice([vertical_mb,horizonal_mb])

    cv2.imwrite(save_path+os.path.split(img)[1], save_choice)


def bit_depth(img):
    read = Image.open(img)

    for i in ([128,64,32,16,8,4,2]):
        lower=read.convert("P",palette=Image.ADAPTIVE, colors=i)
        save_path = ""
        try:
            os.makedirs("lower_bit/" + str(i) + "/")
            save_path = "lower_bit/" + str(i) + "/"
        except:
            pass

        save_path = "lower_bit/" + str(i) + "/" +"_bit_d"+str(i)++"_"
        # print(save_path)
        file_name_path = (save_path+ os.path.split(img)[1][:-3]+".png") #can't save this as jpg so saving as png
        lower=lower.save(file_name_path)

def over_saturate(img):
    read = Image.open(img)
    converter = ImageEnhance.Color(read)
    for i in ([1.33,1.66,2,2.33,2.66,3]):
        saturated = converter.enhance(i)
        save_path = ""
        try:
            os.makedirs("over_saturated/" + str(i) + "/")
            save_path = "over_saturated/" + str(i) + "/"
        except:
            pass

        save_path = "over_saturated/" + str(i) + "/" +"overSat_"+str(i)+"_"
        # print(save_path)
        file_name_path = (save_path+os.path.split(img)[1])
        saturated=saturated.save(file_name_path)

def under_saturate(img):
    read = Image.open(img)
    converter = ImageEnhance.Color(read)
    for i in ([.75,.5,.25]):
        saturated = converter.enhance(i)
        save_path = ""
        try:
            os.makedirs("under_saturated/" + str(i) + "/")
            save_path = "under_saturated/" + str(i) + "/"
        except:
            pass

        save_path = "under_saturated/" + str(i) + "/"+"underSat"+str(i)+"_"
        # print(save_path)
        file_name_path = (save_path+ os.path.split(img)[1])
        saturated = saturated.save(file_name_path)

def bright_and_contrast(img):
    read = Image.open(img)
    brighter = ImageEnhance.Brightness(read)
    # contraster =ImageEnhance.Contrast(read)
    for i in ([1.25,1.5,1.75,2,.75, .5, .25]):
        bright = brighter.enhance(i)
        contraster = ImageEnhance.Contrast(bright)
        final = contraster.enhance(i)
        save_path = ""
        try:
            os.makedirs("bright_contrast/" + str(i) + "/")
            save_path = "bright_contrast/" + str(i) + "/"
        except:
            pass

        save_path = "bright_contrast/" + str(i) + "/"+"brightCont_"+str(i)+"_"
        # print(save_path)
        file_name_path = (save_path+ os.path.split(img)[1])
        final = final.save(file_name_path)

def radial(path):
    img = cv2.imread(path)
    w, h = img.shape[:2]
    center_x = w / 2
    center_y = h / 2
    blur_list = [0.005,0.010,0.015,0.020]
    for blur in blur_list:
        iterations = 5
        growMapx = np.tile(np.arange(h) + ((np.arange(h) - center_x) * blur), (w, 1)).astype(np.float32)
        shrinkMapx = np.tile(np.arange(h) - ((np.arange(h) - center_x) * blur), (w, 1)).astype(np.float32)
        growMapy = np.tile(np.arange(w) + ((np.arange(w) - center_y) * blur), (h, 1)).transpose().astype(np.float32)
        shrinkMapy = np.tile(np.arange(w) - ((np.arange(w) - center_y) * blur), (h, 1)).transpose().astype(np.float32)
        for i in range(iterations):
            tmp1 = cv2.remap(img, growMapx, growMapy, cv2.INTER_LINEAR)
            tmp2 = cv2.remap(img, shrinkMapx, shrinkMapy, cv2.INTER_LINEAR)
            img = cv2.addWeighted(tmp1, 0.5, tmp2, 0.5, 0)

        try:
            os.makedirs("radBlur/" + str(blur) + "/")
            save_path = "radBlur/" + str(blur) + "/"
        except:
            pass

        save_path = "radBlur/" + str(blur) + "/" + "radBlue" + str(blur) + "_"
        # print(img)
        cv2.imwrite(save_path + os.path.split(path)[1], img)

def gaussian_helper(image, rad): #original filename, radius of blur, new filename
    read = Image.open(image)
    img = read.filter(ImageFilter.GaussianBlur(radius = rad))

    return img

def singleside(path, kernel_ratio):
    img=cv2.imread(path)
    width, height = img.shape[:2]
    rad = kernel_ratio*height
    blurimg = gaussian_helper(path, rad)
    src1 = np.array(img)
    src2 = np.array(blurimg)
    maskx, masky = np.meshgrid(np.linspace(0, width, width), np.linspace(0, height, height))
    maskx = maskx / width
    masky = masky / height
    sides=["left", "right", "top", "bottom"]
    side= random.choice(sides)
    if side == 'left':
        mask = maskx
    if side == 'right':
        mask = 1 - maskx
    if side == 'top':
        mask = masky
    if side == 'bottom':
        mask = 1 - masky
    mask = np.tile(np.expand_dims(mask, 2), (3))
    dst = src1 * mask + src2 * (1 - mask)
    finalimg = Image.fromarray(dst.astype(np.uint8))

    try:
        os.makedirs("sideBlur/" + str(kernel_ratio) + "/")
        save_path = "sideBlur/" + str(kernel_ratio) + "/"
    except:
        pass
    save_path = "sideBlur/" + str(kernel_ratio) + "/" + "sideBlur" + str(kernel_ratio) + "_"
    # print(img)
    cv2.imwrite(save_path + os.path.split(path)[1], img)



def main():
    dir="artifacts_test_code/"

    entries = os.listdir(dir)
    for entry in entries:
        path=os.path.join(dir, entry)

        # for i in (.02,.03,.04, .05):
        #     motion_blur(path,i)

        # bit_depth(path)

        # over_saturate(path)

        # radial(path)

        # under_saturate(path)
        # bright_and_contrast(path)

        # for i in (.001,.002,.003,.004,.005):
        #     gaussian_blur(path,i)

        for i in (.001,.002,.003,.004,.005):
            singleside(path,i)


main()


