# import numpy
import torch

import pdb
import sys
import torch
import torch.nn as nn
import torchvision.utils as tvis
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import random
import os
#---import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
#import matplotlib.pyplot as plt
import random
import os
#import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data as data
import pdb
import datetime
from PIL import Image, ImageDraw
import numpy as np
import cv2 as cv
import albumentations
#C:\Users\user\Desktop\DontTouchPLSpytorch\Polygon\img_saved\CheckAccuracy
def uniq_pix(dictionary,path,name=0):
    if(name==0):
        names=os.listdir(path)
        for name in names:
            path_current=path+name
            with Image.open(path_current) as mask_original:
                height,width=mask_original.size
                mask_original=np.asarray(mask_original)
                mask_original=mask_original[:,:,0:3] # <<<<<<<<<< first three channels because fourth chanllen in PNH is always has value = 255
                for w in range(width):
                    for h in range(height):
                        # wtf=mask_original[h,w]
                        # print(mask_original[h,w])
                        pix_current=mask_original[h,w].tobytes()
                        if pix_current in dictionary:
                            dictionary[pix_current]+=1
                        else:
                            dictionary[pix_current]=1
    else:
        path=path+name
        with Image.open(path) as mask_original:
            height,width=mask_original.size
            mask_original=np.asarray(mask_original)
            mask_original=mask_original[:,:,0:3] # <<<<<<<<<< first three channels because fourth chanllen in PNH is always has value = 255
            for w in range(width):
                for h in range(height):
                    pix_current=mask_original[h,w].tobytes()
                    if pix_current in dictionary:
                        dictionary[pix_current]+=1
                    else:
                        dictionary[pix_current]=1
#-------HYPER PARAMS-------------
path="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/Original3/"
# uniq pixels
# C:\Users\user\Desktop\DontTouchPLSpytorch\Polygon\img_saved\CleanMapForTestOnly1
#path="C:/Users/user/Desktop/DontTouchPLSpytorch/data/DB_2020-01-30/Temp/12210/"
#labels_big=os.listdir()

# mask_original_path_label5="C:/Users/user/Desktop/DontTouchPLSpytorch/data/DB_2020-01-30/Temp/12210/"
# name="LABEL_5.png"

pixels_classes = np.array(
    [[255, 255, 255, 255],
     [250, 10, 222, 255],
     [0, 255, 0, 255],
     [255, 0, 0, 255],
     [50, 0, 0, 255],
     [30, 0, 0, 255],
     [0, 0, 255, 255]])
pixels_classes=pixels_classes[:,0:3]
# path_mask_original_img="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/CreatingMaskClassesNotImg/"
path_mask_original_img="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/mask_classes_backup/trainB/" # <<< correct
name="313.png"

with Image.open(path_mask_original_img+name) as img_mask:
    uniq_pixels_dict={}
    uniq_pix(uniq_pixels_dict,path_mask_original_img,name)
    print(len(uniq_pixels_dict))
    img_mask=np.asarray(img_mask)
    img_mask=img_mask[:,:,0:3]
    width, height, channels=img_mask.shape
    list_of_pixels=[]
    for k in uniq_pixels_dict.keys():
        list_of_pixels.append(np.frombuffer(k,dtype=np.uint8))
        wtf=np.frombuffer(k,dtype=np.uint8)
        print(k, np.frombuffer(k,dtype=np.uint8))
        print(uniq_pixels_dict[k])

for pixel in list_of_pixels:
    index=np.where((pixels_classes==pixel).all(axis=1))
    print(index)
    # for w in range (width):
    #     for h in range(height):
print(pixels_classes)
print("end")
