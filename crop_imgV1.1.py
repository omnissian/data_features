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


# tmp = np.frombuffer(key, dtype=type_ar.dtype)
#-------HYPER PARAMS-------------
batch_size=4
momentum=0.9
weight_decay=1e-9
# loss=nn.CrossEntropyLoss()
loss=nn.MSELoss()

learn_rate=3e-3
epochs=100
step_save=50

#------------random crop---------------
pathBig1="C:/Users/user/Desktop/DontTouchPLSpytorch/data/DB_2020-01-30/Temp/12210/"
pathBig2="C:/Users/user/Desktop/DontTouchPLSpytorch/data/DB_2020-01-30/Temp/12220/"
pathBig3="C:/Users/user/Desktop/DontTouchPLSpytorch/data/DB_2020-01-30/Temp/40155/"
pathBig4="C:/Users/user/Desktop/DontTouchPLSpytorch/data/DB_2020-01-30/Temp/40172/"
pathBig5="C:/Users/user/Desktop/DontTouchPLSpytorch/data/DB_2020-01-30/Temp/952172/"
pathSaveBadMask="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/CleanMapLabel2/trainA/"
pathSaveMask="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/CleanMapLabel2/trainB/"

# pathBig2="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/DBMMA/anthr/another_osm_maps_orig/"
# pathBig3="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/DBMMA/anthr/another_osm_maps_orig/"
# pathBig4="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/DBMMA/anthr/another_osm_maps_orig/"
# pathSave="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/DBMMA/anthr/anothcrpjpg/" # JPG
# pathSavePNG="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/DBMMA/anthr/anothcrppng/"

all_paths=[]
all_paths.append(pathBig1)
all_paths.append(pathBig2)
all_paths.append(pathBig3)
all_paths.append(pathBig4)
all_paths.append(pathBig5)


def randomCropSolo(img,newH=256,newW=256):
    curH,curW,_=img.shape
    assert curH>=newH
    assert curW>=newW
    x=random.randint(0,curH-newH)
    y=random.randint(0,curW-newW)
    img_crop=img[x:x+newH,y:y+newW]

    return img_crop

def randomCrop(img,mask,newH=256,newW=256):
    curH,curW,_=img.shape
    assert curH>=newH
    assert curW>=newW
    x=random.randint(0,curH-newH)
    y=random.randint(0,curW-newW)
    img_crop=img[x:x+newH,y:y+newW]
    mask=mask[x:x+newH,y:y+newW]
    return img_crop,mask

num_img_crop=100


path_iter=0
for path in all_paths:
    with Image.open(path+"google_maps.png") as bad_mask, Image.open(path+"LABEL_5.png") as mask:
        for iter in range(num_img_crop):
            ch4=np.full((256,256,1),255,dtype=np.uint8) # check width and height
            mask=np.array(mask)
            bad_mask=np.array(bad_mask)
            bad_mask_crop,mask_crop=randomCrop(bad_mask,mask)
            # bad_mask_crop=randomCrop(bad_mask)
            print ("old mask=",mask.shape)
            print ("new img=",mask_crop.shape)
            mask_crop=np.concatenate((mask_crop,ch4),axis=2)
            mask_crop_png=Image.fromarray(mask_crop)
            mask_crop_png.save(pathSaveMask+str(iter+path_iter)+".png") # used for save From PNG to PNG
            # img_crop=img_crop[:,:,:3] # used for save From PNG to JPG
            # mask_crop_png=mask_crop_png[:,:,:] # used for save From PNG to JPG
            # bad_mask_crop=np.concatenate((bad_mask_crop,ch4),axis=2)
            # plt.imshow(bad_mask_crop)
            bad_mask_crop=Image.fromarray(bad_mask_crop)
            bad_mask_crop.save(pathSaveBadMask+str(iter+path_iter)+".png") # used for save From PNG to JPG # pathSave
            # print(img_crop.shape)
            print("ok")

    path_iter += num_img_crop

# with Image.open()
print("DEBUG")
