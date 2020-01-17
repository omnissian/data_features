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

path="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/HistData4/"
#dictionary: key:-numpy array - shape 3 of pixel
            #value - count of these pixels


fnames=os.listdir(path)
unique_pixs={}
type_ar=0
lfr=0
height=0
width=0
channels=0
with Image.open(path+fnames[0]) as img:
    img=np.asarray(img)
    height,width,channels=img.shape
    channels,=img[0,0].shape
    # type_ar=img[0,0] # dont delete it
    pix=img[0,0]
    pix=pix.tobytes()
    unique_pixs[pix]=1
    # # lfr=pix
    # print("debug1")
    # # unique_pixs.a
    # pix2=img[100,174]
    # pix2=pix2.tobytes()
    # lfr=img[70,70]
    # lfr=lfr.tobytes()
    # # pix_data=img[0,0].data
    # unique_pixs[pix2]=1
# if lfr in unique_pixs:
#     print("davai")

for img_index in range(len(fnames)):
    with Image.open(path+fnames[img_index]) as img:
        img=np.asarray(img)
        for h in range(height):
            for w in range (width):
                pix=(img[h,w]).tobytes()
                if pix in unique_pixs:
                    unique_pixs[pix]+=1
                else:
                    unique_pixs[pix]=1


print("len(unique_pixs)=",len(unique_pixs))

for key in unique_pixs:
    tmp=key
    # y = np.frombuffer(k, dtype=i.dtype)
    tmp=np.frombuffer(key,dtype=type_ar.dtype)
    print(tmp)


print("debug2")
