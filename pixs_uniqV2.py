import os
from PIL import Image
import numpy as np
import cv2 as cv

#C:\Users\user\Desktop\DontTouchPLSpytorch\Polygon\img_saved\CheckAccuracy

path="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/HistData3/"
#dictionary: key:-numpy array - shape 3 of pixel
            #value - count of these pixels


fnames=os.listdir(path)
unique_pixs={}
channels=0
lfr=0
with Image.open(path+fnames[0]) as img:
    img=np.asarray(img)
    channels,=img[0,0].shape
    pix=img[0,0]
    pix=pix.tobytes()
    pix2=img[100,174]
    pix2=pix2.tobytes()
    lfr=img[70,70]
    lfr=lfr.tobytes()
    # pix_data=img[0,0].data
    unique_pixs[pix]=1
    unique_pixs[pix2]=1
    # lfr=pix
    print("debug1")
    # unique_pixs.a

if lfr in unique_pixs:
    print("davai")

print("debug2")
