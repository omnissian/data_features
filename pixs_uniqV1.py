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

path="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/HistData/"
original_names=[]
# real_A_names=[]
# real_B_names=[]
# fake_B_names=[]
real_A=[]
real_B=[]
fake_B=[]


# x = np.random.normal(size = 1000)
# plt.hist(x, normed=True, bins=30)
# plt.ylabel('Probability')
# plt.show()
def find_pix(pix,list_pixs):
    channels,=pix.shape
    found=False
    for i in range(len(list_pixs)):
        found=False
        for ch in range(channels):
            if(pix[ch]==list_pixs[i][ch]):
                found=True
            else:
                found=False
                break
        if(found):
            return i
    return 0

print("debug")
imgs=[]
##-----------Histogramm colours----size ==len(uniq_pixs)
uniq_pixs=[] # the member of this list is a nupmy array, shape 3 dtype float
count_uniq_pix=[] # total!!! NOT per Image
fnames=os.listdir(path)
with Image.open(path+fnames[0]) as img: # add first pix and cut the complementary complexity
        img=np.array(img)
        # null_elem=np.array([256,256,256],dtype=np.uint8)
        # null_elem=np.array([257,257,257])
        # null_elem.astype(np.uint8)
        # null_elem=np.array([256,256,256],dtype=np.uint8)
        null_elem=np.full((3),np.inf)
        null_elem.astype(np.uint8)
        print(type(null_elem.shape))
        print(null_elem.shape)
        uniq_pixs.append(null_elem)
        count_uniq_pix.append(-1)
        # uniq_pixs.append(img[0,0,:])


for i in range(len(fnames)):
    print(fnames[i])
    with Image.open(path+fnames[i]) as img:
        # img = np.array(img)
        # img=np.array(img,dtype=np.float)/255.0
        img=np.array(img)
        imgs.append(img)
        print(img.shape)
        height,width,channels=img.shape
        # uniq_pixs.append(img[0,0,:])
        # print(type(uniq_pixs[0]))
        for h in range(height):
            for w in range (width):
                index=find_pix(img[h,w],uniq_pixs)
                if(index):
                    count_uniq_pix[index]+=1
                else:
                    uniq_pixs.append(img[h,w])
                    count_uniq_pix.append(1)


print("debug")
