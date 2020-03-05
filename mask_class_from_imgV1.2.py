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
path_real_B_train="/storage/3050/MazurM/db/mask_classes/trainB/"
names = os.listdir(path_real_B_train)
item=2
#-----check the amount of unique pixels--------------

dict_pixs={}
uniq_pix(dict_pixs,path_real_B_train,names[2])
print(len(dict_pixs))


#-----check the amount of unique pixels--------------

# path_real_B_train="/storage/3050/MazurM/db/CleanMapLabel2/trainB/"
print(names[item])
realB_img=np.array(Image.open(path_real_B_train+names[item]))
width, height,_=realB_img.shape
plt.imshow(realB_img)
plt.show()
realB_img=realB_img[:,:,0:3]
pixels_classes = np.array(
    [[255, 255, 255, 255],
     [250, 10, 222, 255],
     [0, 255, 0, 255],
     [255, 0, 0, 255],
     [50, 0, 0, 255],
     [30, 0, 0, 255],
     [0, 0, 255, 255]])
pixels_classes=pixels_classes[:,0:3]
num_classes,_=pixels_classes.shape
maskB=np.zeros((width,height,num_classes))
for w in range(width):
    for h in range(height):
        wtf1 = realB_img[w, h]
        class_index = np.where((pixels_classes == realB_img[w, h]).all(axis=1))[0]
        # print("class_index=",class_index)
        # print("img_mask[w,h]= ",img_mask[w,h])
        maskB[w, h, class_index] = 1

img_from_mask=np.zeros((width,height,3),dtype=np.uint8) # dtype=np.uint8 was necessary for pink color,BUT WHY?
for w in range(width):
    for h in range(height):
        for c in range(num_classes):
            if(maskB[w,h,c]==1):
                img_from_mask[w,h]=pixels_classes[c]
                # break
print("last connection")
pdb.set_trace()
plt.imshow(img_from_mask)
plt.show()

#----creating mask using image realB -------
print("all done fine")










