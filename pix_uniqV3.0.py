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



def uniq_pix(dictionary,path,name=0):
    if(name==0):
        names=os.listdir(path)
        for name in names:
            path_current=path+name
            with Image.open(path_current) as mask_original:
                height,width=mask_original.size
                mask_original=np.asarray(mask_original)
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
            for w in range(width):
                for h in range(height):
                    pix_current=mask_original[h,w].tobytes()
                    if pix_current in dictionary:
                        dictionary[pix_current]+=1
                    else:
                        dictionary[pix_current]=1




#-------HYPER PARAMS-------------



# C:\Users\user\Desktop\DontTouchPLSpytorch\Polygon\img_saved\Original3
# path="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/HistData5newA/"
path="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/Original3/"

# uniq pixels
# C:\Users\user\Desktop\DontTouchPLSpytorch\Polygon\img_saved\CleanMapForTestOnly1
#path="C:/Users/user/Desktop/DontTouchPLSpytorch/data/DB_2020-01-30/Temp/12210/"
#labels_big=os.listdir()

mask_original_path_label5="C:/Users/user/Desktop/DontTouchPLSpytorch/data/DB_2020-01-30/Temp/12210/"
name="LABEL_5.png"

uniq_pixels_label5={}
uniq_pix(uniq_pixels_label5,mask_original_path_label5,name)
print("uniq_pixels_label5 = ",len(uniq_pixels_label5))

mask_original_Bulat="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/CleanMapForTestOnly1/"
uniq_pixels_Bulat={}
uniq_pix(uniq_pixels_Bulat,mask_original_Bulat)
print("uniq_pixels_Bulat = ",len(uniq_pixels_Bulat))

for item in uniq_pixels_Bulat:
    print(np.frombuffer(item,dtype=np.uint8))

path_learn_in=""
path_learn_targets="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/Original3/"
path_vaild_in=""
path_vaild_targets=""

# names_valid=os.listdir(path_vaild_in)
names_valid=os.listdir(path_learn_targets)




# with Image.open(mask_original_path) as mask_orignal:
#     # mask_np=np.asarray(mask_orignal)
#     # width, height, channels =mask_np.shape
#     print(mask_orignal.size)
#     width, height=mask_orignal.size
#     print(width,height)
#     for w in range(width):
#         for h in range (height):
#             pix_current=mask_orignal[w,h]
#             # print(pix_current.shape) # (3,)
#             # print(pix_current)
#             pix_current_bytes=Image.frombuffer(pix_current)
#             if pix_current_bytes in uniq_pixels:
#                 uniq_pixels[pix_current_bytes]+=1
#             else:
#                 uniq_pixels[pix_current_bytes]=1
#     print("da")


#------------------------------------------------below workable
# with Image.open(mask_original_path) as mask_orignal:
#     mask_np=np.asarray(mask_orignal)
#     width, height, channels =mask_np.shape
#     for w in range(width):
#         for h in range (height):
#             pix_current=mask_np[w,h].tobytes()
#             # print(pix_current.shape) # (3,)
#             # print(pix_current)
#             # pix_current_bytes=Image.frombuffer(pix_current)
#             if pix_current in uniq_pixels:
#                 uniq_pixels[pix_current]+=1
#             else:
#                 uniq_pixels[pix_current]=1
#     print("da")
#
#
# print(len(uniq_pixels))
print("end")
