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

path="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/CheckAccuracy/"
original_names=[]
# real_A_names=[]
# real_B_names=[]
# fake_B_names=[]
real_A=[]
real_B=[]
fake_B=[]

fnames=os.listdir(path)
for i in range(len(fnames)):
    name=fnames[i]
    name=name[7:-4]
    if(not (name in original_names)):
        original_names.append(name)
for i in range (len(original_names)):
    with Image.open(path+'real_A_'+original_names[i]+".jpg") as real_a:
        real_A.append(np.array(real_a))
    # with Image.open(path+'real_B_'+original_names[i]+".png") as real_b:
    #     real_B.append(np.array(real_b))
    with Image.open(path+'real_A_'+original_names[i]+".jpg") as real_b:
        real_B.append(np.array(real_b))
    real_a=real_A[0]
    # real_b=real_B[0][:,:,0:3]
    real_b=real_B[0]
    cv.imshow("REAL A",real_A[0])
    cv.imshow("REAL B",real_B[0])


eq_per_image=0
for h in range(256):
    for w in range(256):
        equal=False
        for c in range (3):
            if(real_a[h,w,c]==real_b[h,w,c]):
                equal=True
            else:
                equal = False
                break
        if(equal):
            eq_per_image+=1

print(eq_per_image/65536.0) #65536

print("pause")
input()







print("wait")
real_A=Image.open()
