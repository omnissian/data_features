


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






#b g r
# C:\Users\user\Desktop\DontTouchPLSpytorch\Polygon\img_saved
path_save="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/"
# path_save="/home/std_11/MazurM/UnetV1/DataSet1/db_small_2/train/in/"
# name_save="img1plt.jpg"
name_save="img2plt.png"
#-------------------------reader and checker image to unique pixels----------------

img=np.array(Image.open(path_save+name_save))
label=np.array(Image.open(path_save+name_save))
# img = torch.from_numpy(np.asarray(img))
uniq_pixs=np.unique(label,axis=2)
print(label.shape)
print("breakpoint")
#-------------------------------------------------------------------------------------
# j p g
#b g r
h=256
w=256
channels=3
blue = (255, 0, 0)
red = (0, 0, 255)
green = (0, 255, 0)
violet = (180, 0, 180)
yellow = (0, 180, 180)
white = (255, 255, 255)
black = (0, 0, 0)
#-------------------------------------------------------------------------------------

#b g r

h=256
w=256
channels=3
wtf = (155, 155, 155)
blue = (255, 0, 0)
red = (0, 0, 255)
green = (0, 255, 0)
violet = (180, 0, 180)
yellow = (0, 180, 180)
white = (255, 255, 255)
black = (0, 0, 0)
#---------------------------------------------------------------------------
img=np.zeros((h,w,channels),np.uint8)
img.fill(255)
rectangle=[[25,25],[100,100]]
border=2
rect1=[(25,25),(100,100)]
rect1_border=[(25-border,25-border),(100+border,100+border)]
print(rectangle[0][:])

rot_rect=(rect1[0],rect1[1],30)
box=cv.boxPoints(rot_rect)
box=np.int0(box)
cv.drawContours(img,[box],0,black,border)

rot_rect=(rect1[0],rect1[1],30)
box=cv.boxPoints(rot_rect)
box=np.int0(box)
cv.drawContours(img,[box],0,yellow,-1)
print("chechk")

#---------------triangle-----------------
pt1=(125,125)
pt2=(75,175)
pt3=(175,175)
pt4=(175,175)

triangle_dots=np.array([pt1,pt2,pt3])
cv.drawContours(img,[triangle_dots],0,black,4)
cv.drawContours(img,[triangle_dots],0,wtf,-1)

#--------------img save--------------
# image_path_save
print("type(img) ",type(img))
print("type(img) ",img.shape)


cv.imshow("Marat",img)
print("pause")

plt.imsave(path_save+name_save,img)

#--------------------------------------------------------------------


# triang1=[pt1,pt2,pt3,pt4]
# rot_rect=(triang1[0],triang1[1],triang1[2],triang1[3],30)
# box=cv.boxPoints(rot_rect)
# box=np.int0(box)
# cv.drawContours(img,[box],0,yellow,border)







# rot_rect=(triangle_dots[0],triangle_dots[1],triangle_dots[2],18)
# box=cv.boxPoints(rot_rect)
# box=np.int0(box)
# cv.drawContours(img,[box],0,red,-1)


#---------------triangle-----------------

