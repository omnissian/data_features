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
seed=8947
random.seed(seed) # 23 - 2,
count_images=100


# think a bout when any figure is drawn at the image and it touches with visible section only by this background- ONLY AFFECTED when you have0
#-------------------------------------------------------------------------------------
#b g r
h=256
w=256
wh=(h+w)/2.0
channels=3
is_border=False
# is_border=False
border=3
# 203, 117, 173 - pink, violet

colours=[]
wtf = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
#---------------bgr opencv---------------------------------
# blue = (255, 0, 0)
# red = (0, 0, 255)
# green = (0, 255, 0)
# violet = (180, 0, 180)
# yellow = (0, 180, 180)
# another = (180, 180, 0)
#---------------rgb matplotlib-----------------------------

red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
violet = (180, 0, 180)
yellow = (0, 180, 180)
another = (180, 180, 0)

#---------------------------------------------------------

colours.append(wtf)
colours.append(blue)
colours.append(red)
colours.append(green)
colours.append(yellow)
colours.append(another)

#---------------------------------
white = (255, 255, 255) # background
black = (0, 0, 0) #borders
#---------------------------------------------------------------------------
min_size=int(wh*0.01)+2
small_size=int(wh*0.1)
medium_size=int(wh*0.4)
big_size=wh

#----------------------------------
# min_fig_triag=1
# max_fig_triag=11
# min_fig_rectang=1
# max_fig_rectang=11
# min_fig_circle=1
# max_fig_circle=11
#----------------------------------

# img=np.zeros((h,w,channels),np.uint8)
# img.fill(255)

images_count=100
# count_triag=random.randint(min_fig_triag,max_fig_triag)

creation=False
augment=False

#-------------rectangs----------------SMALL

if(creation):
    for num in range(images_count):
        min_fig_triag=0
        max_fig_triag=random.randint(0,15)
        min_fig_rectang=0
        max_fig_rectang=random.randint(0,15)
        min_fig_circle=0
        max_fig_circle=random.randint(0,15)
        #------------------------------------
        min_size = int(np.round((wh * 0.01),0)) + 2
        small_size = int(wh * 0.1)+random.randint(0,int(wh * 0.15))
        medium_size = int(wh * 0.4)
        big_size = wh
        #------------------------------------

        img = np.zeros((h, w, channels), np.uint8)
        img.fill(255)
        rectangs=random.randint(min_fig_rectang,max_fig_rectang)
        # rectangs=10

        for i in range(rectangs):
            size=random.randint(min_size,small_size)
            x1=random.randint(-size,h)
            y1=random.randint(-size,w)
            x2=random.randint(x1-small_size,x1+small_size)
            y2=random.randint(y1-small_size,y1+small_size)

            rect1=[(x1,y1),(x2,y2)]
            rotation_angle_max=91
            rotation=random.randint(0,rotation_angle_max)
            #------------------border------------------------
            if(is_border):
                rot_rect1 = (rect1[0], rect1[1], rotation)
                box = cv.boxPoints(rot_rect1)
                box = np.int0(box)
                cv.drawContours(img, [box], 0, black, border)

                # cv.drawContours(img, [box], 0, black, border)
                #-----------------------------------------------------------------------------
                # rect1_border = [(x1 - border, y1 - border), (x2 + border, y2 + border)]
                # rot_rect_border = (rect1_border[0], rect1_border[1], rotation)
                # box = cv.boxPoints(rot_rect_border)
                # box = np.int0(box)
                # cv.drawContours(img, [box], 0, black, -1)
                # # cv.drawContours(img, [box], 0, black, border)
            #------------------border------------------------

            rot_rect1 = (rect1[0], rect1[1], rotation)
            box = cv.boxPoints(rot_rect1)
            box = np.int0(box)
            cv.drawContours(img, [box], 0, yellow, -1)

        triags = random.randint(min_fig_triag, max_fig_triag)
        for i in range(triags):
            kx_rand=random.uniform(0.01,2)
            ky_rand=random.uniform(0.01,2)

            size=random.randint(min_size,small_size)
            x1=random.randint(-size,h+size)
            y1=random.randint(-size,w+size)
            x2=random.randint(x1-small_size,x1+small_size)
            y2=random.randint(y1-small_size,y1+small_size)
            x3=random.randint(int(np.round((x1+x2)/2.0-kx_rand*size,2)),int(np.round((x1+x2)/2.0+ky_rand*size)))
            y3=random.randint(int(np.round((y1+y2)/2.0-kx_rand*size,2)),int(np.round((y1+y2)/2.0+ky_rand*size)))



            # ---------------------------------------------------
            # triangle_dots = np.array([pt1, pt2, pt3])
            triangle_dots = np.array([(x1,y1), (x2,y2),(x3,y3)])
            if(is_border):
                cv.drawContours(img, [triangle_dots], 0, black, border) # border
            cv.drawContours(img, [triangle_dots], 0, green, -1)

            # ---------------------------------------------------

            # rect1=[(x1,y1),(x2,y2)]
            # border=3
            # rect1_border = [(x1 - border, y1 - border), (x2 + border, y2 + border)]
            # rotation_angle_max=91
            # rotation=random.randint(0,rotation_angle_max)
            # rot_rect = (rect1[0], rect1[1], rotation)
            # box = cv.boxPoints(rot_rect)
            # box = np.int0(box)
            # cv.drawContours(img, [box], 0, black, border)
            #
            # rot_rect_border = (rect1[0], rect1[1], rotation)
            # box = cv.boxPoints(rot_rect_border)
            # box = np.int0(box)
            # cv.drawContours(img, [box], 0, yellow, -1)

        circles=random.randint(min_fig_circle,max_fig_circle)
        for i in range (circles):
            size=random.randint(min_fig_circle,max_fig_circle)
            x=random.randint(0-size,255+size)
            y=random.randint(0-size,255+size)
            if(is_border):
                cv.circle(img, (x, y), size, black, border)  # border
                # cv.circle(img,(x,y),size+border,black,-1) # border
            cv.circle(img,(x,y),size,red,-1)

        path_save="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/"
        name_save="img"+str(num)+"plt.png"
        plt.imsave(path_save+name_save,img)

cv.imshow("imaga",img)
#-------------------saving info---------------------------------
#b g r
# C:\Users\user\Desktop\DontTouchPLSpytorch\Polygon\img_saved
path_save="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/"
# path_save="/home/std_11/MazurM/UnetV1/DataSet1/db_small_2/train/in/"
# name_save="img1plt.jpg"
name_save="img11plt.png"
plt.imsave(path_save+name_save,img)
#-------------------saving info---------------------------------
#-------------------------reader and checker image to unique pixels----------------

img=np.array(Image.open(path_save+name_save))
label=np.array(Image.open(path_save+name_save))
#-------------------------------------------------------------------------------

print("break")
