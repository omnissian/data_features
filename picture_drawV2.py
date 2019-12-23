




import numpy
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


#-------------------------------------------------------------
#b g r
# C:\Users\user\Desktop\DontTouchPLSpytorch\Polygon\img_saved
path_save="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/"
# path_save="/home/std_11/MazurM/UnetV1/DataSet1/db_small_2/train/in/"
name_save="1.jpg"

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
#----------------------------------------------------------------






#---------------------------------------------------------------------------
img=np.zeros((h,w,channels),np.uint8)
img.fill(255)
rectangle=[[25,25],[100,100]]
border=2
rect1=[(25,25),(100,100)]
rect1_border=[(25-border,25-border),(100+border,100+border)]
print(rectangle[0][:])
# cv.drawContours()

# cv.line()

# recter1=cv.rectangle(img,rect1_border[0],rect1_border[1],black,-1)
# recter2=cv.rectangle(img,rect1[0],rect1[1],green,-1)

# cv.rectangle(img,rect[0],rect[1],white,-1)




#-----------------------------------
#
# rows,cols,ht=recter1.shape
# matrix=cv.getRotationMatrix2D( (rows/3, cols/3),60, 1)
# new_img=cv.warpAffine(recter1, matrix,(cols,rows))
#-----------------------------------

# rows,cols,ht=img.shape
# matrix=cv.getRotationMatrix2D( (rows/3, cols/3),60, 1)
# new_img=cv.warpAffine(img, matrix,(cols,rows))

#-----------------------------

# rot_rectb=(rect1_border[0],rect1_border[1],30)
# boxb=cv.boxPoints(rot_rectb)
# boxb=np.int0(boxb)
# cv.drawContours(img,[boxb],0,black,-1)



rot_rect=(rect1[0],rect1[1],30)
box=cv.boxPoints(rot_rect)
box=np.int0(box)
cv.drawContours(img,[box],0,black,border)

rot_rect=(rect1[0],rect1[1],30)
box=cv.boxPoints(rot_rect)
box=np.int0(box)
cv.drawContours(img,[box],0,yellow,-1)


#-----------------------------------
# rect=cv.minAreaRect(rect1)
# box=cv.boxPoints(rect)
# box=np.int0(box)
# cv.drawContours(img,[box],0,yellow,2)
#-----------------------------------

cv.imshow("Marat",img)
# cv.imshow("Marat",new_img)



print("pause")


























#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
#
# # img=Image.new("RGB",(h,w),(0,0,0))
# # draw=ImageDraw.Draw(img)
# # draw.ellipse((10,10,100,100),fill="red",outline="green")
# # del draw
# # img.save(path_save+name_save,"JPEG")
#
#
#
# print("pause")
#
#
#
#
# path_learn_in="/home/std_11/MazurM/UnetV1/DataSet1/db_small_2/train/in/"
# path_learn_targets="/home/std_11/MazurM/UnetV1/DataSet1/db_small_2/train/mask/"
# path_vaild_in="/home/std_11/MazurM/UnetV1/DataSet1/db_small_2/valid/in/"
# path_vaild_targets="/home/std_11/MazurM/UnetV1/DataSet1/db_small_2/valid/mask/"
#
# img=np.zeros((h,w,channels), np.uint8)
#
# img=cv.circle(img,(80,60),30,(0,0,255),3)
#
#
# cv.imshow('Marat1',img)
#
#
# # cv.waitKey(0)
#
#
#
# print("hello")
# # pdb.set_trace()
# print("hello")
#
#
# #-------------------------------------------------------------------------
# #-------------------------------------------------------------------------
# # import shutil
# # import numpy as np
# # import cv2
# #
# #
# # height=10 # y
# # width=10 # x
# # label=np.zeros((height,width),dtype=int)
# # for y in range(height):
# #     for x in range(width):
# #         if(8>y>2 and 8>x>2):
# #             label[y,x]=1
# # label[0,0]=1
# # label[9,0]=1
# # label[9,1]=1
# # output=np.copy(label)
# # output[5,5]=0
# # output[5,6]=0
# # output[0,9]=1
# # output[0,0]=0
# # output[9,6]=1
# # print("output")
# # print(output)
# # print("label")
# # print(label)
# #
# #
# #
# # #-------------------------------------------------
# # # height=10 # y
# # # width=10 # x
# # # label=np.zeros((height,width),dtype=int)
# # # for y in range(height):
# # #     for x in range(width):
# # #         if(8>y>2 and 8>x>2):
# # #             label[y,x]=1
# # # label[0,0]=1
# # # label[9,0]=1
# # # label[9,1]=1
# # # output=np.copy(label)
# # # output[0,0]=0
# # # print("output")
# # # print(output)
# # # print("label")
# # # print(label)
# # #-------------------------------------------------
# #
# #
# # diagonal=np.round((((height)**2) + ((width)**2))** 0.5)
# # # distancem=np.zeros((height,width),dtype=int) # we cant fill it with zeros because OF THINK YOUR STEP FAGGOT
# # distancem=np.full((height,width), diagonal)
# #
# # # bad situation still = n^4, average situation ~ n^3
# # right_counter=0
# # not_cmp_pix=0
# # non_pix_counter=0
# #
# # for y in range(height):
# #     for x in range(width):
# #         if (not label[y, x] == output[y, x]):  # start lfr same class
# #             not_cmp_pix +=1
# #             print("---------looking for-------------")
# #             print("label[",y,",",x,"]=", label[y, x])
# #             print("output[",y,",",x,"]=", output[y, x])
# #             found=False
# #             biggest = (height > width) and height or width
# #             for radius in range(1,biggest):
# #                 if(found):
# #                     break
# #                 ry = rx = radius
# #                 left = ((x - rx) >= 0) and (x - rx) or 0
# #                 right = ((x + rx) < width) and (x + rx) or (width-1)
# #                 # top -  "top" only is top for human and "actually is zero" for omnissiah
# #                 top = ((y - ry) >= 0) and y - ry or 0
# #                 bottom = ((y + ry) < height) and y + ry or (height-1)
# #
# #                 #-------------print----------------
# #                 # print("\n-------------------------------------------")
# #                 # print("left=",left)
# #                 # print("right=",right)
# #                 # print("top=",top)
# #                 # print("bottom=",bottom)
# #                 # print("ry=",ry)
# #                 # print("rx=",rx)
# #                 # print("-------------------------------------------\n")
# #                 #-------------print----------------
# #
# #                 print("-----radius=", radius)
# #                 #                 if()
# #                 for yt in range(top, (bottom+1)):  # a walk through height and take a look at left and right columns
# #                     non_pix_counter+=2
# #                     if (label[yt, left] == output[y, x]):
# #                         dist_tmp = (((yt - y) ** 2) + ((left - x) ** 2)) ** 0.5
# #                         if (dist_tmp <= distancem[y, x]):
# #                             distancem[y, x] = dist_tmp
# #
# #
# #                             # print("(yt-y)**2= (", yt,"-", y,")**2=",(yt-y),"**2=", (yt-y)**2)
# #                             # print("(left-x)**2= (", left,"-", x,")**2=",(left-x),"**2=", (left-x)**2)
# #                             # print("(arg1+arg2)**0.5= (", (yt-y)**2,"+", (left-x)**2,")**0.5=",((yt-y)**2+(left-x)**2)**0.5)
# #                             # print("founded dist=",dist_tmp)
# #                             # print("label[", yt, ",", left, "]=", label[y, x])
# #                             # print("output[", y, ",", x, "]=", output[y, x])
# #                             found=True
# #                             # break
# #                     if (label[yt, right] == output[y, x]):
# #                         dist_tmp = (((yt - y) ** 2) + ((right - x) ** 2)) ** 0.5
# #                         if (dist_tmp <= distancem[y, x]):
# #                             distancem[y, x] = dist_tmp
# #                             # print("(yt-y)**2= (", yt, "-", y, ")**2=", (yt - y), "**2=", (yt - y) ** 2)
# #                             # print("(right-x)**2= (", right, "-", x, ")**2=", (right - x), "**2=", (right - x) ** 2)
# #                             # print("(arg1+arg2)**0.5= (", (yt - y) ** 2, "+", (right - x) ** 2, ")**0.5=", ((yt - y) ** 2 + (right - x) ** 2) ** 0.5)
# #                             # print("founded dist=",dist_tmp)
# #                             # print("label[", yt, ",", right, "]=", label[y, x])
# #                             # print("output[", y, ",", x, "]=", output[y, x])
# #                             found=True
# #                             # break
# #                 for xt in range(left, (right+1)):
# #                     non_pix_counter+=2
# #                     if (label[top, xt] == output[y, x]):
# #                         dist_tmp = (((top - y) ** 2) + ((xt - x) ** 2)) ** 0.5
# #                         if (dist_tmp <= distancem[y, x]):
# #                             distancem[y, x] = dist_tmp
# #                             # print("(top-y)**2= (", top, "-", y, ")**2=", (top - y), "**2=", (top - y) ** 2)
# #                             # print("(xt-x)**2= (", xt, "-", x, ")**2=", (xt - x), "**2=", (xt - x) ** 2)
# #                             # print("(arg1+arg2)**0.5= (", (top - y) ** 2, "+", (xt - x) ** 2, ")**0.5=", ((top - y) ** 2 + (xt - x) ** 2) ** 0.5)
# #                             # print("founded dist=",dist_tmp)
# #                             # print("label[", top, ",", xt, "]=", label[y, x])
# #                             # print("output[", y, ",", x, "]=", output[y, x])
# #                             found=True
# #                             # break
# #                     if (label[bottom, xt] == output[y, x]):
# #                         dist_tmp = (((bottom - y) ** 2) + ((xt - x) ** 2)) ** 0.5
# #                         if (dist_tmp <= distancem[y, x]):
# #                             distancem[y, x] = dist_tmp
# #                             # print("founded dist=",dist_tmp)
# #                             # print("(bottom-y)**2= (", bottom, "-", y, ")**2=", (bottom - y), "**2=", (bottom - y) ** 2)
# #                             # print("(xt-x)**2= (", xt, "-", x, ")**2=", (xt - x), "**2=", (xt - x) ** 2)
# #                             # print("(arg1+arg2)**0.5= (", (bottom - y) ** 2, "+", (xt - x) ** 2, ")**0.5=", ((bottom - y) ** 2 + (xt - x) ** 2) ** 0.5)
# #                             # print("label[", bottom, ",", xt, "]=", label[y, x])
# #                             # print("output[", y, ",", x, "]=", output[y, x])
# #                             found=True
# #                             # break
# #         else:
# #             right_counter+=1
# #             distancem[y,x]=0
# # distancem=np.round(distancem,2)
# # print(distancem)
# # print("height*width=",height*width)
# # print("right_counter=",right_counter)
# # print("not compared pixels=",not_cmp_pix)
# # print("pixel finder for non compared pixels =",non_pix_counter)
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
