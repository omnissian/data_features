
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

albumentations.MultiplicativeNoise

#----------------augmentation funcs---------------------
# def gaussian_noise(img, mean=0, sigma=0.01):
#     img=img.copy()
#     noise=np.random.normal(mean,sigma, img.shape)
#     mask_overflow_upper=img+noise >=1.0
#     mask_overflow_lower=img+noise <0
#     noise[mask_overflow_upper]=1.0
#     noise[mask_overflow_lower]=0
#     img+=(noise)
#     return img
#-----------------
def gaus_blur(img, blur): # blur - double
    image=cv.GaussianBlur(img,(5,5),blur)
    return image
    # cv.imwrite(final_name,img)

def salt_image(image,p,a):
    noisy=image
    num_salt = np.ceil(a * image.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords] = 1
    return image
    # cv2.imwrite(Folder_name + "/Salt-"+str(p)+"*"+str(a) + Extension, image)


def salt_and_paper_img(image,p,a):
    noisy=image
    #salt
    num_salt = np.ceil(a * image.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords] = 1

    #paper
    num_pepper = np.ceil(a * image.size * (1. - p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[coords] = 0
    return image

def interpolation_back (image, type): # interpolates to diff size and then interpolates back
    interpol = {
                "nearest": cv.INTER_NEAREST,
                "linear": cv.INTER_LINEAR,
                "area": cv.INTER_AREA,
                "cubic": cv.INTER_CUBIC,
                "lanczos4": cv.INTER_LANCZOS4
                }
    # print(interpol[0])
    # print(image.shape)
    oldW,oldH,_=image.shape
    newW=random.randint((oldW/4),oldW)
    newH=random.randint((oldH/4),oldH)

    img = cv.resize(image,(newW,newH),interpolation=interpol[type])
    img = cv.resize(img,(oldW,oldH),interpolation=interpol[type])
    # target=np.ones()
    return img


#----------------augmentation funcs---------------------


seed=7423 #was 8947
random.seed(seed) # 23 - 2,
count_images=500


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
# 6  0-5
colours.append(red)
colours.append(green)
colours.append(blue)
colours.append(yellow)
colours.append(wtf)
colours.append(another)

#---------------------------------
white = (255, 255, 255) # background
black = (0, 0, 0) #borders
#---------------------------------------------------------------------------
min_size=int(wh*0.01)+2
small_size=int(wh*0.1)
medium_size=int(wh*0.4)
big_size=wh

#----------------------------------------------------------
# min_fig_triag=1
# max_fig_triag=11
# min_fig_rectang=1
# max_fig_rectang=11
# min_fig_circle=1
# max_fig_circle=11
#----------------------------------------------------------

# img=np.zeros((h,w,channels),np.uint8)
# img.fill(255)

images_count=count_images
# count_triag=random.randint(min_fig_triag,max_fig_triag)
#
creation=False
augment=True
#---------------
# creation=True
# augment=False
#---------------creationpath---------------------

path_save_creation="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/Original3/"

#-----------------augmentation------------------------------
path_read_aug="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/Original3/"
path_save_aug="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/Augmented31/"
#-----------------augmentation------------------------------
# path_read_aug=path_save_aug
iterations=0
# extension=".png"  #png
extension=".jpg"  #jpeg   jpg
extension_symbols=4 # given str="imaga_name.png"  and slice like print(str[:-4]) return "imaga_name"
step_check=10000
type_list=["nearest","linear","area","cubic","lanczos4"]
counter_type=[]
for i in range(len(type_list)):
    counter_type.append(0)
#---------------------------------------------------------------------
if (augment):
    for i in range(iterations+1):
        if (not i%step_check and i>1):
            print("pause break")
        for filename in os.listdir(path_read_aug):
            img = np.array(Image.open(path_read_aug + filename))
            # # img_aug=gaussian_noise(img, 0,0.03)
            # # img=albumentations.MultiplicativeNoise(multiplier=[0.5,1.5], elementwise=True,p=1)
            # print("type(img) ", type(img))
            # print(img)
            final_name=path_save_aug+filename[:-extension_symbols]+extension
            salt_pep_before=20
            salt_pep_after=5
            gauss_inter_chance_before=25
            gauss_inter_chance_after=15
            a_max=0.01
            p_max=0.01
            blur_max=30
            a=random.uniform(0,a_max) # for salt max value
            p=random.uniform(0,p_max) # for salt max value
            blur=random.uniform(0,blur_max) # for gauss
            # blur=random.randint(0,blur_max) # for gauss

            tmp_img=img.copy()
            if(random.randint(0,100)<=salt_pep_before):
                tmp_img=salt_and_paper_img(tmp_img,p,a)
            if(random.randint(0,100)<=gauss_inter_chance_before):
                # plt.imsave(final_name,gaus_blur(tmp_img,blur))
                tmp_img=gaus_blur(tmp_img,blur)
            else:
                type_id=random.randint(0,len(type_list)-1) # random.randint is [,] included range
                counter_type[type_id]+=1
                # plt.imsave(final_name,interpolation_back(tmp_img,type_list[type_id]))
                tmp_img=interpolation_back(tmp_img,type_list[type_id])
            # plt.imsave(final_name,gaus_blur(img,blur)) # gauss
            # plt.imsave(path_save_aug+filename[:-extension_symbols]+extension,img)
            #-----------after-------------
            if(random.randint(0,100)<=gauss_inter_chance_after):
                # plt.imsave(final_name,gaus_blur(tmp_img,blur))
                tmp_img=gaus_blur(tmp_img,blur)
            if (random.randint(0, 100) <= salt_pep_before):
                tmp_img = salt_and_paper_img(tmp_img, p, a)
            plt.imsave(final_name, tmp_img)

#---------------------------------------------------------------------

#------------------------augment by RESAVING----------------------
# if (augment):
#     for i in range(iterations+1):
#         if (not i%step_check and i>1):
#             print("pause break")
#         for filename in os.listdir(path_read):
#             img = np.array(Image.open(path_read + filename))
#             plt.imsave(path_save+filename[:-extension_symbols]+extension,img)
#

#------------------------augment by RESAVING----------------------

print("number of augmentations used:")
total=0
for i in range(len(type_list)):
    print(type_list[i]," used ",counter_type[i], "times")
    total+=counter_type[i]
print("total augments=", total)
print("pause")

#-----------------augmentation------------------------------


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
            cv.drawContours(img, [box], 0, colours[random.randint(0,len(colours)-1)], -1)

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
            cv.drawContours(img, [triangle_dots], 0, colours[random.randint(0,len(colours)-1)], -1)

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
            cv.circle(img,(x,y),size,colours[random.randint(0,len(colours)-1)],-1)

        # path_save_creation="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/"
        name_save="img"+str(num)+"plt.png"
        plt.imsave(path_save_creation+name_save,img)

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




#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-------------------------------------------------OVER------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------





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

#b g r   open cv
#r g b   matplotlib

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
