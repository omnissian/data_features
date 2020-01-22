
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

path="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/HistData5newA/"
#dictionary: key:-numpy array - shape 3 of pixel
            #value - count of these pixels


#---------------crop Bigger image to smaller images-------------BELOW-------------
#
# pathBig="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/MMA/"
# pathSave="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/MMAcropped/" # JPG
# pathSavePNG="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/MMAcroppedPNG/"
#---------
# "C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/DBMMA/anotherosmgmaps/another_osmg_jpg/"
# pathBig="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/DBMMA/anotherosmgmaps/another_osm_maps_orig/"
# pathSave="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/DBMMA/anotherosmgmaps/another_osmg_jpg/" # JPG
# pathSavePNG="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/DBMMA/anotherosmgmaps/another_osmg_png/"
#-------------------------------------------------------------
# C:\Users\user\Desktop\DontTouchPLSpytorch\Polygon\img_saved\DBMMA
# C:\Users\user\Desktop\DontTouchPLSpytorch\Polygon\img_saved\DBMMA\anthr\anothcrpjpg
# C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/DBMMA/anthr/anothcrpjpg/
pathBig="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/DBMMA/anthr/another_osm_maps_orig/"
pathSave="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/DBMMA/anthr/anothcrpjpg/" # JPG
pathSavePNG="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/DBMMA/anthr/anothcrppng/"

num_img_crop=50 # PER IMAGE!!!


def randomCrop(img,newH=256,newW=256):
    curH,curW,_=img.shape
    assert curH>=newH
    assert curW>=newW
    x=random.randint(0,curH-newH)
    y=random.randint(0,curW-newW)
    img_crop=img[x:x+newH,y:y+newW]
    return img_crop

    # print(curH, curW)


    pass

fnames=os.listdir(pathBig)
for name in fnames:
    with Image.open(pathBig+name) as img:
        for iter in range(num_img_crop):
            img=np.array(img)
            img_crop=randomCrop(img)
            print ("old img=",img.shape)
            print ("new img=",img_crop.shape)
            img_png=Image.fromarray(img_crop)
            img_png.save(pathSavePNG+str(iter)+".png") # used for save From PNG to PNG
            img_crop=img_crop[:,:,:3] # used for save From PNG to JPG
            img_crop=Image.fromarray(img_crop)
            img_crop.save(pathSave+str(iter)+".jpg") # used for save From PNG to JPG
            # print(img_crop.shape)
            print("ok")

# with Image.open()
print("DEBUG")




#---------------crop Bigger image to smaller images-------------BELOW-------------





#------------------dataset for pix2pix train image to generate same image-------------
#---------resave the images from realB.png to realA.png (realA were .realA.jpg)
pathB="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/Original3/"
path_newB="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/HistData5newA/"
fnames=os.listdir(pathB)
for i in range(len(fnames)):
    with Image.open(pathB+fnames[i]) as image:
        new_name=fnames[i]
        new_name=new_name[:-4]
        image=np.array(image)
        plt.imsave(path_newB+new_name+".jpg",image)
        print(new_name)
        pass


print("debug")


#------------------dataset for pix2pix train image to generate same image-------------




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
    type_ar=img[0,0] # dont delete it
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
# for i in range(len(unique_pixs)):
#     print("uniq pix=",unique_pixs[i].)

print("len(unique_pixs)=",len(unique_pixs)) #99644

for key in unique_pixs:
    # tmp=key
    print(key) # <<<
    # y = np.frombuffer(k, dtype=i.dtype)
    tmp=np.frombuffer(key,dtype=type_ar.dtype)
    print(tmp) # <<<


print("Debug")
# x = np.random.normal(size = 1000)
# plt.hist(x, normed=True, bins=30)
# plt.ylabel('Probability');

print(len(unique_pixs.keys()))
plt.hist(unique_pixs.values(),bins=len(unique_pixs.keys()))
plt.ylabel('Probability')




print("debug2")



# ##--------------------------------------------------------- below old pix checker checker-----------
# original_names=[]
# # real_A_names=[]
# # real_B_names=[]
# # fake_B_names=[]
# real_A=[]
# real_B=[]
# fake_B=[]
#
#
# # x = np.random.normal(size = 1000)
# # plt.hist(x, normed=True, bins=30)
# # plt.ylabel('Probability')
# # plt.show()
# def find_pix(pix,list_pixs):
#     channels,=pix.shape
#     found=False
#     for i in range(len(list_pixs)):
#         found=False
#         for ch in range(channels):
#             if(pix[ch]==list_pixs[i][ch]):
#                 found=True
#             else:
#                 found=False
#                 break
#         if(found):
#             return i
#     return 0
#
# print("debug")
# imgs=[]
# ##-----------Histogramm colours----size ==len(uniq_pixs)
# uniq_pixs=[] # the member of this list is a nupmy array, shape 3 dtype float
# count_uniq_pix=[] # total!!! NOT per Image
# fnames=os.listdir(path)
# with Image.open(path+fnames[0]) as img: # add first pix and cut the complementary complexity
#         img=np.array(img)
#         # null_elem=np.array([256,256,256],dtype=np.uint8)
#         # null_elem=np.array([257,257,257])
#         # null_elem.astype(np.uint8)
#         # null_elem=np.array([256,256,256],dtype=np.uint8)
#         height,width,channels=img.shape
#         null_elem=np.full((channels),np.inf)
#         null_elem.astype(np.uint8)
#         print(type(null_elem.shape))
#         print(null_elem.shape)
#         uniq_pixs.append(null_elem)
#         count_uniq_pix.append(-1)
#         # uniq_pixs.append(img[0,0,:])
#
#
# for i in range(len(fnames)):
#     print(fnames[i])
#     with Image.open(path+fnames[i]) as img:
#         # img = np.array(img)
#         # img=np.array(img,dtype=np.float)/255.0
#         img=np.array(img)
#         imgs.append(img)
#         print(img.shape)
#         height,width,channels=img.shape
#         # uniq_pixs.append(img[0,0,:])
#         # print(type(uniq_pixs[0]))
#         for h in range(height):
#             print(h)
#             for w in range (width):
#                 index=find_pix(img[h,w],uniq_pixs)
#                 if(index):
#                     count_uniq_pix[index]+=1
#                     # print("yes",w)
#                 else:
#                     uniq_pixs.append(img[h,w])
#                     count_uniq_pix.append(1)
#                     # print("no",w)
#
#
# print("debug")












##-------------------------------------------------------------------- below accuarcy checker

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



