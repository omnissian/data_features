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
#-------HYPER PARAMS-------------
path="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/Original3/"




# uniq pixels
# C:\Users\user\Desktop\DontTouchPLSpytorch\Polygon\img_saved\CleanMapForTestOnly1
#path="C:/Users/user/Desktop/DontTouchPLSpytorch/data/DB_2020-01-30/Temp/12210/"
#labels_big=os.listdir()

# mask_original_path_label5="C:/Users/user/Desktop/DontTouchPLSpytorch/data/DB_2020-01-30/Temp/12210/"
# name="LABEL_5.png"


#----------------------image crop slicer-----------------
#mask file_name name = name of folder of input images
# names of input images before crop
# google_sat.png
# google_maps.png
# esri_sat.png
# esri_map.png
# another_osm__maps.png
# bing_map.png
# cartodb.png
# mapbox_hyb_with_api.png

random.seed(37) # 25
def image_crop(input_images,):  # input_images(numpy tensor
    pass


def images_crop_prep(path_inputs,path_save_inputs,path_masks,path_save_masks,height_resolution=256,width_resolution=256):
    names_folders=os.listdir(path_masks)
    for name in (names_folders):
        with Image.open(path_inputs+name[:-4]+"/"+"google_sat.png")  as big_google_sat, \
                Image.open(path_inputs+name[:-4]+"/"+"google_maps.png") as google_maps, \
                Image.open(path_inputs+name[:-4]+"/"+"esri_sat.png") as esri_sat, \
                Image.open(path_inputs+name[:-4]+"/"+"esri_map.png") as esri_map, \
                Image.open(path_inputs+name[:-4]+"/"+"another_osm__maps.png") as another_osm__maps, \
                Image.open(path_inputs+name[:-4]+"/"+"cartodb.png") as cartodb, \
                Image.open(path_inputs+name[:-4]+"/"+"mapbox_hyb_with_api.png") as mapbox_hyb_with_api, \
                Image.open(path_masks+name) as mask:
            list_original_inputs=[]
            list_original_inputs.append(np.asarray(big_google_sat,dtype=np.uint8)) # 0
            list_original_inputs.append(np.asarray(google_maps,dtype=np.uint8)) # 1
            list_original_inputs.append(np.asarray(esri_sat,dtype=np.uint8)) # 2
            list_original_inputs.append(np.asarray(esri_map,dtype=np.uint8)) # 3
            list_original_inputs.append(np.asarray(another_osm__maps,dtype=np.uint8)) # 4
            list_original_inputs.append(np.asarray(cartodb,dtype=np.uint8)) # 5
            list_original_inputs.append(np.asarray(mapbox_hyb_with_api,dtype=np.uint8)) # 6
            mask=np.asarray(mask)
            for i in range(len(list_original_inputs)):
                list_original_inputs[i]=list_original_inputs[i][:,:,0:3]
                print(list_original_inputs[i].shape)
                # plt.imshow(list_original_inputs[i][:,:,0:3])
                # plt.show()
            width_max, height_max, _ = mask.shape
            num_crops=100   # amount of crops per one original (big) image
            for n in range(num_crops):
                width_new=random.randint(0,width_max-width_resolution)
                height_new=random.randint(0,height_max-height_resolution)
                list_inputs_new=[]
                # mask_new=np.zeros((width_resolution,height_resolution),dtype=np.unit8)
                #-------------------
                big_google_sat_new=list_original_inputs[0][width_new:width_new+width_resolution,height_new:height_new+height_resolution,:]
                plt.imshow(big_google_sat_new)
                google_maps_new = list_original_inputs[1][width_new:width_new + width_resolution,
                                     height_new:height_new + height_resolution, :]
                google_maps_new=Image.fromarray(google_maps_new)
                google_maps_new.save()
                esri_sat_new = list_original_inputs[2][width_new:width_new + width_resolution,
                                     height_new:height_new + height_resolution, :]
                esri_map_new = list_original_inputs[3][width_new:width_new + width_resolution,
                                     height_new:height_new + height_resolution, :]
                another_osm__maps_new = list_original_inputs[4][width_new:width_new + width_resolution,
                                     height_new:height_new + height_resolution, :]
                cartodb_new = list_original_inputs[5][width_new:width_new + width_resolution,
                                     height_new:height_new + height_resolution, :]
                mapbox_hyb_with_api_new = list_original_inputs[6][width_new:width_new + width_resolution,
                                     height_new:height_new + height_resolution, :]

                # google_maps=google_maps[width_new:width_max,height_new:height_max,:]
                # esri_sat=esri_sat[width_new:width_max,height_new:height_max,:]
                # esri_map=esri_map[width_new:width_max,height_new:height_max,:]
                # another_osm__maps=another_osm__maps[width_new:width_max,height_new:height_max,:]
                # cartodb=cartodb[width_new:width_max,height_new:height_max,:]
                # mapbox_hyb_with_api=mapbox_hyb_with_api[width_new:width_max,height_new:height_max,:]
                # mask_new=mask[width_new:width_max,height_new:height_max,:]

                # for i in range (len(list_original_inputs)):
                #     big_google_sat_new=list_original_inputs[i][width_new:width_new+width_resolution,height_new:height_new+height_resolution,:]
                #     plt.imshow(big_google_sat_new)

                    # big_google_sat_new=big_google_sat[width_new:width_max,height_new:height_max,:]
                    # google_maps=google_maps[width_new:width_max,height_new:height_max,:]
                    # esri_sat=esri_sat[width_new:width_max,height_new:height_max,:]
                    # esri_map=esri_map[width_new:width_max,height_new:height_max,:]
                    # another_osm__maps=another_osm__maps[width_new:width_max,height_new:height_max,:]
                    # cartodb=cartodb[width_new:width_max,height_new:height_max,:]
                    # mapbox_hyb_with_api=mapbox_hyb_with_api[width_new:width_max,height_new:height_max,:]
                    # mask_new=mask[width_new:width_max,height_new:height_max,:]

    # (1664, 1664, 3)
    # (1664, 1664, 4)
    # (1664, 1664, 3)
    # (1664, 1664, 3)
    # (1664, 1664, 4)
    # (1664, 1664, 4)
    # (1664, 1664, 3)

    pass

path_to_input_imgs="C:/Users/user/Desktop/DontTouchPLSpytorch/data/DB_2020-01-30/Temp/"
path_save_cropped_inputs="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/HighResolutionNetData/Input/"
path_to_mask="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/CleanMapForTestOnly1/"
path_save_cropped_masks="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/HighResolutionNetData/Masks/"

images_crop_prep(path_to_input_imgs,path_save_cropped_inputs,path_to_mask,path_save_cropped_masks)
names_folders=os.listdir(path_to_mask)
for i in range(len(names_folders)):
    print(names_folders[i][:-4])


print("debug point")



#----------------------image crop slicer-----------------



#----------------check unique pixels-------------

path2="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/CleanMapForTestOnly1/"
pixs_dict={}
uniq_pix(pixs_dict,path2)
print(len(pixs_dict))
print("debug")
#----------------check unique pixels-------------


pixels_classes = np.array(
    [[255, 255, 255, 255],
     [250, 10, 222, 255],
     [0, 255, 0, 255],
     [255, 0, 0, 255],
     [50, 0, 0, 255],
     [30, 0, 0, 255],
     [0, 0, 255, 255]])
pixels_classes=pixels_classes[:,0:3] # <<<<< now fourth channel thrown away
classes,_=pixels_classes.shape
# path_mask_original_img="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/CreatingMaskClassesNotImg/"
path_mask_original_img="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/mask_classes_backup/trainB/" # <<< correct
name="47.png"

with Image.open(path_mask_original_img+name) as img_mask:
    uniq_pixels_dict={}
    uniq_pix(uniq_pixels_dict,path_mask_original_img,name)
    print(len(uniq_pixels_dict))
    img_mask=np.asarray(img_mask)
    img_mask=img_mask[:,:,0:3]
    width, height, channels=img_mask.shape
    mask = np.zeros((width,height,classes),dtype=np.uint8) #  returned mask
    list_of_pixels=[]
    for k in uniq_pixels_dict.keys():
        list_of_pixels.append(np.frombuffer(k,dtype=np.uint8))
        wtf=np.frombuffer(k,dtype=np.uint8)
        print(k, np.frombuffer(k,dtype=np.uint8))
        print(uniq_pixels_dict[k])

    for row in range(classes):
        print(row," : ",pixels_classes[row])
    for w in range (width):
        for h in range(height):
            wtf1=img_mask[w,h]
            class_index=np.where((pixels_classes==img_mask[w,h]).all(axis=1))[0]
            # print("class_index=",class_index)
            # print("img_mask[w,h]= ",img_mask[w,h])
            mask[w,h,class_index]=1

            # for pixel in list_of_pixels:
            #     wtf1=list_of_pixels[h,w,:]
            #     class_index=np.where((pixels_classes==pixel).all(axis=1))[0]
            #     mask[w,h,class_index]=1
            #     # print(class_index)


img_from_mask=np.zeros((width,height,3),dtype=np.uint8)
for w in range(width):
    for h in range(height):
        for c in range(classes):
            # print("mask[w,h]=",mask[w,h])
            # print(mask[w,h,c])
            if(mask[w,h,c]==1):
                wtf1=img_from_mask[w,h]
                wtf2=pixels_classes[c]
                img_from_mask[w,h]=pixels_classes[c]
                break

plt.imshow(img_from_mask)
print("check created mask")
    # for w in range (width):
    #     for h in range(height):
print(pixels_classes)
print("end")



