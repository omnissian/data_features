# import numpy
import torch

import datetime
import pdb
import sys
import torch
import torch.nn as nn
import torchvision.utils as tvis
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import matplotlib as mplot
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
# from PIL import Image

# try:
#     from PIL import __version__
# except ImportError:
#     from PIL import PILLOW_VERSION as __version__

import torch.utils.data as data
import pdb
import datetime
from PIL import Image
import numpy as np
import scipy.spatial.distance




def img_crop(img,w_crp,h_crp,new_width,new_height):
    return img[w_crp:w_crp+new_width,h_crp:h_crp+new_height]


#--------braind damage-----------------
#
# path="/storage/3050/MazurM/db/OrigCleanSegmentBulat/MaskAreBeingUsed/" # 4 channels
# name="LABEL_5_1.png"
path="/storage/3050/MazurM/db/OrigCleanSegmentBulat/learn/12220/" # 4 channels
name="LABEL_5.png"
mask_img=np.array(Image.open(path+name))
print("mask_img.shape= ",mask_img.shape)

#-----------
path_soruces_learn="/storage/3050/MazurM/db/OrigCleanSegmentBulat/learn/"

def getitem2(index,pixels_classes):

    #-------------------------------------------------------------------------------------------------------
    path_soruces_learn = "/storage/3050/MazurM/db/OrigNoiseSegmentBulat/learn/"  # masks and input images are storing in the same directory
    folder_names = []

    names_img_inputs_1 = ["bing_sat", "esri_sat", "google_sat",
                          "mapbox_sat_with_api"]  # 1 frist layer in input (satellite photo)
    extension_im_inputs_1 = "png"  # for ALL images!!!
    names_img_inputs_2 = ["another_osm__maps", "esri_map", "google_maps",
                          "cartodb"]  # 2 second layer in input (masks from Google,Yandex etc.)
    extension_im_inputs_2 = "png"  # for ALL images!!!

    name_mask_label = "LABEL_5"  # all names for mask are the SAME !!!
    extension_name_mask_label = "png"  # for ALL mask labels!!!

    all_names = os.listdir(path_soruces_learn)
    for name in all_names:
        if (os.path.isdir(path_soruces_learn + name)):
            folder_names.append(name)
    index_rand_img1=random.randint(0,len(names_img_inputs_1)-1)
    index_rand_img2=random.randint(0,len(names_img_inputs_2)-1)
    #-------------------------------------------------------------------------------------------------------
    mask_name="LABEL_5"
    extension="png"
    path="/storage/3050/MazurM/db/OrigCleanSegmentBulat/learn/12220/" # 4 channels
    # mask_img = np.array(Image.open(path_soruces_learn + folder_names[index]+"/"+name+"."+extension))
    with Image.open(path_soruces_learn + folder_names[index]+"/"+names_img_inputs_1[index_rand_img1]+"."+extension) as input_img_1, \
        Image.open(path_soruces_learn + folder_names[index]+"/"+names_img_inputs_2[index_rand_img2]+"."+extension) as input_img_2, \
        Image.open(path_soruces_learn + folder_names[index]+"/"+mask_name+"."+extension) as mask_img:
            mask_img=np.array(mask_img)
            print("da")
            width,height,*_=mask_img.shape
            new_size=256 # square
            new_w=random.randint(0,width-new_size)
            new_h=random.randint(0,height-new_size)
            mask_img_crp=img_crop(mask_img,new_w,new_h,new_size,new_size)
            mask_cls_crp=np.zeros((new_size,new_size))
            for w in range(new_size):
                print(w)
                for h in range(new_size):
                    # current_pix=mask_img_crp[w,h,0:3]
                    # print(current_pix)
                    # index_color=np.where((pixels_classes==mask_img_crp[w,h]).all(axis=1))[0]
                    # mask_cls_crp[w,h]=index_color
                    mask_cls_crp[w,h]=np.where((pixels_classes==mask_img_crp[w,h]).all(axis=1))[0]
            print("wtf")


pixels_classes = np.array(
        [[255, 255, 255, 255],
         [250, 10, 222, 255],
         [0, 255, 0, 255],
         [255, 0, 0, 255],
         [50, 0, 0, 255],
         [30, 0, 0, 255],
         [0, 0, 255, 255]])
getitem2(1,pixels_classes)
