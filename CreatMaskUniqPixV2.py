from PIL import Image
import torch.utils.data as data
import pdb
import datetime
from PIL import Image, ImageDraw
import numpy as np
import os
import cv2

pixels_classes = np.array( # no pink
    [[255, 255, 255], # background
     [0, 255, 0],
     [255, 0, 0],
     [50, 0, 0],
     [30, 0, 0],
     [0, 0, 255]],dtype=np.uint8)
     
     
path_big_masks="/storage/3050/MazurM/db/OrigNoiseSegmentBulat/MasksOldBulat1/"
path_save="/storage/3050/MazurM/db/OrigNoiseSegmentBulat/MasksOldBulat1Cleaned/"

dict_uniq_pix={} # np.frombuffer(input,dtype=np.uint8)- return the object
def check_uniq_pix(path_big_masks,dict_uniq_pix,list_uniq_pix_numpy):

    names=os.listdir(path_big_masks) #names- list
    for name in names:
        print("image ", name,"started")
        mask=cv2.imread(path_big_masks+name)
        mask=cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask=np.array(mask)  # type(mask) = <class 'numpy.ndarray'> dtype=uint8) shape= 1664,1664, 3
        widht,height,ch=mask.shape
        for w in range (widht):
            for h in range(height):
                # pix_current=np.tobytes(mask[w,h])
                pdb.set_trace()
                pix_current=mask[w,h].tobytes()
                if pix_current in dict_uniq_pix:
                    dict_uniq_pix[pix_current]+=1
                else:
                    dict_uniq_pix[pix_current]=1
        print("image ", name,"ended")
    for k in dict_uniq_pix.keys():
        tmp=np.frombuffer(k,dtype=np.uint8) 
        list_uniq_pix_numpy.append(tmp)
        

def create_mask(path_big_masks,path_save,pixels_classes,file_type=".png"):
    class_len=len(pixels_classes)
    names=os.listdir(path_big_masks) #names- list
    for name in names:
        print("image ", name,"started")
        mask=cv2.imread(path_big_masks+name,1) 
        mask=cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)       
        mask=np.array(mask,dtype=np.uint8)  # type(mask) = <class 'numpy.ndarray'> dtype=uint8) shape= 1664,1664, 3
        mask_class=np.copy(mask[:,:,0]) # ground truth - NON colored mask  mask_class[1570][1662]
        mask_colored =np.copy(mask)  # image COLORED mask UNIQ pixels  
        width,height,ch=mask.shape
        for w in range(width):
            print("line ", w)
            for h in range(height):
                tmpcmp=(mask[w,h]==pixels_classes[:]).all(axis=1)
                if(tmpcmp.any()):
                    class_id=np.where(tmpcmp)[0][0]
                    mask_class[w,h]=class_id
                    mask_colored[w,h]=pixels_classes[class_id]
                else:
                    mask_class[w,h]=0
                    mask_colored[w,h]=pixels_classes[0]
        
        mask_colored=cv2.cvtColor(mask_colored,cv2.COLOR_RGB2BGR)
        cv2.imwrite(path_save+"7upix_"+name,mask_colored)
     
     
     


