from PIL import Image
import torch.utils.data as data
import pdb
import datetime
from PIL import Image, ImageDraw
import numpy as np
import os
import cv2



# pixels_classes = np.array( # no pink
#     [[255, 255, 255, 255],
#      [0, 255, 0, 255],
#      [255, 0, 0, 255],
#      [50, 0, 0, 255],
#      [30, 0, 0, 255],
#      [0, 0, 255, 255]],dtype=np.uint8)
# pixels_classes=pixels_classes[:,0:3]

pixels_classes = np.array( # no pink
    [[255, 255, 255], # background
     [0, 255, 0],
     [255, 0, 0],
     [50, 0, 0],
     [30, 0, 0],
     [0, 0, 255]],dtype=np.uint8)

# third=pixels_classes[3].tobytes()
# fourth=pixels_classes[4].tobytes() 
# print("third_class", third)
# third_ar=np.frombuffer(third,dtype=np.uint8)
# print(np.frombuffer(third,dtype=np.uint8))

path_big_masks="/storage/3050/MazurM/db/OrigNoiseSegmentBulat/MasksOldBulat1/"
path_save="/storage/3050/MazurM/db/OrigNoiseSegmentBulat/MasksOldBulat1Cleaned/"
    # for k in uniq_pixels_dict.keys():
    #     list_of_pixels.append(np.frombuffer(k,dtype=np.uint8))
    #     wtf=np.frombuffer(k,dtype=np.uint8)
    #     print(k, np.frombuffer(k,dtype=np.uint8))
    #     print(uniq_pixels_dict[k])
list_uniq_pix_numpy=[]

dict_uniq_pix={} # np.frombuffer(input,dtype=np.uint8)- return the object
def check_uniq_pix(path_big_masks,dict_uniq_pix,list_uniq_pix_numpy):

    names=os.listdir(path_big_masks) #names- list
    for name in names:
        print("image ", name,"started")
        mask=cv2.imread(path_big_masks+name)
        mask=cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # with cv2.imread(path_big_masks+name,1) as mask:
        # with Image.open(path_big_masks+name) as mask:
        # pdb.set_trace() # <<<===
        mask=np.array(mask)  # type(mask) = <class 'numpy.ndarray'> dtype=uint8) shape= 1664,1664, 3
        # pdb.set_trace()
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

def check_uniq_pix_NUMPY(path_big_masks,dict_uniq_pix,list_uniq_pix_numpy):

    names=os.listdir(path_big_masks) #names- list
    for name in names:
        print("image ", name,"started")
        mask=cv2.imread(path_big_masks+name,1)
        # with cv2.imread(path_big_masks+name,1) as mask:
        # with Image.open(path_big_masks+name) as mask:
        # pdb.set_trace() # <<<===
        mask=np.array(mask)  # type(mask) = <class 'numpy.ndarray'> dtype=uint8) shape= 1664,1664, 3
        # pdb.set_trace()
        widht,height,ch=mask.shape
        for w in range (widht):
            for h in range(height):
                # pix_current=np.tobytes(mask[w,h])
                pdb.set_trace()
                pix_current=mask[w,h]
                # cmptmp=(pix_current=pixels_classes[:]).all(1)
                if pix_current in dict_uniq_pix:
                    dict_uniq_pix[pix_current]+=1
                else:
                    dict_uniq_pix[pix_current]=1
        print("image ", name,"ended")
    for k in dict_uniq_pix.keys():
        tmp=np.frombuffer(k,dtype=np.uint8) 
        list_uniq_pix_numpy.append(tmp)


# check_uniq_pix(path_big_masks,dict_uniq_pix,list_uniq_pix_numpy)
# pdb.set_trace() # <<<----

# names=os.listdir(path_big_masks) #names- list   
# img = cv2.imread(path_big_masks+names[0]) #  class 'numpy.ndarray'dtype=uint8



# def create_mask(path_big_masks,path_save,pixels_classes,file_type=".png"):
#     class_len=len(pixels_classes)
#     names=os.listdir(path_big_masks) #names- list
#     for name in names:
#         print("image ", name,"started")
#         mask=cv2.imread(path_big_masks+name,1)        
#         mask=np.array(mask,dtype=np.uint8)  # type(mask) = <class 'numpy.ndarray'> dtype=uint8) shape= 1664,1664, 3
#         mask_class=np.copy(mask[:,:,0]) # ground truth - NON colored mask
#         mask_colored =np.copy(mask)  # image COLORED mask UNIQ pixels
#         # pdb.set_trace()
#         width,height,ch=mask.shape
#         for w in range(width):
#             print("line ", w)
#             for h in range(height):
#                 # pdb.set_trace() 
#                 pix_class=0 # 0 - background class
#                 for i in range(class_len):
#                     # if (mask[w,h]==pixels_classes[i]).all(): # both shape (3,)
#                     # if np.array_equal(mask[w,h],pixels_classes[i]): # both shape (3,) # workable example
#                     if ((mask[w,h]==pixels_classes[i]).all()):
#                         # mask_class[w,h]=i
#                         if(i==3 or i==4):
#                             print("check pixel!!!")
#                             pdb.set_trace()
#                         pix_class=i
#                         break
#                     # else:
#                     #     mask_class[w,h]=0
#                     #     mask_colored[w,h]=pixels_classes[0]
#                 mask_class[w,h]=pix_class
#                 mask_colored[w,h]=pixels_classes[pix_class]
#         cv2.imwrite(path_save+"clean"+name,mask_colored)
#         pdb.set_trace()



def create_mask(path_big_masks,path_save,pixels_classes,file_type=".png"):
    class_len=len(pixels_classes)
    names=os.listdir(path_big_masks) #names- list
    for name in names:
        print("image ", name,"started")
        mask=cv2.imread(path_big_masks+name,1) 
        # mask=cv2.imread(path_big_masks+name)
        mask=cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)       
        mask=np.array(mask,dtype=np.uint8)  # type(mask) = <class 'numpy.ndarray'> dtype=uint8) shape= 1664,1664, 3
        mask_class=np.copy(mask[:,:,0]) # ground truth - NON colored mask  mask_class[1570][1662]
        mask_colored =np.copy(mask)  # image COLORED mask UNIQ pixels  
        width,height,ch=mask.shape
        # pdb.set_trace()
        for w in range(width):
            print("line ", w)
            for h in range(height):
                tmpcmp=(mask[w,h]==pixels_classes[:]).all(axis=1)
                # if(((mask[w,h]==pixels_classes[3]).all()).any()):
                #     print("THIRD CLASS")
                if(tmpcmp.any()):
                    class_id=np.where(tmpcmp)[0][0]
                    mask_class[w,h]=class_id
                    mask_colored[w,h]=pixels_classes[class_id]
                    # if(class_id==3 or class_id==4):
                    #     print("class_id = ",class_id)
                else:
                    mask_class[w,h]=0
                    mask_colored[w,h]=pixels_classes[0]
        
        # pdb.set_trace()
        mask_colored=cv2.cvtColor(mask_colored,cv2.COLOR_RGB2BGR)
        cv2.imwrite(path_save+"7upix_"+name,mask_colored)



print("STEP")

# check_uniq_pix(path_save,dict_uniq_pix,list_uniq_pix_numpy)
# pdb.set_trace()


create_mask(path_big_masks,path_save,pixels_classes,file_type=".png")
pdb.set_trace()



# #----------other code------------------------
# def uniq_pix(dictionary,path,name=0):
#     if(name==0):
#         names=os.listdir(path)
#         for name in names:
#             path_current=path+name
#             with Image.open(path_current) as mask_original:
#                 height,width=mask_original.size
#                 mask_original=np.asarray(mask_original)
#                 mask_original=mask_original[:,:,0:3] # <<<<<<<<<< first three channels because fourth chanllen in PNH is always has value = 255
#                 for w in range(width):
#                     for h in range(height):
#                         # wtf=mask_original[h,w]
#                         # print(mask_original[h,w])
#                         pix_current=mask_original[h,w].tobytes()
#                         if pix_current in dictionary:
#                             dictionary[pix_current]+=1
#                         else:
#                             dictionary[pix_current]=1

# #----------other code----------
# #-----------unque checker not mast creator---------
# pdb.set_trace()
# for name in names:
#     with Image.open(path_big_masks+name) as mask:
#          uniq_pixels_dict={}
#             uniq_pix(uniq_pixels_dict,path_mask_original_img,name)
#             print(len(uniq_pixels_dict))
#             img_mask=np.asarray(img_mask)
#             img_mask=img_mask[:,:,0:3]
#             width, height, channels=img_mask.shape
#             mask = np.zeros((width,height,classes),dtype=np.uint8) #  returned mask
#             list_of_pixels=[]
#             for k in uniq_pixels_dict.keys():
#                 list_of_pixels.append(np.frombuffer(k,dtype=np.uint8))
#                 wtf=np.frombuffer(k,dtype=np.uint8)
#                 print(k, np.frombuffer(k,dtype=np.uint8))
#                 print(uniq_pixels_dict[k])

#             for row in range(classes):
#                 print(row," : ",pixels_classes[row])
#             for w in range (width):
#                 for h in range(height):
#                     wtf1=img_mask[w,h]
#                     class_index=np.where((pixels_classes==img_mask[w,h]).all(axis=1))[0]
#                     # print("class_index=",class_index)
#                     # print("img_mask[w,h]= ",img_mask[w,h])
#                     mask[w,h,class_index]=1
#             print("cycle for image Ended, please check")
#             pdb.set_trace()

#-----------------------------------------------





