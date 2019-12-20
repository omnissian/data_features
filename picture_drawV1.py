import numpy as np
import torch.utils.data as data
import pdb
import datetime
import cv2 as cv
from PIL import Image, ImageDraw


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


img=np.zeros((h,w,channels),np.uint8)
img.fill(255)
rectangle=[[25,25],[100,100]]
border=1
rect1=[(25,25),(100,100)]
rect1_border=[(25-border,25-border),(100+border,100+border)]
print(rectangle[0][:])
# cv.drawContours()

# cv.line()
cv.rectangle(img,rect1_border[0],rect1_border[1],black,-1)
cv.rectangle(img,rect1[0],rect1[1],green,-1)

# cv.rectangle(img,rect[0],rect[1],white,-1)


cv.imshow("Marat",img)



print("pause")
