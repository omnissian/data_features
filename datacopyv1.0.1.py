import shutil
import random
import os

random.seed(33)
# path_big_set_in="/home/std_11/MazurM/DB_cut_images/DB_cut/google/"
path_big_set_in="/home/std_11/MazurM/DataMikhailBackup/DB_cut_images/DB_cut/google/" # BIG IN
path_big_set_mask="/home/std_11/MazurM/DataMikhailBackup/DB_cut_images/DB_cut/build/mask_all/"# BIG MASK

new_path_train_in="/home/std_11/MazurM/UnetV1/DataSet1/db_small_1/train/in/"
new_path_train_mask="/home/std_11/MazurM/UnetV1/DataSet1/db_small_1/train/mask/"
new_path_valid_in="/home/std_11/MazurM/UnetV1/DataSet1/db_small_1/valid/in/"
new_path_valid_mask="/home/std_11/MazurM/UnetV1/DataSet1/db_small_1/valid/mask/"



# path_big_set_mask="/home/std_11/MazurM/DataMikhailBackup/DB_cut_images/DB_cut/all_mask/" #5 classes
# #the path above is for: green=building, red=road, white=background(earth), blue=water,black=bridges
names_big_in=os.listdir(path_big_set_in)
names_big_masks=os.listdir(path_big_set_mask)

# new_path_learn_in="/home/std_11/MazurM/DataMikhailBackup/DB_cut_images/DB_cut/small_db_1/train/in/"
# new_path_learn_mask="/home/std_11/MazurM/DataMikhailBackup/DB_cut_images/DB_cut/small_db_1/train/mask/"
# new_path_valid_mask="/home/std_11/MazurM/DB_cut_images/DB_cut/google/"
# new_path_valid_mask="/home/std_11/MazurM/DB_cut_images/DB_cut/google/"


num_train_copy=1500
num_valid_copy=450

ratio=0.75 # ratio examples train/valid

total_copy=num_train_copy+num_valid_copy #distinguish train examples from valid



size_of_set=len(names_big_masks)-2

if(total_copy>=size_of_set):
    total_copy=size_of_set
    num_train_copy=int(round(ratio*total_copy,0))
    num_valid_copy=int(round(((1.0-ratio)*total_copy)-1,0))
# for i in range(10):
#     print("i=",i, random.randint(0,10))

# if (num_train_copy>size_of_set):
#     num_train_copy=size_of_set
#     print("num_train_copy=",size_of_set)
# if (num_valid_copy>size_of_set):
#     num_valid_copy=size_of_set
#     print("num_valid_copy=",size_of_set)

# for i in range()
x=random.sample(range(0,size_of_set),total_copy)


# for i in range(num_train_copy):
#     name=names_big_masks[x[i]]
#     shutil.copy(path_big_set_in+name,new_path_train_in+name)
#     shutil.copy(path_big_set_mask+name,new_path_train_mask+name)

for i in range(num_train_copy,size_of_set):
    name=names_big_masks[x[i]]
    shutil.copy(path_big_set_in+name,new_path_valid_in+name)
    shutil.copy(path_big_set_mask+name,new_path_valid_mask+name)


# for i in range(len(x)):
#     name=names_big_masks[x[i]]
#     print(names_big_masks[x[i]])
#     shutil.copy(path_big_set_in+name,new_path_learn_in+name)
#     shutil.copy(path_big_set_mask+name,new_path_learn_mask+name)
#     # print(names_big_masks[x[i]])



print("break point")


