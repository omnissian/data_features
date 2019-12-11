import shutil
import random
import os

random.seed(33)
path_big_set_in="/home/std_11/MazurM/DB_cut_images/DB_cut/google/"
path_big_set_mask="/home/std_11/MazurM/DB_cut_images/DB_cut/google/all_masks"
names_big_in=os.listdir(path_big_set_in)
names_big_masks=os.listdir(path_big_set_mask)


new_path_learn_in="/home/std_11/MazurM/DB_cut_images/DB_cut/google/"
new_path_learn_mask="/home/std_11/MazurM/DB_cut_images/DB_cut/google/"
new_path_valid_mask="/home/std_11/MazurM/DB_cut_images/DB_cut/google/"
new_path_valid_mask="/home/std_11/MazurM/DB_cut_images/DB_cut/google/"

num_train_copy=0
num_valid_copy=0

size_of_set=len(names_big_masks)-1

# for i in range(10):
#     print("i=",i, random.randint(0,10))
try:
    random.sample(range(0,size_of_set),num_train_copy)
except ValueError:
    print("Sample size exceeded population size.")





















