import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import pdb
import os
import torch
import matplotlib.pyplot as plt


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        # """Initialize this dataset class.

        # Parameters:
        #     opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        # """
        # BaseDataset.__init__(self, opt)
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        # self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        # self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        # self.A_size = len(self.A_paths)  # get the size of dataset A
        # self.B_size = len(self.B_paths)  # get the size of dataset B
        # btoA = self.opt.direction == 'BtoA'
        # input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        # output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        # self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))##<<<-----clean was input_nc == 1)
        # self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))##<<<-----clean was output_nc == 1)
        # # self.mask = np.ones(256, 256)#<<<----------------------------------
        self.path_a="/storage/3050/MazurM/db/pix2pixset1/trainA/"
        self.path_b="/storage/3050/MazurM/db/pix2pixset1/trainB/"

        self.A_paths=[]
        self.B_paths=[]

        self.list_names_full=os.listdir(self.path_a)
        self.list_names=[]
        for i in range(len(self.list_names_full)):
            name=self.list_names_full[i]
            name=name[0:-4]
            self.list_names.append(name)

        self.A_size=len(self.list_names_full)
        self.B_size=len(self.list_names_full)




    def __getitem__(self, index):
        # #     #<<<------------------------------START----------------------------------------
        # """Return a data point and its metadata information.

        # Parameters:
        #     index (int)      -- a random integer for data indexing

        # Returns a dictionary that contains A, B, A_paths and B_paths
        #     A (tensor)       -- an image in the input domain
        #     B (tensor)       -- its corresponding image in the target domain
        #     A_paths (str)    -- image paths
        #     B_paths (str)    -- image paths
        # """
        # A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        # if self.opt.serial_batches:   # make sure index is within then range
        #     index_B = index % self.B_size
        # else:   # randomize the index for domain B to avoid fixed pairs.
        #     index_B = random.randint(0, self.B_size - 1)
        # B_path = self.B_paths[index_B]
        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')#<<<----------clean----------------
        # # B_img = Image.open(B_path)

        # # img_tmp = B_img.copy()#<<<----------------------------------

        # # for i in range(256):
        # #     for j in range(256):
        # #         if img_tmp[i, j, :]==1.0:
        # #             self.mask[i,j]=0
        # #     #<<<------------------------------------------------------------------


        # # apply image transformation
        # A = self.transform_A(A_img)
        # B = self.transform_B(B_img)
        # print("A_path",A_path)
        # print("B_path",B_path)
        # input()
        # pdb.set_trace()
#Image.open(path+original_names[i]+'_real_A'+extension)
        extension_a=".jpg"
        extension_b=".png"
        if (index>=len(self.list_names)):
            index=random.randint(0,len(self.list_names))
            A_path=self.path_a+self.list_names[index]+extension_a
            B_path=self.path_b+self.list_names[index]+extension_b
            # A=Image.open(path_a+list_names[index]+extension_a)
            # B=Image.open(path_a+list_names[index]+extension_a)
            # pdb.set_trace()
            A=torch.Tensor((np.array(Image.open(A_path))).reshape(3,256,256))# .convert('RGB'))
            B=torch.Tensor((np.array(Image.open(B_path).convert('RGB'))).reshape(3,256,256))# .convert('RGB')) # RGB by default???
            # A=torch.from_numpy(A).float()
            # B=torch.from_numpy(B).float()
            A=A/255.0
            B=B/255.0
        else:
            A_path=self.path_a+self.list_names[index]+extension_a
            B_path=self.path_b+self.list_names[index]+extension_b
            # A=Image.open(A_path).convert('RGB')
            # B=Image.open(B_path).convert('RGB') # RGB by default???
            # pdb.set_trace()
            # A=torch.Tensor((np.array(Image.open(A_path),dtype=np.uint8)).reshape(3,256,256))# .convert('RGB')) # correct  <<<----------
            # B=torch.Tensor((np.array(Image.open(B_path).convert('RGB'))).reshape(3,256,256))# .convert('RGB')) # RGB by default??? correct  <<<----------
            # A=torch.Tensor((np.array(Image.open(A_path))))# .convert('RGB')) #   <<<----------WORKABLE???
            # B=torch.Tensor((np.array(Image.open(B_path).convert('RGB'),dtype=np.uint8)))# .convert('RGB')) # RGB by default???  <<<----------WORKABLE???
            A=torch.Tensor(np.array(Image.open(A_path))/255.0)
            A=A.permute(2,0,1)
            B=torch.Tensor(np.array(Image.open(B_path))/255.0)
            B=B.permute(2,0,1)
            plt.imshow(B)
            plt.show()
            pdb.set_trace() # <<<-----------------------------------------
            # wtfA=Image.open(A_path)
            # wtfA.show()
            # wtfB=Image.open(B_path)
            # wtfB.show()


            #----------------------------
            #----------real A
            # imageA=Image.open(A_path)
            # imgA=np.array(imageA) #/255.0
            # imgA=torch.Tensor(imgA)
            # imgA=np.asarray(imgA)
            # plt.imshow(imgA)
            # plt.show()
            #----------real B
            imageB=Image.open(B_path)
            imgB=np.array(imageB) #/255.0
            imgB=torch.Tensor(imgB)
            imgB=np.asarray(imgB)
            plt.imshow(imgB)
            plt.show()

            #----------------------------



            wtfA=Image.open(A_path)
            wtfB=Image.open(B_path)
            wtfA=np.array(wtfA)/255.0
            plt.imshow(wtfA)
            plt.show()
            wtfA=torch.Tensor(wtfA)
            plt.imshow(wtfA)
            plt.show()
            wtfA=np.asarray(wtfA) #
            # wtfA=wtfA/255.0 #


            plt.imshow(wtfA)
            plt.show()
            wtfB=Image.open(B_path)
            wtfB.show()




            imgA=np.array(A)
            imgA=Image.fromarray(imgA,'RGB')
            imgA.show()

            A=A.permute(2,0,1)
            B=B.permute(2,0,1)

            # A=torch.Tensor(Image.open(A_path))# .convert('RGB'))
            # B=torch.Tensor(Image.open(B_path))# .convert('RGB')) # RGB by default???
            A=A/255.0
            B=B/255.0
            imgA=np.asarray(A).reshape(256,256,3)
            imgA=Image.fromarray(imgA,'RGB')
            imgA.show()
            # plt.imshow()

            print("pause")


            c=input()

            # A=torch.from_numpy(A).float()
            # B=torch.from_numpy(B).float()



        #     #<<<------------------------------END----------------------------------------
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path} #<<<------------- clean
        # return {'A': A, 'B': B, 'mask': self.mask,  'A_paths': A_path, 'B_paths': B_path} #<<<------------- 'mask': mask, 

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        # return 1
        return max(self.A_size, self.B_size)
