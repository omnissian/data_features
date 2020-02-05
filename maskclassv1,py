#--------window class creator--------------
#white=255,255,255
#red 255,0,0
#brown 1 [30,0,0]  br1
#brown 2 [50,0,0]  br2
#----------------------------
# index,class,R,G,B
# 0,Background,255,255,255
# 1,Building,0,255,0
# 2,Bridge,30,0,0
# 3,Railroad,50,0,0
# 4,Road,255,0,0
# 5,Hydrography,0,0,255
# 6,Building_edge,250,10,222
#----------------------------
#pink [250,10,222]          r        g            b       br1    br 2        pink      background
unique_pixels=np.array([[255,0,0],[0,255,0],[0,0,255],[30,0,0],[50,0,0],[250,10,222],[255,255,255]])
# overall 7 classes

# #----p.s. right now without window, only between one pixel similarity
# path_noise_mask="C:/Users/user/Desktop/DontTouchPLSpytorch/data/DB_2020-01-30/Temp/12220/"
# noisy_mask_name="LABEL_5.png"
path_noise_mask="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/CleanMapLabel2/trainB/"
noisy_mask_name="18.png"
path_new_mask="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/CleanMapForTestOnly1/12220/"
new_mask_name="12220.png"
mask_new=[]
brown1=np.array([30,0,0])
brown2=np.array([50,0,0])
pink=np.array([250,10,222])

with Image.open(path_noise_mask+noisy_mask_name) as mask:
    # noise_img=np.random.randint(255, size=(128, 128,3))
    mask=np.array(mask)
    plt.imshow(mask)
    mask=mask[:,:,:3]
    mask=mask # normalise
    width, height,channels=mask.shape
    unique_pixels=unique_pixels/255.0 # normalise
    mask_new=np.zeros(mask.shape,dtype=np.float64)
    new_pix =np.copy(unique_pixels[0])
    for h in range(height):

        for w in range(width):
            # print(mask[h, w])
            # print(mask[h, w].shape) #shape = (3,)
            similarity=0
            if (np.array_equal(mask[h, w], brown1)):
                mask_new[h, w] = brown1
                continue
            elif (np.array_equal(mask[h, w], brown2)):
                mask_new[h, w] = brown2
                continue
            elif (np.array_equal(mask[h, w], pink)):
                mask_new[h, w] = pink
                continue
            for item in (unique_pixels):
                # print(item)
                # print(item.shape)
                # print(type(item))
                # print(similarity)
                # print("dot product between: ")
                # print("uniq pixel=", item)
                # print("image pixel=", mask[h,w])
                # if(item==(np.array([250,10,222])/255.0)):
                # if(np.ndarray.all(item,np.array([250,10,222])/255.0)):
                #     print("debug")


                current=mask_new[h,w]/255.0
                new_dist=np.dot(item,current) # 4 and 3
                if (new_dist>similarity):
                    similarity=new_dist
                    new_pix=np.copy(item)
            mask_new[h,w]=new_pix
        print("line ",h,"done")
    if(channels < 4):
        alpha_channel=np.ones((height,width,1))
        mask_new=np.concatenate((mask_new,alpha_channel),axis=2)
    else:
        pass

plt.imshow(mask_new)

mask_new=Image.fromarray(np.uint8(mask_new*255.0))

mask_new.save(path_new_mask+new_mask_name) # used for save From PNG to PNG
