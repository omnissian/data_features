path_noise_mask="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/CleanMapLabel2/trainB/"
noisy_mask_name="18.png"
path_new_mask="C:/Users/user/Desktop/DontTouchPLSpytorch/Polygon/img_saved/CleanMapForTestOnly1/12220/"
new_mask_name="12220.png"
mask_new=[]
brown1=np.array([30,0,0])
brown2=np.array([50,0,0])
pink=np.array([250,10,222])

with Image.open(path_noise_mask+noisy_mask_name) as mask:
    mask=np.array(mask)
    plt.imshow(mask)
    mask=mask[:,:,:3]
    width, height,channels=mask.shape
    mask_new=np.zeros(mask.shape,dtype=np.float64)
    for h in range(height):
        for w in range(width):
            similarity=0.0
            if (np.array_equal(mask[h, w], brown1)):
                mask_new[h, w] = brown1/255.0
                continue
            elif (np.array_equal(mask[h, w], brown2)):
                mask_new[h, w] = brown2/255.0
                continue
            elif (np.array_equal(mask[h, w], pink)):
                mask_new[h, w] = pink/255.0
                continue
            current=mask[h,w]/255.0
            for item in (unique_pixels):
                new_dist=np.dot(item,current) # 4 and 3
                if (new_dist>similarity):
                    similarity=new_dist
                    new_pix=item
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
