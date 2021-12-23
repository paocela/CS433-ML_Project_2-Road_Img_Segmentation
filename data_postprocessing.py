patch_size = 16 

def process_road_holes(original_imgs):
    imgs = original_imgs

    # for each image, correct alone road patches
    for index, im in enumerate(imgs):

        # traverse all patches
        for j in range(0, im.shape[1], patch_size): 
            for i in range(0, im.shape[0], patch_size): 
                patch = im[i:i + patch_size, j:j + patch_size, :] 

                if patch.all() == 0:
                  continue
                
                val = 1

                # remove road patch if its "alone" between background patches
                if j + 2*patch_size < im.shape[1] and j - patch_size >= 0:
                    if i + 2*patch_size < im.shape[0] and i - patch_size >= 0:

                        patch_right = im[i:i + patch_size, j + patch_size:j + 2*patch_size, :]
                        patch_left = im[i:i + patch_size, j - patch_size:j, :]
                        patch_top = im[i - patch_size:i, j:j + patch_size, :]
                        patch_bottom = im[i + patch_size:i + 2*patch_size, j:j + patch_size, :]

                        if patch_right.all() == 0 and patch_left.all() == 0 and patch_bottom.all() == 0 and patch_top.all() == 0:
                            val = 0

                  
                im[i:i + patch_size, j:j + patch_size, :] = val
                # save postprocessed image
        imgs[index] = im
    

    
    # for each image, correct 1 or 2 road holes (with a patch granularity)
    for index, im in enumerate(imgs):

        # traverse all patches
        for j in range(0, im.shape[1], patch_size): 
            for i in range(0, im.shape[0], patch_size): 
                patch = im[i:i + patch_size, j:j + patch_size, :] 
                val = 0
                # skip if patch label is already row (so 1)
                if patch.all() == 1:
                  continue

                # check horizontally (for 1 patch holes)
                if j + 2*patch_size < im.shape[1] and j - patch_size >= 0:
                  patch_right = im[i:i + patch_size, j + patch_size:j + 2*patch_size, :]
                  patch_left = im[i:i + patch_size, j - patch_size:j, :]

                  # if near both road, then assign road
                  if patch_right.all() == 1 and patch_left.all() == 1:
                    val = 1
                  
                  # check horizontally (for 2 patch holes)
                  if val == 0 and j + 3*patch_size < im.shape[1] and j - 2*patch_size >= 0: 
                    patch_2right = im[i:i + patch_size, j + 2*patch_size:j + 3*patch_size, :]
                    patch_2left = im[i:i + patch_size, j - 2*patch_size:j - patch_size, :]
                  
                    if patch_right.all() == 1 and patch_2left.all() == 1:
                      val = 1
                    elif patch_left.all() == 1 and patch_2right.all() == 1:
                      val = 1
                      

                # check vertically (for 1 patch holes)
                if i + 2*patch_size < im.shape[0] and i - patch_size >= 0:
                  patch_top = im[i - patch_size:i, j:j + patch_size, :]
                  patch_bottom = im[i + patch_size:i + 2*patch_size, j:j + patch_size, :]

                  # if near both road, then assign road
                  if patch_top.all() == 1 and patch_bottom.all() == 1:
                    val = 1
                  
                  # check vertically (for 2 patch holes)
                  if val == 0 and i + 3*patch_size < im.shape[0] and i - 2*patch_size >= 0: 
                    patch_2top = im[i - 2*patch_size:i - patch_size, j:j + patch_size, :]
                    patch_2bottom = im[i + 2*patch_size:i + 3*patch_size, j:j + patch_size, :]
                  
                    if patch_top.all() == 1 and patch_2bottom.all() == 1:
                      val = 1
                    elif patch_bottom.all() == 1 and patch_2top.all() == 1:
                      val = 1
                      
                  
                im[i:i + patch_size, j:j + patch_size, :] = val
                
    return imgs
        



