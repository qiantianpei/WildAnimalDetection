import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import argparse
from torch.autograd import Variable
import torch


def composite( illumination, red, green, blue, percent ):
    '''
    Function: composite images from huge 3D matrices, where the length of z-channel is the # of photos to be composited
    Input: (1) illumination: illumination of each pixel in 'z' photos (z-channel length is # of photos to be composited)
           (2) red: R channel of each photo into one "red" matrix;
           (3) green: G channel of each photo into one "green" matrix;
           (4) blue: B channel of each photo into one "blue" matrix;
    Output: composited image of size (nrows, ncols, nchannels)
    Notice *: all items in the photo_cache list should be of exactly same size (because they have been resized) '''
    
    assert illumination.shape == red.shape == green.shape == blue.shape
    
    R = np.empty([red.shape[0],red.shape[1]])
    G = np.empty([red.shape[0],red.shape[1]])
    B = np.empty([red.shape[0],red.shape[1]])
    
    # illumination matrix is float type
    # selected_illu is also float type, because it's capturing percentile
    # cannot convert to int
    # even keeping as float is hard to have exact equality
    selected_illu = np.percentile(illumination, percent, axis=2, keepdims = True).astype(int)
    
    print(selected_illu)
    #print(type(selected_illu))
    
    #global s
    #global ill
    #s = selected_illu
    #ill = illumination
    
    for i in range(selected_illu.shape[0]):
        # find the closest element in a list to a fixed number
        #np.abs(test_matrix[0,:,:] - test_ref[0,:]).argmin(axis = 0)
        idx0 = np.abs(illumination[i,:,:] - selected_illu[i,:,:]).argmin(axis = 1)
        indices = [np.array(range(selected_illu[i,:,:].shape[0])),idx0]
        
        R[i,:] = red[i,:,:][indices]
        G[i,:] = green[i,:,:][indices]
        B[i,:] = blue[i,:,:][indices]
    
    return R,G,B


def get_gcc_composite(photos_cache, percent, camera, period, root):
    '''
    Function: computes gcc value for a dictionary or list of photos. It also saves the composite image
    Input: (1) a cache of photos; (2) week of cache
           (3) group of camera; (4) root directory of JR server folder; 
           (5) format of cache: "dic" or "list", if "dic", then the cache is in tree-like structure;
           (6) illumination percentage, a reasonable illumination level for composite() to search.
    Output: GCC value of the composite image, made from the photos_cache
    Notice *: this composite image method does not need to call gcc() after getting the composite image
    '''
    print(root)
    #opt_size = (1000, 1500)
    opt_size = (224, 224)
    # Then do the resize and following manipulation
    red_matrix = np.empty([opt_size[0] , opt_size[1], 100])
    green_matrix = np.empty([opt_size[0] , opt_size[1], 100])
    blue_matrix = np.empty([opt_size[0] , opt_size[1], 100])
    illumination_matrix = np.empty([opt_size[0] , opt_size[1], 100])

    k = 0
    for index in tqdm(np.random.choice(len(photos_cache), min(len(photos_cache),100), replace = False)):
        print(index)
        try: 
            #image = transforms.Resize((1000,1500))(Image.open(os.path.join(root, photos_cache.loc[index, "up_name"])))
            #image = transforms.Resize((1000,1500))(Image.open(os.path.join(root, photos_cache.loc[index, "up_name"])))
            image = transforms.Resize((224,224))(Image.open(os.path.join(root, photos_cache.loc[index, "up_name"])))
            item = np.asarray( image, dtype="int32" )
            Y = 16 + 65.481/255 * item[:,:,0] + 128.553/255 * item[:,:,1] + 24.966/255 * item[:,:,2]
            #print(Y.shape)
            illumination_matrix[:,:,k] = Y
            red_matrix[:,:,k] = item[:,:,0]
            green_matrix[:,:,k] = item[:,:,1]
            blue_matrix[:,:,k] = item[:,:,2]
            k += 1
        except Exception as ex:
            print(ex)

    R, G, B = composite( illumination_matrix, red_matrix, green_matrix, blue_matrix, percent )

    # save the composite image
    output_img = np.empty([R.shape[0],R.shape[1],3])
    output_img[:,:,0] = R
    output_img[:,:,1] = G
    output_img[:,:,2] = B

    Image.fromarray(output_img.astype('uint8'), 'RGB').save("composite_images_large/" + period + "_" + "_".join(camera.split("/")) + '.png')
    

def main():
    
    
    for period in ["day", "night"]:
        period_info = pd.read_excel("metadata/" + period + "_final_train.xlsx")
        root = "data_" + period  + "_final/train_signs"
        print(len(period_info))
        none_period_images = period_info[period_info["Species"] == 0]
        print(none_period_images["CameraPath"].value_counts())
        camera_list = none_period_images["CameraPath"].unique().tolist()
        camera_list = ['CAMERA_ARRAY_B/camera_B2']
        for camera in camera_list:
            none_camera_images = none_period_images[none_period_images["CameraPath"] == camera]
            none_camera_images = none_camera_images.reset_index(drop = True)
            print(period + " " + camera + ": " + str(len(none_camera_images)))
            get_gcc_composite(none_camera_images, 70, camera, period, root)
            
    


#-------------------------------
if __name__ == "__main__":
    main()