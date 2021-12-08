import numpy
import sys
import os

sys.path.insert(0, "C:\\Personnal\\M2\\Vision\\project\\part_2")
os.chdir("C:\\Personnal\\M2\\Vision\\project\\part_2")

import calculate as clc 
import cv2




def main():
    
    pixel_coords, mask, normals = clc.get_normal()
    
    crop_dims = clc.get_mask_info(mask)
    
    mask = clc.crop(normals, crop_dims)
    normals = clc.crop(normals, crop_dims)
    
    h, w, _ = normals.shape
     
    for i range(h):
        
        previous = np.array([0,0,0])
        
        for j in range(w):
            if mask[i,j] != 0:
                
                
                
                    
    
    pass


if __name__ == "__main__":
    
    _, mask, normals = clc.get_normal()
    cv2.imshow("", mask)
    cv2.waitKey(0)

