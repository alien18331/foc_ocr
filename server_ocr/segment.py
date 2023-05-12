# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 09:26:11 2022

@author: 00048766
"""

import cv2
import os 
import matplotlib.pyplot as plt

path = "C:\\Users\\00048766\\Downloads\\IPCamera1229"

out_path = path + "\\proc\\"

for filename in os.listdir(path):
    if(filename[-3:]=="jpg"):
        file = path+"\\"+filename
                
        
        # load
        frame = plt.imread(file)
        
        # Cropped
        region_topx = 420
        region_topy = 860
        region_bottomx = 540
        region_bottomy = 1180
        Cropped = frame[region_topx:region_bottomx, region_topy:region_bottomy]
        
        plt.imshow(Cropped, 'gray')
        plt.show()
        
        # save
        output = out_path+filename
        file_bgr = cv2.cvtColor(Cropped, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output, file_bgr)
