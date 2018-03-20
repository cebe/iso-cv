# -*- coding: utf-8 -*-
"""
Created: 2017/10/25
Last Update: 2018/03/20
Version 2.9.1
@author: Moritz Lürig
"""

# %% import packages

import cv2
import os
import copy
import numpy as np
import numpy.ma as ma
from collections import Counter
import fileinput

#import shutil
#import re
#from PIL import Image



#%% set directories

# make this your directory
os.chdir("E:\GitHub\\iso-cv\\python\\camera")

# you need thee folders:
in_dir = "in" # raw colour images
gray_dir = "gray" # grayscale images with adjusted brightness
out_dir = "out" # output directory with control images and text files

for folder in [in_dir, gray_dir, out_dir]:
    if not os.path.exists(folder):
        os.makedirs(folder)


#%% adjust grayscale
         
ref = 240 # set reference value, depending on your brightness (fairly arbitrarily, between 180 and 248) - all images will have their histograms adjusted to this value

for i in os.listdir(in_dir):
    if i.endswith(".jpg") or i.endswith(".JPG"):
        
        new_img_name = os.path.splitext(i)[0] + "_gray" +  ".jpg"
        
# read imgs, check if exists to prevent overwriting
        if not os.path.isfile(os.path.join(gray_dir, new_img_name)):
            img = cv2.imread(os.path.join(in_dir, i),0)

# reduce resolution by 50% - important if you have a lot of images
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
            
# threshold and collapse pixels into vector 
            ret,thresh_img = cv2.threshold(img,245,0,cv2.THRESH_TOZERO_INV)
            erosion = cv2.erode(thresh_img,np.ones((7,7), np.uint8),iterations = 5)
            vec = np.ravel(erosion[np.nonzero(erosion)])
            
# find most common grayscale values in vector
            mc = Counter(vec).most_common(9)
            g = [item[0] for item in mc]
            med = np.median(g)
            
# correct picture by subtraction or addition from reference (240)
            img_corr = img - (med-ref)
            print(new_img_name)
            
# save adjusted image
            cv2.imwrite(os.path.join(gray_dir, new_img_name), img_corr)

#%% set detection and phenotyping parameters + make textfile
    
detection_value = 799 # lower = lower sensitivity (e.g. if extremities are too clearly visible)
detection_iterations = 3 # higher = removes more noise from the picture, but also cuts pixels of objects

phenotyping_value = 599
phenotyping_iterations = 3

# add pixels to borders of object
kernel_close = (3,3) # the bigger the more gets added, increase if your isopods are "holey
iterations_close = 3 

 # cut off pixels around borders of object
kernel_open = (7,7)  # the bigger the more gets cut off, increase if legs and antennae need to be trimmed 
iterations_open = 3 
 

# make ouput text files
if not os.path.isfile(os.path.join(out_dir, 'camera.txt')):
    res_file = open(os.path.join(out_dir, 'camera.txt'), 'w')
    res_file.write("Family" + "\t" + "Individual" + "\t" + "File" + "\t" + 'Length' +'\t' + 'Area'+ '\t'+ 'Mean'+ '\t'+  'StdDev'+ '\n')
    res_file.close()
    
#%% phenotyping procedure

for i in os.listdir(gray_dir):
    
    # extract info from filename, family name and individual
    fam = i[0:6]
    ind = i[7:9]

    # pick right scale (70 pixels = 1 mm)
    scale = 70
    
    # read images
    if all([not os.path.isfile(os.path.join(out_dir,"good", i)),
    not os.path.isfile(os.path.join(out_dir,"redone", i)),
    ]):

        if os.path.isfile(os.path.join(gray_dir,"redo", i)):
            img = cv2.imread(os.path.join(gray_dir,"redo", i),0)
            first = False
        else:
            img = cv2.imread(os.path.join(gray_dir, i),0)
            first = True

        ret,thresh_img = cv2.threshold(img,245,0,cv2.THRESH_TOZERO_INV)
        np.place(thresh_img, thresh_img > 0, 255)

        dilation = cv2.dilate(thresh_img,np.ones((9,9),np.uint8),iterations = 3)
        erosion = cv2.erode(dilation,np.ones((51,51), np.uint8),iterations = 10)
        
# =============================================================================
# i) find ROI in image
# =============================================================================
        
        # if first run, create ROI, else take redo-chunk:
        if first == True:
            morph = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,detection_value,detection_iterations)
            morph =  ma.array(data = morph, mask = np.logical_not(erosion)).filled(0)
            morph = cv2.morphologyEx(morph,cv2.MORPH_CLOSE,np.ones((kernel_close),np.uint8), iterations = iterations_close)
            
            morph1 = cv2.morphologyEx(morph,cv2.MORPH_OPEN,np.ones((kernel_open),np.uint8), iterations = iterations_open)
            ret, contours1, hierarchy = cv2.findContours(copy.deepcopy(morph1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
                       
            # create ROI
            areas = []
            if contours1:
                for c in contours1:
                    areas.append(cv2.contourArea(c))
                largest = contours1[np.argmax(areas)]
            else:
                ret, contours, hierarchy = cv2.findContours(copy.deepcopy(morph),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
                for c in contours:
                    areas.append(cv2.contourArea(c))
                largest = contours[np.argmax(areas)]
                
            (x,y),radius = cv2.minEnclosingCircle(largest)
            x = int(x)
            y = int(y)
            q=400
            roi = img[max(0,y-q):y+q,max(0,x-q):x+q]   # img[y-400:y+400, x-400:x+400] 
        else:
            roi = img
            
# =============================================================================
# ii) work with ROI
# =============================================================================
            
        morph2 = cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,phenotyping_value,phenotyping_iterations)
        morph2 = cv2.morphologyEx(morph2,cv2.MORPH_CLOSE,np.ones((kernel_close),np.uint8), iterations = iterations_close)
        morph2 = cv2.morphologyEx(morph2,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_CROSS,kernel_open), iterations = iterations_open)
        ret2, contours2, hierarchy2 = cv2.findContours(copy.deepcopy(morph2),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)    
    
        # continue ONLY make files IF countour exists, otherwise don't add line to text file BUT make image
        if contours2: 
            areas2 = [cv2.contourArea(cnt) for cnt in contours2]  # list of contours in ROI
            largest2 = contours2[np.argmax(areas2)]               # largest contour in ROI
        
        # combine contours from multiple detection procedures
        # conc = np.concatenate((largest1, largest2), axis=0)
            conc = largest2
            
            
            ##### step 3 - create masked array and do algebra on area #####
            mask = np.zeros_like(roi) # Create mask where white is what we want, black otherwise
            mask = cv2.drawContours(mask, [conc], 0, 255, -1) # Draw filled contour in mask
            mask = cv2.erode(mask,np.ones((5,5),np.uint8),iterations = 1)
            masked =  ma.array(data = roi, mask = np.logical_not(mask))
            
        #resized = cv2.resize(img, (0,0), fx=0.3, fy=0.3) 
        #cv2.imshow('Output', resized)
        #cv2.waitKey(0) 
        
            (x,y),radius = cv2.minEnclosingCircle(conc)
            radius = int(radius)
            length = round((radius * 2)/scale,2)
            
            try:
                mean = int(np.mean(masked))
                sd = round(float(np.std(masked)),2)
                area = round((cv2.contourArea(conc)/scale)/scale,2)
            except:
                pass
            
            ##### step 4 - write to files #####
            res_string = fam + "\t" + ind +  "\t"  + i + "\t" + str(length) + "\t" + str(area) + "\t" + str(mean) + "\t" + str(sd) + "\n"

            if i in open(os.path.join(out_dir, 'camera.txt'), "r").read():
                for line in fileinput.input(os.path.join(out_dir, 'camera.txt'), inplace=1, backup='.bak'):
                    if i in line:
                        print(res_string, end="")
                    else:
                        print(line, end="")
            else:
                with open(os.path.join(out_dir, 'camera.txt'), "a") as res_file:
                    print(res_string, end="", file = res_file)
            
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            roi = cv2.circle(roi,(int(x),int(y)), radius,(255,0,0),2)
            roi = cv2.drawContours(roi, [largest2], 0, (0,0,255), 2)  
        cv2.imwrite(os.path.join(out_dir, i), roi)  
        print(i)

    
#%% redo bad detections 
        
#        
## crop images and redo (select roi manually)
#timepoint_list = ["t0", "t1", "t2", "t3", "t4"] 
#
#for timepoint in timepoint_list:
#    
#    in_dir = "E:\\Python_wd1\\2017_Asellus_plasticity\\grayscale\\" + timepoint 
#    bad = "E:\\Python_wd1\\2017_Asellus_plasticity\\processed\\" + timepoint + "\\bad\\"
#    redo = "E:\\Python_wd1\\2017_Asellus_plasticity\\grayscale\\" + timepoint + "\\redo\\"
#    
#    if not os.path.exists(redo):
#        os.makedirs(redo)
#        
#    for file in os.listdir(bad):
#        if os.path.isfile(os.path.join(in_dir, file)) and not os.path.isfile(os.path.join(redo, file)):
#            img = cv2.imread(os.path.join(in_dir, file), 0)
#            print(file)
#            factor = 2
#            resized = cv2.resize(img, (0,0), fx=1/factor, fy=1/factor) 
#            rect = cv2.selectROI(resized)
#            cropped = img[int(rect[1]*factor):(int(rect[1])+int(rect[3]))*factor,int(rect[0]*factor):(int(rect[0])+int(rect[2]))*factor]
#            cv2.imwrite(os.path.join(redo, file), cropped)  
#
## open paint from loop
#timepoint_list = [ "t2", "t3", "t4"]
#
#for timepoint in timepoint_list:     
#    
#    processed = "E:\\Python_wd1\\2017_Asellus_plasticity\\processed\\" + timepoint 
#    redo = "E:\\Python_wd1\\2017_Asellus_plasticity\\grayscale\\" + timepoint + "\\redo\\"
#
#    for file in os.listdir(processed):
#        if os.path.isfile(os.path.join(processed, file)):
#           os.system(redo + file)
#           
#
#

