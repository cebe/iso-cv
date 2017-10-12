# -*- coding: utf-8 -*-
"""
Spyder Editor
Moritz Lürig
28.09.2017

"""

import cv2
import os
import copy
import numpy as np
import numpy.ma as ma
from PIL import Image
from collections import Counter
    
# instructions  -----------------------------------------------------------------------------------------------------------
#
# - you need an empty main folder to work from. for simplicity I will use "C:/Python/Asellus"
# - inside this folder should be another folder that contains the raw images to be processed. lets call it "raw".
# - this script will generate two more folder inside the main folder ("grayscale" and "processed"), if they not already exist.
# - "grayscale will have pictures with adjusted brightness (equalized histograms), "processed" will contain the phenotyped images. 
# - the data will be contained in a txt-file in the processed folder.

# prep before running  -----------------------------------------------------------------------------------------------------------

# 0) make sure all the packages are installed (best via conda or pip from powershell)
# 1) make sure the folder structure is correct 
# 2) put the raw images in the "raw" folder
# 3) measure scale (e.g. with millimeter paper), e.g. in image J. this is, how many pixels are in one millimeter?
# 4) run part one of the script - the grayscale folder should now contain compressed and brightness adjusted images
# 5) run part two of the script - this will take a while, you can see how the "processed" folder fills up
# 6) if you are not happy with your results change some detection parameters (Kernel size and number of iterations),
#    and run ONLY part two again, it will overwrite the previous attempt 

# some parameters   -----------------------------------------------------------------------------------------------------------

main = "E:\\Python_wd1\\UV1\\"
scale = 70 #scale: x pixel = 1 mm
ref = 240 # set reference value, depending on your brightness (fairly arbitrarily, between 180 and 248)

def get_date_taken(path):                           # extracts dates from pictures
    return Image.open(path)._getexif()[36867]



# detectionb parameters IMPORTANT -----------------------------------------------------------------------------------------------------------

detection_value = 799 # lower = lower sensitivity (e.g. if extremities are too clearly visible)
detection_iterations = 4 # higher = removes more noise from the picture, but also cuts pixels of objects

# kernel: only odd numbers
# iteration how often procedure for each operation is performed

# add pixels to borders of object
kernel_close = (3,3) # the bigger the more gets added, increase if your isopods are "holey"
iterations_close = 3 

 # cut off pixels around borders of object
kernel_open = (7,7)  # the bigger the more gets cut off, increase if your isopods' extremeties need to be trimmed 
iterations_open = 3 



# part 1   -----------------------------------------------------------------------------------------------------------
# make folders
in_dir = os.path.join(main + "raw\\pic4\\")
out_dir = os.path.join(main + "grayscale\\")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
for h in os.listdir(in_dir):
    fam = h
    subdir = in_dir + h + "\\"
    
    for i in os.listdir(subdir):
       img = cv2.imread(subdir + i,0)
       label = i[0:len(i)-4]
       
       #get fam + date info
       date = get_date_taken(subdir + i)
       date = date[0:4] + date[5:7] + date[8:10]
       
       # reduce resolution by 50%
       img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
       
       #IMPORTANT: take reference area from the middle of the image where the isopod should be ....
       roi = img[int(img.shape[1]*0.2):int(img.shape[1]*0.8),int(img.shape[1]*0.25):int(img.shape[1]*0.75)] # starting from pic1 I think
       vec = np.ravel(roi)
       mc = Counter(vec).most_common(9)
       g = [item[0] for item in mc]
       med = np.median(g)
       
       # .... but then correct the WHOLE picture
       img_corr = img - (med-ref)
       new_img_name = label + "_" + date + ".jpg"
       print(new_img_name)
       
       cv2.imwrite(os.path.join(out_dir + new_img_name), img_corr)   
    
               
# part 2   -----------------------------------------------------------------------------------------------------------
# make folders
in_dir = os.path.join(main + "grayscale\\")
out_dir = os.path.join(main + "processed\\")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# make ouput text file
if not os.path.isfile(out_dir + 'output.txt'):
    res_file = open((out_dir + 'output.txt'), 'w')
    res_file.write("Label"  "\t" + "Source_file" + "\t" + 'Length' +'\t' + 'Area'+ '\t'+ 'Mean'+ '\t'+  'StdDev'+ '\n')
    res_file.close()

# detection 
for i in os.listdir(in_dir):
    # loop through images, extract info from filename
    label = i[0:len(i)-13]
    date = i[len(i)-12:len(i)-4]

    # read images
    img = cv2.imread(in_dir + i,0)
    
    # cut off edges
    img = img[int(img.shape[1]*0.1):int(img.shape[1]*0.9),int(img.shape[1]*0.1):int(img.shape[1]*0.9)]


    ##### step 1 - RECOGNITION - find largest object and create ROI ####
    morph = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,999,4)
    morph = cv2.morphologyEx(morph,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8), iterations = 1)
    morph = cv2.morphologyEx(morph,cv2.MORPH_OPEN,np.ones((3,3),np.uint8), iterations = 3)
    ret, contours, hierarchy = cv2.findContours(copy.deepcopy(morph),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
               
    # create ROI
    if contours: 
       areas = [cv2.contourArea(cnt) for cnt in contours]
       largest = contours[np.argmax(areas)]
       (x,y),radius = cv2.minEnclosingCircle(largest)
       x = int(x)
       y = int(y)
       q=400
       roi = img[max(0,y-q):y+q,max(0,x-q):x+q]   # img[y-400:y+400, x-400:x+400] 
    else:
       roi = img


    ##### step 2 - PHENOTYPING - extract object from ROI and perform thresholding/morphology #####
    # Adaptive gaussian 
    morph2 = cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,detection_value,detection_iterations)
    morph2 = cv2.morphologyEx(morph2,cv2.MORPH_CLOSE,np.ones((kernel_close),np.uint8), iterations = iterations_close)
    morph2 = cv2.morphologyEx(morph2,cv2.MORPH_OPEN,np.ones((kernel_open),np.uint8), iterations = iterations_open)
    ret2, contours2, hierarchy2 = cv2.findContours(copy.deepcopy(morph2),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)    

#resized = cv2.resize(morph2, (0,0), fx=0.5, fy=0.5) 
#cv2.imshow('Output', resized)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

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
        
        (x,y),radius = cv2.minEnclosingCircle(conc)
        radius = int(radius)
        length = round((radius * 2)/scale,2)
        try:
            mean = int(np.mean(masked))
            sd = int(np.std(masked))
            area = round((cv2.contourArea(conc)/scale)/scale,2)
        except:
            pass
        
        
        ##### step 4 - write to files #####
        res_file = open((out_dir + 'output.txt'), 'a')
        res_file.write(label + "\t" + date +   "\t" 
                       + str(length) + "\t" + str(area) + "\t" + str(mean) + "\t" + str(sd) + "\n")
        res_file.close()
        
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        roi = cv2.circle(roi,(int(x),int(y)), radius,(255,0,0),2)
        #roi = cv2.drawContours(roi, [conc], 0, (0,255,0), 2)   
        #roi = cv2.drawContours(roi, [largest1], 0, (0, 255,0), 2)  
        roi = cv2.drawContours(roi, [largest2], 0, (0,0,255), 2)  
        #cv2.putText(img, str(date + '_' + fam + '_' + i[0:len(i)-4]),(346,519), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
        #roi = roi[200:600,200:600]
    print(i)
    cv2.imwrite(os.path.join(out_dir, i), roi)   
