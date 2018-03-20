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
import shutil
import re
import fileinput


from PIL import Image
from collections import Counter

def get_date_taken(path):                           # extracts dates from pictures
    return Image.open(path)._getexif()[36867]


ref = 240 # set reference value, depending on your brightness (fairly arbitrarily, between 180 and 248)

raw_dir = "E:\\Documents_PHD\\1_Asellus\\2017_Asellus_plasticity\\2017_plasticity_data\\phase1\\"
main_dir = "E:\\Python_wd1\\2017_Asellus_Plasticity\\"



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


# %% import packages

# step 1 - fix filenames
#  -------------------------------------------------------------------------- #
timepoint_list = ["t0", "t1", "t2", "t3", "t4"] #, 

file_list = open((main_dir + 'file_list_plasticity_experiment.txt'), 'w')
file_list.write("Filename" + "\n")

for timepoint in timepoint_list:
    
    in_dir = "E:\\Documents_PHD\\1_Asellus\\2017_Asellus_plasticity\\2017_plasticity_data\\phase1\\" + timepoint
    out_dir = "E:\\R\\2017_Asellus_plasticity\\"
    
    for dirpath, subdir, files in os.walk(in_dir):
        for i in (files):
            if os.path.splitext(i)[1]==".jpg" or os.path.splitext(i)[1]==".JPG":
                    fam = "fam" + re.findall('\d+', i)[0]
                    ind = re.findall('\d+', i)[1][0:2]
                    timepoint = timepoint
                    if not ind == "00":
                        file_list.write(i + "\n") # fam + "\t" + ind + "\t" + timepoint + "\t" + 
                        #new_img_name = fam + "_" + ind + "_" + timepoint + ".jpg"
                        #os.replace(os.path.join(dirpath,i), os.path.join(dirpath,new_img_name))  

# delete old imgs
i="fam_096_56_20170504.jpg"
i ="fam099_00_t0.jpg"
                    
in_dir = "E:\\Documents_PHD\\1_Asellus\\2017_Asellus_plasticity\\2017_plasticity_data\\phase1\\" + "t0\\"   
              
for dirpath, subdir, files in os.walk(in_dir):
    for i in (files):
        if len(i)>20:
            os.remove(os.path.join(dirpath,i))
            
# post analysis - remove date from grayscale
            
            
            
            
# step 2 - scales
#  -------------------------------------------------------------------------- #
raw_dir = "E:\\Documents_PHD\\1_Asellus\\2017_Asellus_plasticity\\2017_plasticity_data\\phase1\\"
main_dir = "E:\\Python_wd1\\2017_Asellus_Plasticity\\"
scale_dir = os.path.join(main_dir, "scale")

# copy files over - PHASE 1 ONLY
for dirpath, subdir, files in os.walk(raw_dir):
    for i in files:
        if i.endswith(".jpg") and i[7:9] == "00":
            shutil.copy(os.path.join(dirpath,i), os.path.join(main_dir, "scale", i))

# attach date to img name
            
for i in os.listdir(scale_dir):
    if os.path.isfile(os.path.join(scale_dir, i)):
        date = get_date_taken(os.path.join(scale_dir, i))
        date = date[0:4] + date[5:7] + date[8:10]
        new_file_name = i.replace(".jpg", "_" + date + ".jpg")
        os.rename(os.path.join(scale_dir, i), os.path.join(scale_dir, new_file_name))
            
# measure with imageJ, and write pixel counter for 10 mm to filename
            
# make scale per date 

scale_list = open(os.path.join(main_dir, 'scalelist.txt'), 'w')
#scale_list.write("Date"  + "\t" + "Scale_px/mm" + "\n" )

for i in os.listdir(scale_dir):
    if os.path.isfile(os.path.join(scale_dir, i)):
        date = get_date_taken(os.path.join(scale_dir, i))
        scale_date = date[0:4] + date[5:7] + date[8:10]
        scale = str(i[[m.start() for m in re.finditer(r"_",i)][3]+1:[m.start() for m in re.finditer(r"\.",i)][0]])
        scale = str(int(int(scale)/10))
        scale_list.write(scale_date  + "_" + scale + "\n" )
scale_list.close()



# step 3 - grayscale
#  -------------------------------------------------------------------------- #
ref = 240 # set reference value, depending on your brightness (fairly arbitrarily, between 180 and 248)

raw_dir = "E:\\Documents_PHD\\1_Asellus\\2017_Asellus_plasticity\\2017_plasticity_data\\phase1\\"
main_dir = "E:\\Python_wd1\\2017_Asellus_Plasticity\\"

timepoint_list = ["t0","t1", "t2","t3","t4"]

for timepoint in timepoint_list:

    sub_dir = "E:\\Python_wd1\\2017_Asellus_plasticity\\" + timepoint + "\\"    
    raw_dir =  os.path.join("E:\\Documents_PHD\\1_Asellus\\2017_Asellus_plasticity\\2017_plasticity_data\\phase1\\", timepoint)
    
    out_dir = os.path.join(main_dir, "grayscale", timepoint)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    for dirpath, subdir, files in os.walk(raw_dir):
        for i in files:
            if i.endswith(".jpg") and not i[7:9] == "00":
            
                label = os.path.splitext(i)[0]
                #get fam + date info
                date = get_date_taken(os.path.join(dirpath, i))
                date = date[0:4] + date[5:7] + date[8:10]
                
                new_img_name = label + "_" + date + ".jpg"
                
                # read img
                if not os.path.isfile(os.path.join(out_dir, new_img_name)):
                    img = cv2.imread(os.path.join(dirpath, i),0)
                    #img = cv2.imread(j,0)

                    # reduce resolution by 50%
                    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
                    
                    #threshold to find border
                    if timepoint=="t0":
                        vec = np.ravel(img)
                    else:
                        ret,thresh_img = cv2.threshold(img,245,0,cv2.THRESH_TOZERO_INV)
                        erosion = cv2.erode(thresh_img,np.ones((7,7), np.uint8),iterations = 5)
    
                        # most common 
                        vec = np.ravel(erosion[np.nonzero(erosion)])

                    mc = Counter(vec).most_common(9)
                    g = [item[0] for item in mc]
                    med = np.median(g)
                    
                    # .... but then correct the WHOLE picture
                    img_corr = img - (med-ref)
                    
                    print(new_img_name)
                    
                    cv2.imwrite(os.path.join(out_dir, new_img_name), img_corr)


resized = cv2.resize(thresh_img, (0,0), fx=0.25, fy=0.25) 
cv2.imshow('Output', resized)
cv2.waitKey(0)


# step4 - recognition
#  -------------------------------------------------------------------------- #
main_dir = "E:\\Python_wd1\\2017_Asellus_Plasticity\\"
txt_dir = os.path.join(main_dir, "output")

# make dirs by timepoint
timepoint_list = ["t0", "t1",  "t2","t3","t4"]
for timepoint in timepoint_list:
  
    in_dir = "E:\\Python_wd1\\2017_Asellus_Plasticity\\grayscale\\" + timepoint + "\\"
    out_dir = "E:\\Python_wd1\\2017_Asellus_Plasticity\\processed\\" + timepoint + "\\"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # make ouput text files
    if not os.path.isfile(txt_dir + '\\output_'+ timepoint + '.txt'):
        res_file = open((txt_dir + '\\output_'+ timepoint + '.txt'), 'w')
        res_file.write("Family" + "\t" + "Individual" + "\t" + "Timepoint" + "\t" + "Date" + "\t" + "File" + "\t" + 
                       'Length' +'\t' + 'Area'+ '\t'+ 'Mean'+ '\t'+  'StdDev'+ '\n')
        res_file.close()
    
    
# detection parameters ------------------------------------------------------ #
    detection_value = 799 # lower = lower sensitivity (e.g. if extremities are too clearly visible)
    detection_iterations = 3 # higher = removes more noise from the picture, but also cuts pixels of objects
    
    phenotyping_value = 599
    phenotyping_iterations = 3
    
    # add pixels to borders of object
    kernel_close = (3,3) # the bigger the more gets added, increase if your isopods are "holey
    iterations_close = 3 
    
     # cut off pixels around borders of object
    kernel_open = (7,7)  # the bigger the more gets cut off, increase if your isopods' extremeties need to be trimmed 
    iterations_open = 3 
    
#  -------------------------------------------------------------------------- #
    for i in os.listdir(in_dir):
        if os.path.isfile(in_dir + i):
            
            # extract info from filename
            fam = i[0:6]
            ind = i[7:9]
            time = i[10:12]
            date = i[13:21]
            
            # pick right scale 
            scale_list = open((os.path.join(main_dir, "scale\\scalelist.txt")), 'r')
            for scale_line in scale_list.readlines():
                if date in scale_line:
                    scale = scale_line[9:12]
            scale = int(int(scale)/2)
            scale_list.close()
            
            # read images
            if all([not os.path.isfile(os.path.join(out_dir,"good", i)),
            not os.path.isfile(os.path.join(out_dir,"redone", i)),
            # not os.path.isfile(os.path.join(out_dir, i)),
            not ind == "00",
            ]):

                if os.path.isfile(os.path.join(in_dir,"redo", i)):
                    img = cv2.imread(os.path.join(in_dir,"redo", i),0)
                    first = False
                else:
                    img = cv2.imread(os.path.join(in_dir, i),0)
                    first = True
    
                ret,thresh_img = cv2.threshold(img,245,0,cv2.THRESH_TOZERO_INV)
                np.place(thresh_img, thresh_img > 0, 255)

                dilation = cv2.dilate(thresh_img,np.ones((9,9),np.uint8),iterations = 3)
                erosion = cv2.erode(dilation,np.ones((51,51), np.uint8),iterations = 10)
                
                ##### RECOGNITION #####
                # i) if first run, create ROI, else take redo-chunk:
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
                    
                # ii) actual phenotypinmg
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
                    res_string = fam + "\t" + ind +  "\t"  + time + "\t" + date + "\t" + i + "\t" + str(length) + "\t" + str(area) + "\t" + str(mean) + "\t" + str(sd) + "\n"

                    if i in open(txt_dir + '\\output_'+ timepoint + '.txt', "r").read():
                        for line in fileinput.input(txt_dir + '\\output_'+ timepoint + '.txt', inplace=1, backup='.bak'):
                            if i in line:
                                print(res_string, end="")
                            else:
                                print(line, end="")
                    else:
                        with open(txt_dir + '\\output_'+ timepoint + '.txt', "a") as res_file:
                            print(res_string, end="", file = res_file)
                    
                    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                    roi = cv2.circle(roi,(int(x),int(y)), radius,(255,0,0),2)
                    roi = cv2.drawContours(roi, [largest2], 0, (0,0,255), 2)  
                cv2.imwrite(os.path.join(out_dir, i), roi)  
                print(i)

# redo bad detections #       
#  -------------------------------------------------------------------------- #
            
# crop images and redo (select roi manually)
timepoint_list = ["t0", "t1", "t2", "t3", "t4"] 

for timepoint in timepoint_list:
    
    in_dir = "E:\\Python_wd1\\2017_Asellus_plasticity\\grayscale\\" + timepoint 
    bad = "E:\\Python_wd1\\2017_Asellus_plasticity\\processed\\" + timepoint + "\\bad\\"
    redo = "E:\\Python_wd1\\2017_Asellus_plasticity\\grayscale\\" + timepoint + "\\redo\\"
    
    if not os.path.exists(redo):
        os.makedirs(redo)
        
    for file in os.listdir(bad):
        if os.path.isfile(os.path.join(in_dir, file)) and not os.path.isfile(os.path.join(redo, file)):
            img = cv2.imread(os.path.join(in_dir, file), 0)
            print(file)
            factor = 2
            resized = cv2.resize(img, (0,0), fx=1/factor, fy=1/factor) 
            rect = cv2.selectROI(resized)
            cropped = img[int(rect[1]*factor):(int(rect[1])+int(rect[3]))*factor,int(rect[0]*factor):(int(rect[0])+int(rect[2]))*factor]
            cv2.imwrite(os.path.join(redo, file), cropped)  

# open paint from loop
timepoint_list = [ "t2", "t3", "t4"]

for timepoint in timepoint_list:     
    
    processed = "E:\\Python_wd1\\2017_Asellus_plasticity\\processed\\" + timepoint 
    redo = "E:\\Python_wd1\\2017_Asellus_plasticity\\grayscale\\" + timepoint + "\\redo\\"

    for file in os.listdir(processed):
        if os.path.isfile(os.path.join(processed, file)):
           os.system(redo + file)
           






#cv2.imshow('Output', resized)

