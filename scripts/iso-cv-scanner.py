# -*- coding: utf-8 -*-
"""
Created: 2016/03/31
Last Update: 2018/03/20
Version 1.2
@author: Moritz Lürig
"""

# %% import packages

import cv2
import os
import numpy as np
import numpy.ma as ma
import copy
import math

#%% set directories

# make this your directory
os.chdir("E:\Python1\\iso-cv")

# in_dir should contain the images to analyze, out_dir is for the output that is generated (control images and text files)
in_dir = "python\\in"
out_dir = "python\\out"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#%% procedure
    
# all steps are repeated for each image
for i in os.listdir(in_dir):
    if os.path.isfile(os.path.join(in_dir, i)):
        
# make separate text-file for each image, write to out_dir
        res_file = open(os.path.join(out_dir + i[0:len(i)-4] + '.txt'), 'w')
        res_file.write('PyLabel' + '\t' + 'X' + '\t'+  'Y'+ '\t'+  'Length'+ '\t'+ 'Area'+ '\t'+ 'Mean'+ '\t'+  'StdDev'+ '\t'+  'Min'+ '\t'+  'Max' + '\n')
        res_file.close()
        
# load image, convert to grayscale
        img = cv2.imread(os.path.join(in_dir, i))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
# =============================================================================
# i) find ROIs in image
# =============================================================================
        
# tresholding image
        
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,299,9)
    

# cleanup - "closing operation" with rectangle-shaped kernel, "opening operation" with cross-shaped kernel (good for removing legs) - refer to https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
        morph1 = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel1, iterations = 3)
        morph2 = cv2.morphologyEx(morph1,cv2.MORPH_OPEN,kernel2, iterations = 5)

# find contours 
        ret, contours, hierarchy = cv2.findContours(copy.deepcopy(morph2),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)

        
#        vis = cv2.resize(morph2, (0,0), fx=0.2, fy=0.2)
#        cv2.imshow('image',vis)
#        cv2.waitKey(0)
#        cv2.imwrite("morph2.jpg", morph2)   
#        
# loop through all contours in image
        idx = 0
        #rows = list()
        for cnt in contours:
            if len(cnt) > 50:
                rx,ry,w,h = cv2.boundingRect(cnt)
                ellipse = cv2.fitEllipse(cnt)
                center,axes,orientation = ellipse 
                L = np.mean([math.sqrt(axes[1]*axes[0]*math.pi),max(axes)])
                if L > 190 and h > 150 and w < 1000:
                    idx += 1
                    #ROI
                    roi=gray[max(0,ry-100):ry+h+100,max(0,rx-100):rx+w+100]
                    
                    
                    
                    
# =============================================================================
# ii) thresholding on ROI
# =============================================================================
                    
                    ret, roi_thresh = cv2.threshold(roi,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                    # actual cleanup: define kernel size based on shape size, do OPENING
                    if L > 600:
                        k = 3; niter = int(round(L * 0.007) -3)   #+ 1.5
                    else:
                        k = 3; niter = int(round(L * 0.025) - 3)       
                    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
                    opening1 = cv2.morphologyEx(roi_thresh,cv2.MORPH_CLOSE,kernel2, iterations = niter)
                    kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
                    opening1 = cv2.morphologyEx(opening1,cv2.MORPH_OPEN,kernel3, iterations = niter+3)
                    #create contour, centroid, and min. circle diameter
                    ret1, contours1, hierarchy1 = cv2.findContours(copy.deepcopy(opening1),cv2.RETR_LIST ,cv2.CHAIN_APPROX_TC89_L1)       
                    if contours1:
                        areas = [cv2.contourArea(cnt1) for cnt1 in contours1]                
                        shape = contours1[np.argmax(areas)]
                        (cx,cy),radius = cv2.minEnclosingCircle(shape)
                    #shape masking for measurements
                        mask = np.zeros_like(opening1) # Create mask where white is what we want, black otherwise
                        mask = cv2.drawContours(mask, contours1, np.argmax(areas), 255, -1) # Draw filled contour in mask
                        mask = cv2.erode(mask,np.ones((5,5),np.uint8),iterations = 1)
                        masked =  ma.array(data=roi, mask = np.logical_not(mask))
                    
                    #calculate all kinds of things and write to file
                    M = cv2.moments(shape)
                    a = int((M['m10']/M['m00'])+max(0,rx-100))
                    b = int((M['m01']/M['m00'])+max(0,ry-100))
                    c = (int(radius) * 2)/94.6876
                    d = np.mean(masked)
                    e = np.std(masked)
                    f = np.min(masked)
                    g = np.max(masked)
                    area = (cv2.contourArea(shape)/94.6876)/94.6876
                    
                    
                    res_file = open(os.path.join(out_dir  + i[0:len(i)-4] + '.txt'), 'a')
                    res_file.write(str(idx) + '\t' + str(round(a,2)) + '\t' + str(round(b,2)) + "\t" + str(round(c,2)) + "\t" + str(round(area,2)) + "\t" +str(round(d,2)) + "\t" + str(round(e,2)) + "\t" + str(round(f,2)) + "\t" +  str(round(g,2))+ "\n")
                    res_file.close()
                    
                    
                    # draw bounding rectangle, contour and circle into original image to check if they are correct
                    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                    #roi = cv2.circle(roi,(int(cx),int(cy)),int(radius),(255,0,0),2)
                    img[max(0,ry-100):ry+h+100,max(0,rx-100):rx+w+100] = roi
                    
#                    img = cv2.drawContours(img, [cnt], 0, (0,255,0), 2)   
#                    img = cv2.rectangle(img,(rx,ry),(rx+w,ry+h),(255,0, 0),2)
                    img = cv2.rectangle(img,(max(0,rx-100),max(0,ry-100)),(rx+w+100,ry+h+100),(0,0, 255),5)


                    #img = cv2.ellipse(img ,ellipse,(0,0,255),2)
                    #rows.append((a,b))
                    #cv2.putText(img, str(idx),(a,b), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),3,cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir , i[0:len(i)-4] + '_' + 'output.jpg'), img)   
        print(i)

        cv2.imwrite(os.path.join(out_dir , i[0:len(i)-4] + '_' + 'roi.jpg'), roi)   

    


