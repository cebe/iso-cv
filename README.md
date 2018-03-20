<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [iso-cv](#iso-cv)
  - [two types of scripts](#two-types-of-scripts)
  - [working principle](#working-principle)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# iso-cv
**mass-phenotyping of isopods using the opencv library** 

The intent of this document is to explain the general workflow for "high-throughput" phenotyping of the freshwater isopod *Asellus aquaticus* and of the working principles in each step. For the scientific background refer to http://luerig.net/Research/#Isopods. For more detailed information on the actual code refer to the inline annotations inside the python scripts.


<img src=https://mluerig.github.io/iso-cv/images/iso-cv-fig1.png width=70%>

*This document aims at explaining the red arrow: how to extract phenotypic information from digital images*

---

## two types of scripts

There are two types of procedures: one was programmed for use with scanner-images that contain many specimens, the other is intended for camera-images that contain only single isopods:

<img src=http://luerig.net/wp-content/uploads/iso-cv-fig2.png width=70%>

*Left: dead specimens of A. aquaticus that were scanned in a modified flatbed-scanner. Right: alive specimen that was photographed with a camera-stand*
  
Images created with a flatbed scanner have consistent exposure and fixed resolution, so no brightness or scale information needs to be processed prior to phenotyping if the scanner settings are kept the same. For images taken with a camera it is important to ensure reproducible brightness/exposure and distance from the specimens across multiple measurements. Moreover, camera images tend to have more noise and light-artifacts, which may require some treatment to make the scripts work.  

---

## working principle

The scripts are using thresholding algorithms to "separate" the foreground, in our case the isopod, from the background. Depending on the type of image, a different type of algorithm is used (adaptive thresholding for camera images, or Otsu's binarization for scanned isopods [https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html]). From a thresholded image then a binary mask can be created to select the region of interest (ROI) that will be inlcuded for phenotyping. 

<img src=http://luerig.net/wp-content/uploads/iso-cv-fig3.png width=70%>

*From left to right: *
