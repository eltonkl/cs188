# Brian Segmentation & Skull Stripping 

<div style="text-align:center"><img src ="http://atc.udg.edu/nic/margaToolbox/images/graph.png" /></div>

## Introduction

Skull stripping, which is one form of brain segmentation, refers to the 
process of isolating the brain from other structures in a medical image 
of the brain. This usually involves the removal of the skull and any other 
non-brain tissue from the image, such as the dura and scalp. It is often 
done as a pre-processing step as it is improves the speed and accuracy of 
many brain image analysis/processing algorithms, such as coregistration and 
tissue segmentation, while also decreasing algorithm complexity (Roy 2015). 
Skull stripped images have become a standard preliminary step for many 
brain image processing tools. 

This website will be updated on the status of our progress on the project.

## Updates

* [4/14/17: First meeting with Dr. Fabian Scalzo](meeting.md)
* [4/28/17: Background research and literature review](lit-review.md) 
* [5/2/17: Acquiring the FLAIR data](data.md)
* [5/13/17: Creating the ground truth bitmasks with OsiriX](bitmasks.md)
* [5/21/17: Preprocessing the images](preprocessing.md)
* [5/29/17: Training the model with scikit-learn](scikit.md)
* [6/1/17: Optimizations, flags, and documentation](optimize.md)

## Results

Our final report can be found here:

[Final Report](report.pdf)