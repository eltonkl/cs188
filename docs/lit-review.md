# Background Research and Literature Review

Before starting our project, we performed some background research to
understand more about the problem, including reading online articles on skull stripping
to find out what it is used for. We then studied recent research papers written
by experts in the field to better understand the state-of-the-art methods currently
used for skull stripping. We then summarized our research, findings, and project plan in
a [literature review document](public/lit_review.pdf), which is also reproduced below.

# Skull Stripping - Literature Review

## Introduction and Background

Skull stripping, which is one form of brain segmentation, refers to the process of 
isolating the brain from other structures in a medical image of the brain. This usually 
involves the removal of the skull and any other non-brain tissue from the image, 
such as the dura and scalp. It is often done as a pre-processing step as it is improves 
the speed and accuracy of many brain image analysis/processing algorithms, such as 
coregistration and tissue segmentation, while also decreasing algorithm 
complexity (Roy 2015). Skull stripped images have become a standard preliminary 
step for many brain image processing tools. 

Skull stripping is very commonly done on magnetic resonance (MR) images and so we will
be focusing our research on MR images. More specifically, we will be working with axial
views of the brain taken using the fluid-attenuated inversion recovery (FLAIR) pulse 
sequence. The images primarily contain brain tissue surrounded by the scalp, skull, and 
dura tissue.
 
Although manually processing the images is arguably the most accurate approach to skull 
stripping, it is time intensive and does not scale for large projects involving hundreds 
of patients and large number of slices for each patient (Hwang 2011). Our goal is to use 
machine learning to develop a tool that is able to segment the brain tissue in MR images 
with reasonably high accuracy and speed. 

## Current Research and Methods

There are several state-of-the-art methods currently developed for skull stripping. 
The Brain Extraction tool is a fast method that utilizes a sphere in the center of 
gravity and continuously deforms it until it expands to encompass the surface of the 
brain (Kleesiek 2016). Another common method is Hybrid Watershed Algorithm that finds 
the outline of the brain using edge detection, but tends to do poorly when faced with 
images containing brain tumors. A more recent approach is the Robust Learning-Based 
Brain Extraction (ROBEX), where voxels on the brain boundary are detected using random 
forests (Butman 2017).

Many of these techniques mentioned required tuning of several numerical parameters 
depending on the dataset in order to achieve the best results. However, in an research 
paper published in 2016 by Kleesiek et al., they used deep learning architecture in 
order to successfully extract the brain from several different modalities of images 
with minimal (or even no) need for parameter turning. In particular, they trained 3D 
deep convolutional neural networks that can automatically learn the important features 
in the dataset from training, and used them to able to extract the brains from various 
types of scans with impressive accuracy (Kleesiek 2016). Their mean Dice score, which 
is a common measure of accuracy for these sort of tasks, was 95.19, which outperforms 
many of other techniques mentioned above.

## Conclusion and Project Plan

For our project, we will be doing something similar to the approach presented by 
Kleesiek, with some modifications. In particular, we will also be stripping the 
ventricles in the center of the brain from the image. We will be using the OsiriX 
software to manually prepare the training data we will use. By selecting the brain 
sections with the program, we can export a binary mask of the image. Finally, we will 
train our dataset with a machine learning algorithm written using the SciKit-Learn library, and test its accuracy. 