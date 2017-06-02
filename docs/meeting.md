# First meeting with Dr. Fabian Scalzo (4/14/17)

We met with our professor, Dr. Fabian Scalzo, to get an overview of the project requirements
and to get some advice on how to start. 

## Methods and Procedure

We talked about what methods we could use to perform skull stripping. In particular, we 
decided on first using a medical imaging software to manually create the ground truth
bitmasks on images of patients' skulls and brains. Professor Scalzo recommended us a few
programs we could use, including Horus, 3D slicer, 3DSkullStrip, and OsiriX. We tried
a few of them out and decided on OsiriX for its ease of use. 

We planned to use machine learning methods such as neural networks in order
to train a model using manually extracted data. To make the project slightly more complex
and interesting, we also decided to segment out ventricles in the brain as well.

## OsiriX Viewer

![OsiriX Viewer Logo](https://upload.wikimedia.org/wikibooks/en/0/0b/OsiriX_Logo.jpg "OsiriX Viewer Logo")

Professor Scalzo then showed us how to use the OsiriX software, which is one of the leading
DICOM medical image viewers. He explained how to grow ROIs and set the pixel values in order
to create the ground truth bitmasks. There were several algorithms available for selecting
ROIs on the software, so we'll be experimenting with them and the parameter values in order
to produce the best results.

After the meeting, we got in contact through email with our professor and several other
medical imaging researchers at UCLA in order to collect the data we needed for our project.
