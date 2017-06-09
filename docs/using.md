# Using the program

This page will go through the steps necessary to run the code and reproduce our results.

## Dependency setup

The first step is to install the latest version of Python 3 [here](https://www.python.org/downloads/),
if it is not already installed. Next, there are several python packages that needs to be installed
using python's bundled package manager `pip`. Run the following command on the command line:

```bash
pip install pydicom numpy sklearn scipy
```

This will allow us to read in the DICOM images and use the machine learning functions 
in scikit-learn.

## Running the program

Navigate to the directory containing the skullstrip.py file, and run `python skullstrip.py` to
run the program. This will default to training a machine learning model using training data located
in `./data/train/` and storing the trained classifier as `./classifier.pkl`. To change the default
behavior, here are the possible flags and options that can be passed to the script:

```
optional arguments:
  -h, --help            show this help message and exit
  -t TRAINPATH, --trainpath TRAINPATH
                        path to the training data. defaults to ./data/train/
  -c CLASSIFIER, --classifier CLASSIFIER
                        path to the classifier, if saved trained and saved.
                        path to save the classifier otherwise. defaults to
                        ./classifier.pkl
  -p, --process         process an image by stripping it's skull, instead of
                        training the model. defaults to output the changed
                        image, with the skull stripped.
  -i IMAGEPATH, --imagepath IMAGEPATH
                        path to the images to process. defults to ./data/test/
  -b, --bitmask         stores the result as a bitmask instead of stripping
                        the original image.
  -r RESULTPATH, --resultpath RESULTPATH
                        path to store the results of the processing. defaults
                        to ./data/result
  -m MASK MASK MASK, --mask MASK MASK MASK
                        apply a mask to dicom images [image directory] [mask
                        directory] [output directory]
  -d DICE DICE, --dice DICE DICE
                        compute dice scores [ground truth directory] [result
                        directory]
```

In particular, once a model is trained, it can be used to perform skull stripping with the 
flag `-p`. This defaults to performing skull stripping on the DICOM images in `./data/test/` and
storing the processed images in the directory `./data/result/`. These default directories can be
changed with the `-i IMAGEPATH` and `-r RESULTPATH` options, respectively. The default directory for
the training data is `./data/train/` which can be changed using `-t TRAINPATH`.

## Two-phased approach

To run the two-phased approach described in our previous post, run the shell script `DualPass.sh`.
This defaults to running both the training and processing procedures. Note that this shell script
will only work when the first phase training images are located in `./data/train_phase1/` and
the second phase training images are located in `./data/train_phase2/`.

If `DualPass.sh train` in run instead, only the training step will be done. Similarly,
`DualPass.sh process` will only to the processing and skull stripping part.

Note, the shell script assumes that the system has a `python3` environment variable pointing to
the python executable.