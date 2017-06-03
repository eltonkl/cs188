# Training the machine learning model

Now it was time to finally start coding up our machine learning model. We decided to train a model that
takes a n x n square grid of pixel values, where n is odd, and predict whether the center pixel is 
inside the skull or not. To do this, we have to first read in the training images and convert each 
into sets of n x n pixels.

## Preprocessing the images

Our first step was to read in all of our training data. We looped through all files
in the input directory that started with 'IM', and used a Python library called
pydicom to read in the DICOM images.

```python
# Iterate through all the files in the given path
for file_path in os.listdir(dir_path):

    # Ensure only files that start with IM are processed
    if not file_path.startswith('IM-'):
        continue

    # File paths for training data set (2 images: FLAIR skull MRI and associate bitmask)
    train_file = dir_path + file_path
    result_file = dir_path + groundtruth_filepath(file_path)

    # Read training images
    try:
        # Input FLAIR MRI dicom file
        input_dc = dicom.read_file(train_file)
        input_dc = preprocess_image(input_dc)
```

After reading in the DICOM images with pydicom, we preprocess each image:

```python
##
# Preprocessor for image
#
# dicom_file: [pydicom object] Image which should be preprocessed
#
# Note: This could be adapted to downscale the image, reduce noise in it, etc
#       We found that just scaling yielded the best results. Noise reduction and
#       downscaling decreased processing time but worsened output significantly
##
def preprocess_image(dicom_file):
	# We take our image, center to the mean and scale it to a unit variance
	return preprocessing.scale(dicom_file.pixel_array)
```

This centers the image to the mean and scale it to a unit variance, which allows
us to standardize the training images and produce better results. The pixel array
is what we'll be dealing with in the code, as it contains the 256-bit greyscale
pixel values of the image.

Next step is to read in the ground truth bitmasks we created using OsiriX earlier:

```python
# Ground truth file for the MRI file
gt_dc = dicom.read_file(result_file)
gt_dc = gt_dc.pixel_array
```

We now have all DICOM images read in and stored in memory. The next step is to break
each image down into a series of n x n grids, where the value of n was varied to give
the best results:

```python
# Iterate through all pixels (excluding MARGINs)
for y in range (MARGIN, height - MARGIN):
    for x in range (MARGIN, width - MARGIN):

        # Useful if we want to decrease the amount of training data
        # rand = random.random()
        # if rand < .60 or (input_dc[y, x] < .0 and rand < .80):
        # 	continue;

        # For each pixel gather its pixel value and surrounding and pixel values
        temp_trainer_data = []

        # For each pixel gather its pixel value and surrounding and pixel values
        # as well as its location relative to the center of the image
        temp_trainer_data.append(x - width/2)
        temp_trainer_data.append(y - height/2)

        for pix_x in range (x - MARGIN, x + MARGIN + 1):
            for pix_y in range (y - MARGIN, y + MARGIN + 1):
                temp_trainer_data.append(input_dc[pix_y, pix_x])

        # Add pixel data to our training set
        pixel_data.append(temp_trainer_data)
```

We went through all pixels in each image, excluding the margins, and for each pixel
we stored the n x n grid that has that pixel as the center. We also looked to our
generated bitmasks to store what the actual value of this pixel should be - 1 for
inside the brain, and 0 for outside:

```python
if gt_dc[y, x] > 1:
    pixel_result.append(1)
else:
    pixel_result.append(0)
```

Now that we are done preprocessing the image, we can return the pixel data and 
pixel results to be used in training the machine learning model:

```python
# Return the training set
	return (pixel_data, pixel_result)
```

## Training the model

At first, we tried using the Tensorflow library and train a neural network on the
pixel data. However, this turned out to be cumbersome and inaccurate, so we decide
to use the scikit-learn library instead. After training the model on the data,
we stored the generated classifier to an output file for later use. This way, we
would not have to retrain the model every time we wanted to use it to perform skull
stripping:

```python
##
# Trains the classifier given a set of training data, dumps the generated classifier
# to an output file for later use
#
# output_file_name: [string] Output filename of the generated classifier
#
# USES GLOBAL VARIABLE (due to python memory sharing constraints):
# pixel_data: [2D float array] Each array entry contains the pixel values (BLOCK_SIZE^2) that surround
#             a pixel in the image and the location of the pixel relative to the center of the image
# pixel_results: [1D binary array] Array regarding whether or not each pixel entry from pixel_data array
#                is part of the brain (value = 1) or not (value = 0)
##
def train_model(output_file_name):
	# Create the classifier and fit the data to it
	my_classifier = get_classifier()
	my_classifier.fit(pixel_data, pixel_result)

	# Dump the generated classifier to an output file for later use
	joblib.dump(my_classifier, output_file_name)
```

For the classifier itself, we tried numerous different types of classifiers,
including nearest neighbors, SVM, and decision trees. However, our best accuracy
came from the random forest classifier:

```python
##
# Configure and return a classifier for our data
#
# We found that the Random Forest Classifier gave us the best results (in a reasonable
# amount of time)
##
def get_classifier():
	# n_jobs = -1 allows us to use all cores of our machine
	return RandomForestClassifier(n_jobs = -1)
```

Note that we are multithreading the training process to speed it up.

## Testing the trained model

To evaluate which classifier worked the best and for us to get an idea of the
accuracy of our methods, we split the training set into a training set and a
testing set. After training the model on the training set, we tested its accuracy
using the testing set:

```python
##
# Given a set of data splits it into (1 - test_size) of training data and test_size of testing data.
# Trains the classifier using the training data and provides the accuracy of the classifier using
# the testing data
#
# test_size: [float] Percent of data that should be used
#
# USES GLOBAL VARIABLE (due to python memory sharing constraints):
# pixel_data: [2D float array] Each array entry contains the pixel values (BLOCK_SIZE^2) that surround
#             a pixel in the image and the location of the pixel relative to the center of the image
# pixel_results: [1D binary array] Array regarding whether or not each pixel entry from pixel_data array
#                is part of the brain (value = 1) or not (value = 0)
##
def test_model(test_size):
	# Split the data into training and testing data
	X_train, X_test, y_train, y_test = train_test_split(pixel_data, pixel_result, test_size=test_size)

	# Create the classifier and trains it on the the training data
	my_classifier = get_classifier()
	my_classifier.fit(X_train, y_train)

	# Tests the classifier on the testing data and prints the accuracy results
	predictions = my_classifier.predict(X_test)
	print(accuracy_score(y_test, predictions))
```

Our model using the random forest classifier consistently gave us around 96% to 97%
accuracy, which is fairly good. We also determined that the best block size was 11.