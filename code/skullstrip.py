import os
import random
import dicom
import numpy as np
import argparse

from itertools import repeat
from multiprocessing import Pool

from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from scipy.spatial import distance
from scipy.ndimage.filters import gaussian_filter

# File Path Constants
TRAINING_DATA_PATH = './data/train/'
TESTING_DATA_PATH = './data/test/'
CLASSIFIER_FILE_PATH = './classifier.pkl'
RESULT_DATA_PATH = './data/result/'

# Block size we are training on
MODEL_BLOCK_SIZE = 11
MARGIN = MODEL_BLOCK_SIZE // 2

# Bitmask mode.
BTMASK_MODE = False

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

##
# Possible postprocessor for image
#
# image_data: [2D array] Image which needs to be post processed
#
# Note: Not being used. Results subpar.
##
def neighbor_postprocess(image_data):
	# Compute dimensions of the image
	height = len(image_data)
	width = len(image_data[0])

	neighbor_grid = 2
	grid_size = (2*neighbor_grid + 1) * (2*neighbor_grid + 1)
	threshold = grid_size * 21/25

	# Iterate through each row of pixels (excluding MARGINs)
	for y in range (neighbor_grid, height - neighbor_grid):
		for x in range (neighbor_grid, width - neighbor_grid):
			neighbor_count = 0
			for pix_x in range (x - neighbor_grid, x + neighbor_grid + 1):
				for pix_y in range (y - neighbor_grid, y + neighbor_grid + 1):
					if image_data[pix_y, pix_x] != 0:
						neighbor_count += 1

			if image_data[y, x] == 0:
				if neighbor_count > threshold:
					image_data[y, x] = 1000
			else:
				if grid_size-neighbor_count > threshold:
					image_data[y, x] = 0

	return image_data

##
# Takes an input dicom image and saves a processed copy
#
# input_name: [string] name of image file
# last_four_args: [tuple] contains:
# 	input_dir: [string] directory where image file is be stored
# 	output_dir: [string] directory where output image file will be stored
# 	classifier_path: [string] path to classifier pickle
# 	btmask_mode: [boolean] if true, the output image will be the bitmask for the
#   	            skull stripped image, otherwise it will be the actual skull
#       	        stripped image
##
def process_image(image_name, last_four_args):

	input_dir, output_dir, classifier_path, bitmask_mode = last_four_args

	try:
		print(image_name + " is being processed...")

		try:
			my_classifier = joblib.load(classifier_path)
		except Exception as e:
			print(e)
			print("Trained classifier should be located at: " + classifier_path)
			exit()

		# Read input FLAIR MRI dicom file
		predict_dc_file = dicom.read_file(input_dir + image_name)
		predict_dc = preprocess_image(predict_dc_file)

		# Compute dimensions of the image
		height = len(predict_dc)
		width = len(predict_dc[0])

		# Color top and bottom margins black
		for y in range (-MARGIN,  MARGIN):
			for x in range(0, width):
				predict_dc_file.pixel_array[y, x] = 0

		# Iterate through each row of pixels (excluding MARGINs)
		for y in range (MARGIN, height - MARGIN):

			pixel_data = []

			for x in range (MARGIN, width - MARGIN):

				# For each pixel gather its pixel value and surrounding and pixel values
				# as well as its location relative to the center of the image
				temp_predict_data = []
				temp_predict_data.append(x - width/2)
				temp_predict_data.append(y - height/2)

				for pix_x in range (x - MARGIN, x + MARGIN + 1):
					for pix_y in range (y - MARGIN, y + MARGIN + 1):
						temp_predict_data.append(predict_dc[pix_y, pix_x])

				pixel_data.append(temp_predict_data)

			# Predict output for row
			prediction = my_classifier.predict(pixel_data)

			# Color side margins black
			for x in range (0,  MARGIN):
				predict_dc_file.pixel_array[y, x] = 0
			for x in range (width - MARGIN, width):
				predict_dc_file.pixel_array[y, x] = 0

			# Apply prediction results to image
			for x in range (MARGIN, width - MARGIN):
				if prediction[x - MARGIN] == 0:
					predict_dc_file.pixel_array[y, x] = 0
				elif bitmask_mode:
					predict_dc_file.pixel_array[y, x] = 100000

		# if bitmask_mode:
		# 	predict_dc_file.pixel_array = neighbor_postprocess(predict_dc_file.pixel_array)

		# Save modified image in new file
		predict_dc_file.PixelData = predict_dc_file.pixel_array.tostring()
		predict_dc_file.save_as(output_dir + image_name)

		print(image_name + " completed! File saved to " + output_dir + image_name)

	except Exception as e:
		print("[ERROR] " + image_name + ": ")
		print(e)

##
# Returns the path for the ground truth file given the input file
#
# file_path: [string] input file path (i.e. IM-1234-12345.dcm)
##
def groundtruth_filepath(file_path):
	return 'B' + file_path[1:]

##
# Given a directory, finds all dicom files that start with IM and feeds them in for processing
# Result images are skullstripped images in the format '{input_file_name}.ML_out.dcm'
# Uses multiprocessing to get faster results
#
# dir_path: [string] directory path for input images
##
def batch_process_images(input_dir, output_dir, classifier_path, bitmask_mode):
	file_list = []
	for name in os.listdir(input_dir):
		if name.startswith('IM-'):
			file_list.append(name)

	po = Pool()
	last_four_args = [input_dir, output_dir, classifier_path, bitmask_mode]
	res = po.starmap(process_image, zip(file_list, repeat(last_four_args)))
	po.close()
	po.join()


# Have to use global arrays so that multiple copies are not created while multithreading
pixel_data = []
pixel_result = []

##
# Given a directory, finds all dicom files that start with IM and saves generated training data to global variables
# named pixel_data and pixel_results
# pixel_data is a 2D array in which each array entry contains the pixel values (BLOCK_SIZE^2) that
#             surround a pixel in the image and the location of the pixel relative to the center of the image
# pixel_results is a 1D array with the information regarding whether or not each pixel entry from pixel_data array
#               is part of the brain (value = 1) or not (value = 0)
#
# dir_path: [string] directory path for input images
##
def generate_training_data(dir_path):

	global pixel_data
	global pixel_result

	# Output data for our classifier
	# pixel_data: array of MODEL_BLOCK_SIZE^2 pixel values for each pixel
	# pixel_result: value for each pixel in the groundtruth dicom file
	pixel_data = []
	pixel_result = []

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

			# Ground truth file for the MRI file
			gt_dc = dicom.read_file(result_file)
			gt_dc = gt_dc.pixel_array
		except Exception as e:
			print("[ERROR] " + file_path + ": ")
			print(e)
			continue

		# Dimensions  of the image
		height = len(input_dc)
		width = len(input_dc[0])

		print("Generating training data for " + file_path + " (" + str(width) + ", " + str(height) + ")")

		# Ensure file meets minimum dimension requirements
		if height < MODEL_BLOCK_SIZE or width < MODEL_BLOCK_SIZE:
			exit()

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
				if gt_dc[y, x] > 1:
					pixel_result.append(1)
				else:
					pixel_result.append(0)

	# Return the training set
	return (pixel_data, pixel_result)

##
# Configure and return a classifier for our data
#
# We found that the Random Forest Classifier gave us the best results (in a reasonable
# amount of time)
##
def get_classifier():
	# n_jobs = -1 allows us to use all cores of our machine
	return RandomForestClassifier(n_jobs = -1)

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

##
# Applies masks to all files in the image_dir_path and places the files in output_dir_path
#
# image_dir_path: [string] Directory with MRI images
# mask_dir_path: [string] Directory with skull stripping masks
# output_dir_path: [string] Directory where masked images are stored
##
def apply_mask(image_dir_path, mask_dir_path, output_dir_path):
	file_list = []
	for name in os.listdir(image_dir_path):
		if not name.startswith('IM-'):
			continue

		try:
			# Input FLAIR MRI dicom file
			input_dc_file = dicom.read_file(image_dir_path + name)
			input_dc = input_dc_file.pixel_array

			mask_dc_file = dicom.read_file(mask_dir_path + name)
			mask_dc = mask_dc_file.pixel_array
		except Exception as e:
			print("[ERROR] " + image_dir_path + ": ")
			print(e)
			continue

		# Dimensions  of the image
		height = len(input_dc)
		width = len(input_dc[0])

		print("Masking " + name + " (" + str(width) + ", " + str(height) + ")")

		# Iterate through all pixels
		for y in range (0, height):
			for x in range (0, width):
				if mask_dc[y, x] == 0:
					input_dc[y, x] = 0

		# Save modified image in new file
		input_dc_file.PixelData = input_dc.tostring()
		input_dc_file.save_as(output_dir_path + name)

		print("File saved to " + output_dir_path + name)

##
# Computes the dice score for all images in dir_path_ground and dir_path_test with the
# same file name (and start with IM)
#
# dir_path_ground: [string] Directories with ground truth dicom images
# dir_path_test: [string] Directories with test truth dicom images
##
def compute_dice(dir_path_ground, dir_path_test):
	dice_sum = 0
	dice_count = 0

	file_list = []
	for name in os.listdir(dir_path_ground):
		if not name.startswith('IM-'):
			continue

		try:
			ground_file = dicom.read_file(dir_path_ground + name)
			ground = ground_file.pixel_array

			test_file = dicom.read_file(dir_path_test + name)
			test = test_file.pixel_array
		except Exception as e:
			print("[ERROR]")
			print(e)
			continue

		# Dimensions  of the image
		height = len(ground)
		width = len(ground[0])

		print("Computing dice score for " + name + " (" + str(width) + ", " + str(height) + ")")

		size_gt = 0
		size_tst = 0
		positive = 0
		# Iterate through all pixels
		for y in range (0, height):
			for x in range (0, width):
				if ground[y, x] != 0 and test[y, x] != 0:
					positive += 1
				
				if ground[y, x] != 0:
					size_gt += 1

				if test[y, x] != 0:
					size_tst += 1


		dice = 2 * positive / (size_gt + size_tst)

		print("> " + str(dice))
		dice_sum += dice
		dice_count += 1

	print("Average dice score: " + str(dice_sum/dice_count))


def main():
	parser = argparse.ArgumentParser(description='Trains and tests a machine learning model to perform skull stripping on MRI images.')
	parser.add_argument('-t','--trainpath', help='path to the training data. defaults to ./data/train/', default=TRAINING_DATA_PATH)
	parser.add_argument('-c','--classifier', help='path to the classifier, if saved trained and saved. path to save the classifier otherwise. defaults to ./classifier.pkl', default=CLASSIFIER_FILE_PATH)
	parser.add_argument('-p', '--process', action='store_true', help='process an image by stripping it\'s skull, instead of training the model. defaults to output the changed image, with the skull stripped.')
	parser.add_argument('-i', '--imagepath', help='path to the images to process. defults to ./data/test/', default=TESTING_DATA_PATH)
	parser.add_argument('-b', '--bitmask', action='store_true', help='stores the result as a bitmask instead of stripping the original image.')
	parser.add_argument('-r', '--resultpath', help='path to store the results of the processing. defaults to ./data/result', default=RESULT_DATA_PATH)
	parser.add_argument('-m', '--mask', nargs=3, help='apply a mask to dicom images [image directory] [mask directory] [output directory]', default=[])
	parser.add_argument('-d', '--dice', nargs=2, help='compute dice scores [ground truth directory] [result directory]', default=[])
	args = vars(parser.parse_args())

	bitmask_mode = args['bitmask']

	training_data_path = args['trainpath']
	testing_data_path = args['imagepath']
	result_data_path = args['resultpath']
	classifier_file_path = args['classifier']

	if args['process']:
		if not os.path.exists(result_data_path):
			os.makedirs(result_data_path)
		batch_process_images(testing_data_path, result_data_path, classifier_file_path, bitmask_mode)
	elif len(args['mask']) > 0:
		if not os.path.exists(result_data_path):
			os.makedirs(result_data_path)
		apply_mask(args['mask'][0], args['mask'][1], args['mask'][2])		
	elif len(args['dice']) > 0:
		compute_dice(args['dice'][0], args['dice'][1])
	else:
		generate_training_data(training_data_path)
		train_model(classifier_file_path)
		test_model(0.2)

if __name__ == "__main__":
	main()