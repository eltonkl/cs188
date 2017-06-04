# Using the trained model

Now that we had trained our model using the skull stripping data, we wanted to apply its learning on our testing data. 

## Applying the classifier to the test data

To apply our trained classifer on the testing data, we first loaded the classifier from the classifier file that we had written to during training. We then used it to predict the results for each pixel in the testing image, row by row. Once we predicted the results for all the rows in the image we saved the updated dicom file to an output file given by the "output_dir" parameter. Note, we set all the margins to a value of 0 in our output image.

```python
##
# Takes an input dicom image and saves a processed copy
#
# input_name: [string] name of image file
# last_four_args: [tuple] contains:
#   input_dir: [string] directory where image file is be stored
#   output_dir: [string] directory where output image file will be stored
#   classifier_path: [string] path to classifier pickle
#   btmask_mode: [boolean] if true, the output image will be the bitmask for the
#                   skull stripped image, otherwise it will be the actual skull
#                   stripped image
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

        # Save modified image in new file
        predict_dc_file.PixelData = predict_dc_file.pixel_array.tostring()
        predict_dc_file.save_as(output_dir + image_name)

        print(image_name + " completed! File saved to " + output_dir + image_name)

    except Exception as e:
        print("[ERROR] " + image_name + ": ")
        print(e)
```

## Speeding up the testing process

Processing the testing images one by one was a tedious process and so we automated it. We wrote the batch_process_images function to allow us to process all images in the input directory that start with "IM-" and save them to output directory.
We used Python's Pool library to multiprocess this task and gained a linear decrease in the amount of time spent processing the testing data.

```python
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
```