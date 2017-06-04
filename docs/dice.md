# Analyzing our results

Although we could visually see that our results were quite good, we wanted to quantify the accuracy of our results. For this we used the Sørensen–Dice coefficient which is often used in medical imaging to quantify segmentation quality.

## Computing the Dice Score
We computed the Dice score as such:
<div style="text-align:center"><img src ="https://wikimedia.org/api/rest_v1/media/math/render/svg/e02846ea5780d8c2afaecff495bdcd654d1f93f5" /></div>
where X is the set of segmented pixels in the ground truth bit masks and Y is the set of segmented pixels in the resulting bit masks

```python
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
```