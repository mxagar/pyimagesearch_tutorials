# USAGE
# python crop_faces.py --dataset train_dataset --output cropped_train_dataset \
# --prototxt face_crop_model/deploy.prototxt.txt \
# --model face_crop_model/res10_300x300_ssd_iter_140000.caffemodel
#
# python crop_faces.py --dataset test_dataset --output cropped_test_dataset \
# --prototxt face_crop_model/deploy.prototxt.txt \
# --model face_crop_model/res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from imutils.paths import list_images
from tqdm import tqdm
import numpy as np
import argparse
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-o", "--output", required=True,
	help="path to output dataset")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# check if the output dataset directory exists, if it doesn't, then
# create it
if not os.path.exists(args["output"]):
	os.makedirs(args["output"])

# grab the file and sub-directory names in dataset directory
print("[INFO] grabbing the names of files and directories...")
names = os.listdir(args["dataset"])

# loop over all names
print("[INFO] starting to crop faces and saving them to disk...")
for name in tqdm(names):
	# build directory path
	dirPath = os.path.join(args["dataset"], name)

	# check if the directory path is a directory
	if os.path.isdir(dirPath):
		# grab the path to all the images in the directory
		imagePaths = list(list_images(dirPath))

		# build the path to the output directory
		outputDir = os.path.join(args["output"], name)

		# check if the output directory exists, if it doesn't, then
		# create it
		if not os.path.exists(outputDir):
			os.makedirs(outputDir)

		# loop over all image paths
		for imagePath in imagePaths:
			# grab the image ID, load the image, and grab the
			# dimensions of the image
			imageID = imagePath.split(os.path.sep)[-1]
			image = cv2.imread(imagePath)
			(h, w) = image.shape[:2]

			# construct an input blob for the image by resizing to a
			# fixed 300x300 pixels and then normalizing it
			blob = cv2.dnn.blobFromImage(cv2.resize(image,
				(300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

			# pass the blob through the network and obtain the
			# detections and predictions
			net.setInput(blob)
			detections = net.forward()

			# extract the index of the detection with max 
			# probability and get the maximum confidence value
			i = np.argmax(detections[0, 0, :, 2])
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the 
			# `confidence` is greater than the minimum confidence
			if confidence > args["confidence"]:
				# grab the maximum dimension value
				maxDim = np.max(detections[0, 0, i, 3:7])

				# check if max dimension value is greater than one,
				# if so, skip the detection since it is erroneous
				if maxDim > 1.0:
					continue

				# clip the dimension values to be between 0 and 1
				box = np.clip(detections[0, 0, i, 3:7], 0.0, 1.0)

				# compute the (x, y)-coordinates of the bounding
				# box for the object
				box = box * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
			
				# grab the face from the image, build the path to
				# the output face image, and write it to disk
				face = image[startY:endY,startX:endX,:]
				facePath = os.path.join(outputDir, imageID)
				cv2.imwrite(facePath, face)

print("[INFO] finished cropping faces and saving them to disk...")