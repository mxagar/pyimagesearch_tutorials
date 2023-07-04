# USAGE
# python inference.py

# import the necessary packages
from pyimagesearch.dataset import TripletGenerator
from pyimagesearch.dataset import MapFunction
from pyimagesearch.model import SiameseModel
from matplotlib import pyplot as plt
from pyimagesearch import config
from tensorflow import keras
import tensorflow as tf
import os

# create the data input pipeline for test dataset
print("[INFO] building the test generator...")
testTripletGenerator = TripletGenerator(
	datasetPath=config.TEST_DATASET)
print("[INFO] building the test `tf.data` dataset...")
testTfDataset = tf.data.Dataset.from_generator(
	generator=testTripletGenerator.get_next_element,
	output_signature=(
		tf.TensorSpec(shape=(), dtype=tf.string),
		tf.TensorSpec(shape=(), dtype=tf.string),
		tf.TensorSpec(shape=(), dtype=tf.string),
	)
)
mapFunction = MapFunction(imageSize=config.IMAGE_SIZE)
testDs = (testTfDataset
	.map(mapFunction)
	.batch(4)
	.prefetch(config.AUTO)
)

# load the siamese network from disk and build the siamese model
modelPath = config.MODEL_PATH
print(f"[INFO] loading the siamese network from {modelPath}...")
siameseNetwork = keras.models.load_model(filepath=modelPath)
siameseModel = SiameseModel(
	siameseNetwork=siameseNetwork,
	margin=0.5,
	lossTracker=keras.metrics.Mean(name="loss"),
)

# load the test data
(anchor, positive, negative) = next(iter(testDs))
(apDistance, anDistance) = siameseModel((anchor, positive, negative))
plt.figure(figsize=(10, 10))
rows = 4
for row in range(rows):
	plt.subplot(rows, 3, row * 3 + 1)
	plt.imshow(anchor[row])
	plt.axis("off")
	plt.title("Anchor image")
	plt.subplot(rows, 3, row * 3 + 2)
	plt.imshow(positive[row])
	plt.axis("off")
	plt.title(f"Positive distance: {apDistance[row]:0.2f}")
	plt.subplot(rows, 3, row * 3 + 3)
	plt.imshow(negative[row])
	plt.axis("off")
	plt.title(f"Negative distance: {anDistance[row]:0.2f}")

# check if the output directory exists, if it doesn't, then
# create it
if not os.path.exists(config.OUTPUT_PATH):
	os.makedirs(config.OUTPUT_PATH)

# save the inference image to disk
outputImagePath = config.OUTPUT_IMAGE_PATH
print(f"[INFO] saving the inference image to {outputImagePath}...")
plt.savefig(fname=outputImagePath)