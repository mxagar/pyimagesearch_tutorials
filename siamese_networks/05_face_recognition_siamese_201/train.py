# USAGE
# python train.py

# import the necessary packages
from pyimagesearch.dataset import TripletGenerator
from pyimagesearch.model import get_embedding_module
from pyimagesearch.model import get_siamese_network
from pyimagesearch.model import SiameseModel
from pyimagesearch.dataset import MapFunction
from pyimagesearch import config
from tensorflow import keras
import tensorflow as tf
import os

# create the data input pipeline for train and val dataset
print("[INFO] building the train and validation generators...")
trainTripletGenerator = TripletGenerator(
	datasetPath=config.TRAIN_DATASET)
valTripletGenerator = TripletGenerator(
	datasetPath=config.TRAIN_DATASET)
print("[INFO] building the train and validation `tf.data` dataset...")
trainTfDataset = tf.data.Dataset.from_generator(
	generator=trainTripletGenerator.get_next_element,
	output_signature=(
		tf.TensorSpec(shape=(), dtype=tf.string),
		tf.TensorSpec(shape=(), dtype=tf.string),
		tf.TensorSpec(shape=(), dtype=tf.string),
	)
)
valTfDataset = tf.data.Dataset.from_generator(
	generator=valTripletGenerator.get_next_element,
	output_signature=(
		tf.TensorSpec(shape=(), dtype=tf.string),
		tf.TensorSpec(shape=(), dtype=tf.string),
		tf.TensorSpec(shape=(), dtype=tf.string),
	)
)

# preprocess the images
mapFunction = MapFunction(imageSize=config.IMAGE_SIZE)
print("[INFO] building the train and validation `tf.data` pipeline...")
trainDs = (trainTfDataset
    .map(mapFunction)
    .shuffle(config.BUFFER_SIZE)
    .batch(config.BATCH_SIZE)
    .prefetch(config.AUTO)
)
valDs = (valTfDataset
    .map(mapFunction)
    .batch(config.BATCH_SIZE)
    .prefetch(config.AUTO)
)

# build the embedding module and the siamese network
print("[INFO] build the siamese model...")
embeddingModule = get_embedding_module(imageSize=config.IMAGE_SIZE)
siameseNetwork =  get_siamese_network(
	imageSize=config.IMAGE_SIZE,
	embeddingModel=embeddingModule,
)
siameseModel = SiameseModel(
	siameseNetwork=siameseNetwork,
	margin=0.5,
	lossTracker=keras.metrics.Mean(name="loss"),
)

# compile the siamese model
siameseModel.compile(
	optimizer=keras.optimizers.Adam(config.LEARNING_RATE)
)

# train and validate the siamese model
print("[INFO] training the siamese model...")
siameseModel.fit(
	trainDs,
	steps_per_epoch=config.STEPS_PER_EPOCH,
	validation_data=valDs,
	validation_steps=config.VALIDATION_STEPS,
	epochs=config.EPOCHS,
)

# check if the output directory exists, if it doesn't, then
# create it
if not os.path.exists(config.OUTPUT_PATH):
	os.makedirs(config.OUTPUT_PATH)

# save the siamese network to disk
modelPath = config.MODEL_PATH
print(f"[INFO] saving the siamese network to {modelPath}...")
keras.models.save_model(
	model=siameseModel.siameseNetwork,
	filepath=modelPath,
	include_optimizer=False,
)