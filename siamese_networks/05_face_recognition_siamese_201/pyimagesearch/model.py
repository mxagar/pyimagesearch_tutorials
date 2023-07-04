# import the necessary packages
from tensorflow.keras.applications import resnet
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

def get_embedding_module(imageSize):
	# construct the input layer and pass the inputs through a
	# pre-processing layer
	inputs = keras.Input(imageSize + (3,))
	x = resnet.preprocess_input(inputs)
	
	# fetch the pre-trained resnet 50 model and freeze the weights
	baseCnn = resnet.ResNet50(weights="imagenet", include_top=False)
	baseCnn.trainable=False
	
	# pass the pre-processed inputs through the base cnn and get the
	# extracted features from the inputs
	extractedFeatures = baseCnn(x)

	# pass the extracted features through a number of trainable layers
	x = layers.GlobalAveragePooling2D()(extractedFeatures)
	x = layers.Dense(units=1024, activation="relu")(x)
	x = layers.Dropout(0.2)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dense(units=512, activation="relu")(x)
	x = layers.Dropout(0.2)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dense(units=256, activation="relu")(x)
	x = layers.Dropout(0.2)(x)
	outputs = layers.Dense(units=128)(x)

	# build the embedding model and return it
	embedding = keras.Model(inputs, outputs, name="embedding")
	return embedding

def get_siamese_network(imageSize, embeddingModel):
	# build the anchor, positive and negative input layer
	anchorInput = keras.Input(name="anchor", shape=imageSize + (3,))
	positiveInput = keras.Input(name="positive", shape=imageSize + (3,))
	negativeInput = keras.Input(name="negative", shape=imageSize + (3,))

	# embed the anchor, positive and negative images
	anchorEmbedding = embeddingModel(anchorInput)
	positiveEmbedding = embeddingModel(positiveInput)
	negativeEmbedding = embeddingModel(negativeInput)

	# build the siamese network and return it
	siamese_network = keras.Model(
		inputs=[anchorInput, positiveInput, negativeInput],
		outputs=[anchorEmbedding, positiveEmbedding, negativeEmbedding]
	)
	return siamese_network

class SiameseModel(keras.Model):
	def __init__(self, siameseNetwork, margin, lossTracker):
		super().__init__()
		self.siameseNetwork = siameseNetwork
		self.margin = margin
		self.lossTracker = lossTracker

	def _compute_distance(self, inputs):
		(anchor, positive, negative) = inputs
		# embed the images using the siamese network
		embeddings = self.siameseNetwork((anchor, positive, negative))
		anchorEmbedding = embeddings[0]
		positiveEmbedding = embeddings[1]
		negativeEmbedding = embeddings[2]

		# calculate the anchor to positive and negative distance
		apDistance = tf.reduce_sum(
			tf.square(anchorEmbedding - positiveEmbedding), axis=-1
		)
		anDistance = tf.reduce_sum(
			tf.square(anchorEmbedding - negativeEmbedding), axis=-1
		)
		
		# return the distances
		return (apDistance, anDistance)

	def _compute_loss(self, apDistance, anDistance):
		loss = apDistance - anDistance
		loss = tf.maximum(loss + self.margin, 0.0)
		return loss

	def call(self, inputs):
		# compute the distance between the anchor and positive,
		# negative images
		(apDistance, anDistance) = self._compute_distance(inputs)
		return (apDistance, anDistance)

	def train_step(self, inputs):
		with tf.GradientTape() as tape:
			# compute the distance between the anchor and positive,
			# negative images
			(apDistance, anDistance) = self._compute_distance(inputs)

			# calculate the loss of the siamese network
			loss = self._compute_loss(apDistance, anDistance)

		# compute the gradients and optimize the model
		gradients = tape.gradient(
			loss,
			self.siameseNetwork.trainable_variables)
		self.optimizer.apply_gradients(
			zip(gradients, self.siameseNetwork.trainable_variables)
		)

		# update the metrics and return the loss
		self.lossTracker.update_state(loss)
		return {"loss": self.lossTracker.result()}

	def test_step(self, inputs):
		# compute the distance between the anchor and positive,
		# negative images
		(apDistance, anDistance) = self._compute_distance(inputs)

		# calculate the loss of the siamese network
		loss = self._compute_loss(apDistance, anDistance)
		
		# update the metrics and return the loss
		self.lossTracker.update_state(loss)
		return {"loss": self.lossTracker.result()}

	@property
	def metrics(self):
		return [self.lossTracker]