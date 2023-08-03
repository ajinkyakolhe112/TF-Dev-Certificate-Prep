"""
Goal:	Mnist dense classification to CNN classification
"""

#%%
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers,activations, losses, metrics, optimizers


import os
from loguru import logger
import tensorflow.experimental.numpy as tf_np
import tensorflow as tf
import numpy as np

#%% data in bchw format

logger.debug("path = " + os.getcwd())
relative_path = "/data/mnist.npz"
data_path = os.getcwd() + relative_path
logger.debug(data_path)

dataset_train, dataset_validation = keras.datasets.mnist.load_data(data_path)
(images_train, labels_train), (images_validation, labels_validation) = dataset_train, dataset_validation

images, labels = images_train, labels_train		# easy code reading & writing
logger.debug(f'{images.shape}{labels.shape}, -> B,H,W,C (Channels Last)')
B, H, W = images.shape
images = images.reshape(B,H,W,1)
images = images/255.0	# np.max(images) - np.min(images)
labels = keras.utils.to_categorical(labels,num_classes= 10)

logger.debug(f'{images[0][0:15,0:15,]} -> First 15 Rows & Columns of 1st Image')
logger.debug(f'{labels[0:5]}')

#%%

smallest_model = keras.models.Sequential([
	layers.Input(shape=(28,28,1)),
	layers.Flatten(),
	layers.Dense(10, "relu"),
	layers.Dense(10, "softmax"),
])

logger.debug(f'single prediction = {smallest_model(tf_np.random.randn(1,28,28,1))}') # sum of output = 1
logger.debug(f'{smallest_model.summary()}')
#%%

smallest_model.compile(
	loss= losses.categorical_crossentropy,
	optimizer= optimizers.SGD(),
	metrics = ["accuracy"],
)

#%%
class CustomCallback(keras.callbacks.Callback):
	def on_batch_end(self, batch_no, logs=None):
		dir(self)
	
	def on_epoch_end(self, epoch_no, logs=None):
		if logs['accuracy'] - 0.89 >= 0: # Hidden Behaviour here
			self.model.stop_training = True
			logger.debug(f"Accuracy of {logs['accuracy']}, achieved in {epoch_no}")
		
	
smallest_model.fit(images, labels, epochs = 10, callbacks = [CustomCallback()])
# %%
smallest_cnn_model = keras.models.Sequential([
	layers.Input(shape=(28,28,1)),
	layers.Conv2D(10,3,activation="relu"),
	layers.Flatten(),
	layers.Dense(10, "relu"),
	layers.Dense(10, "softmax"),
])
# %%
