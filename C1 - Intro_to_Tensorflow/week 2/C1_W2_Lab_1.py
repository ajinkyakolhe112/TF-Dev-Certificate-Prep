#%%
from loguru import logger
import pytest
import wandb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
#%%
"Y = activation(dot(X,W))"
"X \cdot W -> activation -> error -> update w to reduce error"
import tensorflow as tf
from tensorflow.keras import datasets, models, layers, activations, losses, optimizers, metrics
"Every Tensor is TF Tensor not Pytorch"
"Think in terms of higher level abstractions for keras"
"Assume nothing. Code is point to point logical connections. Each step, checked, tested, designed, will lead to overall good coding. Better you have complete design of solution in mind, better you will code"

#%%
train_data, test_data = datasets.fashion_mnist.load_data()

logger.debug(f'train_data [type] {type(train_data)}, \
                test_data [type] {type(test_data)}')
logger.info(f'train_data & test_data are tuples. x,y = train_data & x,y = test_data')

x_train, y_train = train_data
x_test, y_test = test_data

def test_dataset():
    """
    Goal Format: (B,C,H,W)

    Advanced
    1. Why channel first is better than channel last
    2. 
    """
    assert len(x_train.shape) == 4 # B, C, H, W

    logger.debug(f'x_train shape ={x_train.shape}, y_train shape = {y_train.shape}')
    logger.info(f'No channel found in x_train. need to add it')

    logger.info(f'x_train = len,height,width = {x_train[0][0]}{x_train[0][0]}{x_train[0][0]}')

#%%
# Single Example


def test_single_example():
    """

    """
    one_example = x_train[0]

    logger.info(f'type = {type(one_example)}, not Tensorflow Tensor')
    logger.debug(f'tensorflow has tf.math.maximum(x,y) function \n and tf.math.reduce_max to get maximum value')
    logger.debug(f'one example shape = {one_example.shape} EQUIVALENT to {"1, "+str(one_example.shape)}')
    logger.debug(f'Channels = 1 infered. 1 means one color, max to lowest of that. Black & White')
    logger.info(f'values 0 to 255. Not standardized YET')
    logger.info(f'Reshape to add channel')
    logger.info(f'one pixel value = One value between {np.min(one_example)} & {np.max(one_example)}')
    logger.debug(f'256 values = 2^8 => 8 bits per color ')

    logger.debuf(f'One pixel = tuple of (R, G, B) , (R = 2^8, G = 2^8, B = 2^8 )=> 24 bits RGB => TRUE COLOR Human Eye Color Sensitivy')


    logger.debug(f'Visualize IMAGE one at a time too')
    plt.imshow(one_example)

    plt.imshow(one_example,cmap="gray")








# %%
