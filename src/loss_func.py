from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.losses import mean_squared_error

from configs import pretrained_network
from utilities import get_model_layers

def mean_squared_loss(y_true, y_pred):
    loss = mean_squared_error(y_true, y_pred)
    return loss

def schedular(epoch):
    if epoch > 2300:
        return float(0.1)
    if epoch > 1500:
        return float(0.5)
    return float(1)


