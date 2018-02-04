from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.losses import mean_squared_error

from configs import pretrained_network
from utilities import get_model_layers

def mean_squared_loss(y_true, y_pred):
    if "leaky" in y_pred.name:
        return K.constant(0)
    loss = mean_squared_error(y_true, y_pred)
    return loss / K.sum(loss)
