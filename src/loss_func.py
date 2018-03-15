from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.losses import mean_squared_error

from configs import pretrained_network
from utilities import get_model_layers

def mean_squared_loss(y_true, y_pred):
    if "activation" in y_pred.name:
        return mean_squared_error(y_true, y_pred) * 0
    true_max = K.mean(y_true)
    if true_max != 0:
        y_true /= true_max
    pred_max = K.mean(y_pred)
    if pred_max != 0:
        y_pred /= pred_max
    loss = mean_squared_error(y_true, y_pred)
    return loss / K.sum(loss)
