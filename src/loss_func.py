from keras import backend as K
import numpy as np
import tensorflow as tf

from configs import pretrained_network
from utilities import get_model_layers

def mean_squared_loss(y_true, y_pred):
    return y_pred
