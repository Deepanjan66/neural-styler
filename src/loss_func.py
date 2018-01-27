import numpy as np

from configs import pretrained_network
from utilities import get_model_layers

def mean_squared_loss(y_true, y_pred):
    pred_functors = get_model_layers(pretrained_network)
    targets = []
    print(y_true)
    exit(1)
    y_true = np.array(y_true[0])
    targets.append([np.array(func([y_true, 1.]))[0][0] for func in pred_functors['style']])
    print(targets)
    return y_pred
