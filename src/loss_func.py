from keras import backend as K
import numpy as np

from configs import pretrained_network
from utilities import get_model_layers

def mean_squared_loss(y_true, y_pred):
    pred_functors = get_model_layers(pretrained_network)
    sess = K.get_session()
    print(y_pred.eval(session=sess))

    #targets = []
    #targets.append(np.array([np.array(func([y_pred, 1.]))[0][0] for func in pred_functors['style']]))
    #print(targets.shape)
    exit(1)
    return y_pred
