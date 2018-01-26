from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class pretrained_layer(Layer):
    def __init__(self, output_dim, pretrained_layer_functors, **kwargs):
        self.output_dim = output_dim
        self.pretrained_layer_functors = pretrained_layer_functors
        super(pretrained_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(pretrained_layer, self).build(input_shape)

    def call(self, x):
        return np.array([np.array(func(img, 1.))[0][0] for func in self.pretrained_layer_functors])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

