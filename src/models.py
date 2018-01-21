from keras.applications import vgg19
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense, Conv2D, UpSampling2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
import matplotlib.pyplot as plt
import numpy as np

from utilities import *


class NeuralModel:
    def __init__(self, pretrained_model, input_shape):
        self.pretrained_model = pretrained_model
        self.input_shape = input_shape
        self.network = None

    def define_generator_model(self, dim=256, num_tensors=6):
        inputs = []
        for i in range(num_tensors):
            inputs.append(Input(shape=(dim/(2**i), dim/(2**i), 3), name="z_input" + str(i)))

        models = []
        for z_input in inputs:
            model = get_conv_block(num_blocks=3, model_input=z_input, filters=8, 
                                   kernels=[(3,3), (3,3), (1,1)],\
                                   normalization=BatchNormalization(), \
                                   activation=LeakyReLU(alpha=.001))

            models.append(model)
        
        for i in reversed(range(len(models))):

            upsampled_output = UpSampling2D(size=(2,2), data_format=None)(models[i])
            prev_normalized = BatchNormalization()(upsampled_output)
            next_normalized = BatchNormalization()(models[i-1])
            print(prev_normalized, "And another: ", next_normalized)
            exit(1)

            merged_model = concatenate([prev_scale, model])


        self.network = model
        
        return models



network = NeuralModel(None, (0,))
network.define_generator_model()
"""
img = np.array(image.load_img('test.png', target_size=(3, 224, 224)))
img = np.expand_dims(img, axis=0)
input = Input(shape=(3, 224, 224), name='image_input')
model = vgg19.VGG19(include_top=False, weights='imagenet')
model = model(input)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mean_squared_error')
result = model.predict(img)
print(result.shape)
"""
