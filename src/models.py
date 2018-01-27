from collections import defaultdict
from keras.applications import vgg19
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense, Conv2D, UpSampling2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

from utilities import *
from custom_layers import pretrained_layer
from loss_func import *

from configs import pretrained_network

class NeuralModel:
    def __init__(self, pretrained_model, input_shape):
        #self.loss_function = loss_function
        self.input_shape = input_shape
        self.network = None
        if pretrained_model:
            self.add_pretrained_model(pretrained_model)

    def add_pretrained_model(self, model):
        self.pretrained_model = model
        self.pred_functors = get_model_layers(model)

    def define_generator_model(self, dim=256, num_tensors=6, num_channels=3):
        inputs = []
        for i in range(num_tensors):
            inputs.append(Input(shape=(dim/(2**i), dim/(2**i), num_channels), name="z_input" + str(i)))

        models = []
        for z_input in inputs:
            model = get_conv_block(\
                    num_blocks=3, model_input=z_input, filters=8, 
                    kernels=[(3,3), (3,3), (1,1)],\
                    normalization=BatchNormalization(), \
                    activation=LeakyReLU(alpha=.001))

            models.append(model)

        # We'll increase the number of filters after every join by 8
        curr_filters = 16
        # This variable is assigned the last model output so that
        # the same variable can be used to merge all outputs in the loop
        merged_models = models[-1]

        for i in reversed(range(1, len(models))):
            
            upsampled_output = UpSampling2D(size=(2,2), data_format=None)(merged_models)
            prev_normalized = BatchNormalization()(upsampled_output)
            next_normalized = BatchNormalization()(models[i-1])
            merged_models = concatenate([prev_normalized, next_normalized])

            merged_models = get_conv_block(
                                num_blocks=3, model_input=merged_models, 
                                filters=curr_filters, 
                                kernels=[(3,3), (3,3), (1,1)],
                                normalization=BatchNormalization(), 
                                activation=LeakyReLU(alpha=.001))

            curr_filters += 8

        final_out = get_conv_block(
                            num_blocks=3, model_input=merged_models, \
                            filters=[curr_filters, curr_filters, curr_filters],\
                            kernels=[(3,3), (3,3), (1,1)],\
                            normalization=BatchNormalization(), \
                            activation=LeakyReLU(alpha=.001))

        texture_image = get_conv_block(
                            num_blocks=1, model_input=final_out, 
                            filters=num_channels,
                            kernels=[(1,1)],
                            normalization=BatchNormalization(), 
                            activation=LeakyReLU(alpha=.001))
        print(texture_image)
        self.model = Model(inputs=inputs, outputs=[texture_image])
        self.model.compile(optimizer='sgd', loss=mean_squared_loss)
    
    def fit_through_pretrained_network(self, images):
        if not self.pred_functors:
            raise ValueError("Please provide pretrained model for training")
        
        targets = []
        for img in images['style']:
            #targets.append([np.array(func(img, 1.))[0][0] for func in self.pred_functors['all']])
            targets.append(np.array([np.array(func([img, 1.]))[0][0] for func in self.pred_functors['style']]))
        gram_values = []
        for layer in targets[0]:
            gram_values.append(gram_matrix_sum(layer))
        return gram_values
        #print(np.array(targets).shape)

    def fit(self, images):
        target = self.fit_through_pretrained_network(images)
        rand_img = [np.random.randn(2, 4, 3) for _ in range(6)]
        self.model.fit(rand_img, target, epochs=10)


        

network = NeuralModel(pretrained_network, (0,))
network.define_generator_model()

img = np.array(image.load_img('test.png', target_size=(256, 256, 3)))
img = np.expand_dims(img, axis=0)
network.fit({'style':[img]})
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
