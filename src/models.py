from collections import defaultdict
from keras.applications import vgg19
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense, Conv2D, UpSampling2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate, Lambda, Add
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model

from utilities import *
from custom_layers import pretrained_layer
from loss_func import *

from configs import pretrained_network

class NeuralModel:
    def __init__(self, pretrained_model, input_shape, style_layers, content_layers):
        #self.loss_function = loss_function
        self.input_shape = input_shape
        self.network = None
        self.style_layers = style_layers
        self.content_layers = content_layers
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

        #input_tensor = Input(shape=texture_image.shape[1:], name="vgg_input")
        vgg_model = vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=texture_image[1:])
        
        intermediary_layers = defaultdict(list)
        input_vec = texture_image
        for i in range(len(vgg_model.layers)):
            vgg_model.layers[i].trainable = False
            input_vec = vgg_model.layers[i](input_vec)
            if i in self.style_layers:
                intermediary_layers['style'].append(input_vec)
            elif i in self.content_layers:
                intermediary_layers['content'].append(input_vec)

        gram_res = []
        print("started calculating gram matrices for network outputs")
        for layer in intermediary_layers['style']:
            print("Looking at:",layer)
            gram_res.append(Lambda(gram_matrix_sum, name="style" + str(len(gram_res)))(layer))
            print("Finished with:",layer)

        print("Done getting all gram matrices")
        output_layers = [texture_image] + intermediary_layers['content'] + gram_res
        sgd = SGD(lr=0.01, momentum=0.1, decay=0.0, nesterov=True)

        self.model = Model(inputs=inputs, outputs=output_layers)
        self.model.compile(optimizer=sgd, loss=mean_squared_loss)
        print("Creating graph image")
        plot_model(self.model, to_file='updated_model1.png')
        print("Created graph image")


    def fit_through_pretrained_network(self, images):
        if not self.pred_functors:
            raise ValueError("Please provide pretrained model for training")
        
        targets = []
        for img in images['content']:
            #targets.append([np.array(func(img, 1.))[0][0] for func in self.pred_functors['all']])
            targets.append(np.array([np.array(func([img, 1.]))[0][0] for func in self.pred_functors['content']]))
        for img in images['style']:
            #targets.append([np.array(func(img, 1.))[0][0] for func in self.pred_functors['all']])
            targets.append(np.array([np.array(func([img, 1.]))[0][0] for func in self.pred_functors['style']]))
        gram_values = [targets[0]]
        for layer in targets[1]:
            mat = np.array(gram_matrix_training(layer))
            gram_values.append(mat)
        return gram_values
        #print(np.array(targets).shape)

    def fit(self, images):
        target = self.fit_through_pretrained_network(images)
        rand_img = [np.expand_dims(np.random.randint(0, high=1, size=(int(256/2**i),int(256/2**i), 3)), axis=0) \
                for i in range(6)]
        print("Training network with provided training images")
        checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)

        self.model.fit(rand_img, images['content'] + target, epochs=10000, callbacks=[checkpointer])
        self.model.save_weights('/tmp/weights.hdf5')

    def pred(self, img):
        return self.model.predict(img)




network = NeuralModel(pretrained_network, (0,), [1,4,7,12,17], [13])
network.define_generator_model()

img = np.array(image.load_img('test.png', target_size=(256, 256, 3))) / 255
img = np.expand_dims(img, axis=0)
content_img = np.array(image.load_img('content_test.png', target_size=(int(256), int(256), 3))) / 255
content_img = np.expand_dims(content_img, axis=0)
network.fit({'style':[img], 'content':[content_img]})
rand_img = [np.expand_dims(np.random.randint(0, high=1, size=(int(256/2**i),int(256/2**i), 3)), axis=0) \
        for i in range(1, 6)]
img = np.array(image.load_img('content_test.png', target_size=(int(256), int(256), 3))) / 255
img = np.expand_dims(img, axis=0)
rand_img = [img] + rand_img
res = network.pred(rand_img)[0].reshape((256,256,3))

plt.imshow(res)
plt.show()

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
