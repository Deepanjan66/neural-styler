from collections import defaultdict
from keras.applications import vgg19
from keras.optimizers import SGD, Adam
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense, Conv2D, UpSampling2D, Activation
from keras.models import Model, load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate, Lambda, Add
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
from tqdm import tqdm

from utilities import *
from custom_layers import pretrained_layer
from loss_func import *

from configs import *

class NeuralModel:
    def __init__(self, input_shape, 
                style_layers=pretrained_style_layers, 
                content_layers=pretrained_content_layers, 
                pretrained_model=pretrained_network,
                weight_file=None,
                loss_weights=[1,1,1,1,1,1,1]):
        #self.loss_function = loss_function
        self.input_shape = input_shape
        self.network = None
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.add_pretrained_model(pretrained_model)
        self.weight_file=weight_file
        self.loss_weights = loss_weights

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
                            activation=Activation('tanh'))

        #texture_image = texture_image / K.max(texture_image)

        intermediary_layers = defaultdict(list)
        input_vec = texture_image
        for i in range(len(self.pretrained_model.layers)):
            self.pretrained_model.layers[i].trainable = False
            input_vec = self.pretrained_model.layers[i](input_vec)
            if i in self.style_layers:
                intermediary_layers['style'].append(input_vec)
            elif i in self.content_layers:
                intermediary_layers['content'].append(input_vec)

        gram_res = []
        print("started calculating gram matrices for network outputs")
        for layer in intermediary_layers['style']:
            gram_res.append(Lambda(gram_matrix_sum, name="style" + str(len(gram_res)))(layer))
            print("Finished with:",layer)
        
        print("Done getting all gram matrices")

        output_layers = [texture_image] + intermediary_layers['content'] + gram_res
        adam = Adam(lr=0.01)
        self.model = Model(inputs=inputs, outputs=output_layers)
        self.model.compile(optimizer=adam, loss=mean_squared_loss, loss_weights=self.loss_weights)
        print("Creating graph image")
        plot_model(self.model, to_file='updated_model1.png')
        print("Created graph image")
        if self.weight_file:
            self.model.load_weights(self.weight_file)


    def fit_through_pretrained_network(self, images):
        if not self.pred_functors:
            raise ValueError("Please provide pretrained model for training")
        
        targets = []

        for content, style in zip(images['content'], images['style']):
            content_functors = np.array([np.array(func([content, 1.]))[0][0] for func in self.pred_functors['content']])
            style_functors = np.array([np.array(func([style, 1.]))[0][0] for func in self.pred_functors['style']]) 
            targets.append([content_functors, style_functors])
        all_targets = []
        for target in targets:
            gram_values = [target[0]]
            for layer in target[1]:
                mat = np.array(gram_matrix_training(layer))
                gram_values.append(mat)
            all_targets.append(gram_values)

        return all_targets
    

    def fit(self, images):
        all_targets = self.fit_through_pretrained_network(images)
        
        print("Training network with provided training images")
        checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
        #lr_schedular = LearningRateScheduler(schedular)
        #for _ in range(1000):
        mixed_targets = [[content] + target for content, target in zip(images['content'], all_targets)]
        for _ in tqdm(range(100)):
            rand_img = [np.expand_dims(np.random.randint(0, high=1, 
                        size=(int(256/2**i),int(256/2**i), 3)), axis=0) \
                        for i in range(6)]
            self.model.fit(rand_img, mixed_targets[0], callbacks=[checkpointer], batch_size=1)
            self.model.save_weights('my_weights.hdf5')
        self.model.save('my_model.h5')

    def pred(self, img):
        rand_img = [np.expand_dims(np.random.randint(0, high=1, 
                    size=(int(256/2**i),int(256/2**i), 3)), 
                    axis=0) for i in range(1, 6)]
        img = [img] + rand_img
        return self.model.predict(img)


