from keras.layers import Conv2D
import numpy as np

def get_conv_block(num_blocks, model_input, filters, \
                   kernels, activation, \
                   normalization):
    """
    :param model_input: Input vector for the model
    :param filter: Number of filters to be used in the model
    :param kernel: Shape of the kernel to be used for convolution
    :param activation: Activation function to be used in this layer
    :para, normalization: Normalization to be used in this layer
    """
    list_args = convert_to_list(num_blocks, filters=filters, \
                                kernels=kernels, \
                                activation=activation, \
                                normalization=normalization)
    model = model_input
    for i in range(num_blocks):
        model = Conv2D(filters=list_args['filters'][i], kernel_size=list_args['kernels'][i], \
                strides=(1,1), padding="same", \
                activation="linear", data_format="channels_last")(model)
        model = list_args['normalization'][i](model)
        model = list_args['activation'][i](model)

    return model

def convert_to_list(num_blocks, **kwargs):
    for kwarg, val in kwargs.items():
        if type(val) is not list:
            kwargs[kwarg] = [val for _ in range(num_blocks)]

    return kwargs

def gram_matrix_sum(arr):
    shape_dict = {'x':arr.shape[0], 'y': arr.shape[1], 'layers':arr.shape[2]}
    j = 0
    gram_sum = 0
    for layer in range(j, shape_dict['layers']):
        for neuron in range(j + 1, shape_dict['layers']):
            gram_sum += np.multiply(arr[:,:,layer], arr[:,:,neuron])
        j += 1

    return gram_sum


