from keras.layers import Conv2D

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
                strides=(1,1), padding="valid", \
                activation="linear", data_format="channels_last")(model)
        model = list_args['normalization'][i](model)
        model = list_args['activation'][i](model)

    return model

def convert_to_list(num_blocks, **args):
    for arg, val in args.items():
        if type(val) is not list:
            args[arg] = [val for _ in range(num_blocks)]

    return args

