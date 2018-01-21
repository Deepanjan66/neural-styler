from keras.layers import Conv2D

def get_conv_block(model_input, filters, \
                   kernel, activation, \
                   normalization):
    """
    :param model_input: Input vector for the model
    :param filter: Number of filters to be used in the model
    :param kernel: Shape of the kernel to be used for convolution
    :param activation: Activation function to be used in this layer
    :para, normalization: Normalization to be used in this layer
    """
    model = Conv2D(filters=filters, kernel_size=kernel_size, \
            strides=(1,1), padding="valid", \
            activation="linear")(model_input)
    model = normalization(model)
    model = activation(model)

    return model
