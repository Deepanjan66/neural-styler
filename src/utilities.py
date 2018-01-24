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

def gram_sum(arr):
    l, w, n = arr.shape
    arr = np.flatten(arr)
    
    pos_i = l*w
    pos_j = l*w
    sum = 0
    """
    while pos_i + (l*w) < i*w*n:
        for i in range(pos_i - (l*w), pos_i):
            for j in range(pos_j, pos_j + (l*w)):
                sum += arr[i] * arr[j]
            pos_j += (l*w)
        pos_i += (l*w)
        pos_j = pos_i
    """


