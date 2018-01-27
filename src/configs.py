from keras.applications import vgg19
from keras.layers import Input

pre_in = Input(shape=(256, 256, 3), name="network_output")
pretrained_network = vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=pre_in)

