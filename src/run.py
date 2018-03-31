import numpy as np
import matplotlib.pyplot as plt

from models import NeuralModel
from utilities import *

if __name__=="__main__":

    network = NeuralModel((0,))
    network.define_generator_model()


    #style_images = get_all_images('style_images/', n = 2)

    style_img = np.array(image.load_img('test.png', target_size=(int(256), int(256), 3))) / 255
    style_img = np.expand_dims(style_img, axis=0)

    print(np.amax(style_img))

    content_img = np.array(image.load_img('content_test.png', target_size=(int(256), int(256), 3))) / 255
    content_img = np.expand_dims(content_img, axis=0)
    network.fit({'style':[style_img], 'content':[content_img]})

    img = np.array(image.load_img('content_test.png', target_size=(int(256), int(256), 3))) / 255
    img = np.expand_dims(img, axis=0)
    res = network.pred(img)[0].reshape((256,256,3))
    print(np.amax(res))
    res = res * 255

    plt.imshow(res)
    plt.show()
