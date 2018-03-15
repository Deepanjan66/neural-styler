# Neural-Styler

This is an exploratory keras implementation of a neural style transfer algorithm that was heavily inspired by the following
two brilliant research papers:

i) [Texture Networks: Feed-forward Synthesis of Textures and Stylized Images](https://arxiv.org/pdf/1603.03417)

ii) [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/pdf/1607.08022)

Neural style transfer algorithms are a class of deep neural networks that given a content image (image of a dog(s), human 
being(s) ,building(s) or something of interest) and a style image (a famous painting or a colorful scenary), produces an image
that highlights the main subject(s) of the content image and adds stylistic aspects from the style image. The [original paper](https://arxiv.org/pdf/1508.06576)
amazed everyone with this new way of using a neural network. The results obtained were beautiful and appealed to people from 
all walks of life.

## How does stylization with neural networks work?

At the heart of neural style transfer algorithms lie pretrained deep neural networks from the [ImageNet Large Scale Visual Recognition Challenge](http://www.image-net.org/challenges/LSVRC/). The authors of the original paper realised that the deep 
networks used for identifying different subjects from the ImageNet dataset encode different types of information about
images at different layers. They found that the layers at the bottom (for a network that runs from bottom to top) encode 
information about the overall texture and look of the image while the layers towards the top encode details and information 
about the actual subject in the image. They used this feature of deep models trained on the ImageNet dataset to extract
relavant informatiom from the style and content image. From here on, we will identify output of lower layers (used in case of 
style images) as _**style outputs**_ and output of higher layers (used in case of content images) as _**content outputs**_.
They then trained their own network that takes the content outputs and style outputs as input and after some training 
produces an image that pleasantly merges useful details of both images. They compared the output of their network with the 
output of higher layers of VGG (for content image) and lower layers of VGG (for style image) to train the network. Hence, 
their optimization problem became the task of optimizing style and content loss obtained from comparing the output of their
network and the style and content outputs from the VGG network.

In the original paper, the authors also noted the use of gram matrices for the style outputs. They used gram matrices to
ensure that their network is invariant to the absolute locations of the pixels from the style outputs. This makes sense 
when you look at the stylized images. The subject of the content image remains noticable in its actual postion while the 
different aspects of the style image seem more distributed which creates variety in stylization.

## Fast Stylization approaches

While the results obtained from the network discussed in the original paper were really good, the optimization step
was time consuming. The optimization step would have to be run separately for every style image. Researchers interested in the 
field then started looking at improvements that could reduce the time taken and help the network generalize better. The next
breakthrough was obtained by Russian researchers [i] who came up with the idea of _Texture networks_. They moved the 
time consuming operation of optimizing network for every style image to the training step. 


## My implementation so far

With my implementation, I aim to use the research outcomes of the papers mentioned above and explore different techniques
for creating a network that generalizes better. My current architecture is very similar to the architecture showed in [i].
Below is a keras generated image of the network in my implementation.



## Results from my network

This implementation is still being trained. Intermediary training results are shown below:













