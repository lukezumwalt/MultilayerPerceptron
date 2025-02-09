# MultilayerPerceptron
Multi-Layer Perception (MLP) Model, trained on MNIST data.

## Data
MNIST is a popular dataset used in the machine learning and deep learning communities.
It consists of 60000 training images and 10000 test images composed of black-and-white
handwritten digits from 0 − 9. The images are of the shape 28 × 28, however, we will
work exclusively with flattened versions of shape 282 = 784. Flattened means that we
reinterpret the 2-D images as 1-D vectors instead. Please note that this means d0, the
dimensionality of the input to the MLP model, must always be 784. Similarly, dn,
the dimensionality of the output, must always be 10, for the 10 classes.
