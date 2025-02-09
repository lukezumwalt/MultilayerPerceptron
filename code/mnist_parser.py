'''
MNIST Data Parser

Accepts and pre-processes, flattens data files provided as: 
- <context>-images.idx3-ubyte
- <context>-labels.idx1-ubyte

Example usage:
( x_train, y_train ) = ...\
    load_and_preprocess_mnist("train-images.idx3-ubyte",
                              "train-labels.idx1-ubyte")
( x_test, y_test )= ...\
    load_and_preprocess_mnist("t10k-images.idx3-ubyte",
                              "t10k-labels.idx1-ubyte")
'''

import struct
import numpy as np

import torch

def load_mnist_images(filename):
    '''
    Loads MNIST images from IDX3-UBYTE format file.
    '''
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def load_mnist_labels(filename):
    '''
    Loads MNIST labels from IDX1-UBYTE format file.
    '''
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def preprocess_mnist(images, labels):
    '''
    Preprocess MNIST data by normalizing images and converting labels to one-hot encoding.
    '''
    # Normalize pixel values
    images = torch.tensor(images, dtype=torch.float32) / 255.0
    # Convert labels to one-hot encoding
    labels_onehot = torch.eye(10)[torch.tensor(labels, dtype=torch.long)]
    # Flatten images
    return images.reshape(images.shape[0], -1), labels_onehot

def load_and_preprocess_mnist(image_path, label_path):
    '''
    Loads and preprocesses MNIST dataset from given paths.
    '''
    images = load_mnist_images(image_path)
    labels = load_mnist_labels(label_path)
    return preprocess_mnist(images, labels)
