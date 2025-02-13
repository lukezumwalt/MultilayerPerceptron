#Original source: https://www.kaggle.com/code/hojjatk/read-mnist-dataset
#It has been modified for ease of use w/ pytorch

#You do NOT need to modify ANY code in this file!

import numpy as np
import struct
from array import array
import torch

class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):
        n = 60000 if "train" in images_filepath else 10000
        labels = torch.zeros((n, 10))
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            l = torch.tensor(array("B", file.read())).unsqueeze(-1)
            l = torch.concatenate((torch.arange(0, n).unsqueeze(-1), l), dim = 1).type(torch.int32)
            labels[l[:,0], l[:,1]] = 1
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = torch.zeros((n, 28**2))
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            #img = img.reshape(28, 28)
            images[i, :] = torch.tensor(img)      
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)