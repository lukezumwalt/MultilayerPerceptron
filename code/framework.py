"""
Multi-Layered Perceptron.

Design to implement:
- ReLU
- Leaky ReLU

Trained on:
- MNIST, handwritten 0-9
"""
from typing import Callable
import numpy as np
import torch
import tqdm
# from torchvision import datasets, transforms
# from mnist import MnistDataloader
from mnist_parser import *
from activations import ReLU, LeakyReLU

class MLP:
    '''
    This class should implement a generic MLP learning framework.
    The core structure of the program has been provided for you.
    But, you need to complete the following functions:
    1: initialize()
    2: forward(), including activations
    3: backward(), including activations
    4: TrainMLP()
    '''
    def __init__(self, layer_sizes: list[int]):
        # Baseline architecture
        self.layer_sizes: list[int] = layer_sizes
        self.num_layers = len(layer_sizes) - 1

        # Storage for model Parameters
        self.weights: list[torch.Tensor] = []
        self.biases: list[torch.Tensor] = []

        # Storage for hidden-layer activations for backwards-propagation
        self.features: list[torch.Tensor] = []

        # Default model Hyper-Parameters
        self.learning_rate: float = 1
        self.batch_size: int = 1
        self.activation_function: Callable[[torch.Tensor], torch.Tensor] = ReLU

        # self.learning_rate: float = 1e-3
        # self.activation_function = ReLU  # or LeakyReLU, etc.


    def set_hp(self, lr: float, bs: int, activation: object) -> None:
        '''
        Defines Hyper-Parameters.
        '''
        self.learning_rate = lr
        self.batch_size = bs
        self.activation_function = activation

    def initialize(self) -> None:
        '''
        Initialize all biases to zero.
        Initialize all weights with random sampling from a uniform distribution.
        This uniform distribution should have range +/- sqrt(6 / (d_in + d_out))
        '''
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            d_in = self.layer_sizes[i]
            d_out = self.layer_sizes[i+1]
            # Range for uniform dist
            limit = np.sqrt(6.0 / (d_in + d_out))
            # Implemented below:
            # ((2*range*random_dist) - range) ∃ [-lim, lim]
            W = (torch.rand(d_in, d_out)*2*limit) - limit   # shape: [d_in, d_out]
            b = torch.zeros(d_out)                          # shape: [d_out]

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass through all layers. 
        x_i+1 = f(x_i*W + b)

        For the hidden layers, apply the chosen activation function.
        For the final layer, return raw logits (no activation).
        Stores the "post-activation" features at each layer for back-prop.
        '''
        # Clear out the old features each time
        self.features = []
        self.features.append(x)  # x^0 = input

        # Instantiate the activation neuron object (same function used for all hidden layers)
        act = self.activation_function()

        for i in range(self.num_layers):
            # z = x_{i} * W_{i} + b_{i}
            z = self.features[i] @ self.weights[i] + self.biases[i]

            if i < self.num_layers - 1:
                # Hidden layer => apply activation
                out = act.forward(z)
            else:
                # Final layer => raw logits, no activation
                out = z

            self.features.append(out)

        return self.features[-1]  # Return the final output (logits)
    
    def backward(self, delta: torch.Tensor) -> None:
        '''
        This function backpropagates the provided delta through the entire MLP, 
        and updates the weights according to the hyper-parameters
        stored in the class variables.
        '''
        # Helpful terms stored for re-use
        act = self.activation_function()
        L = self.num_layers

        # Backward layer by layer
        for i in reversed(range(L)):

            
            # current layer's output was self.features[i+1]
            # current layer's input  was self.features[i]
            if i < (L - 1):
                # We are at a hidden layer, so multiply by derivative of activation
                # The 'x' passed to backward(...) is the pre-activated input,
                # which here is the tensor before the ReLU clamp.  If you only
                # stored post-activation, you can still check where that is >0, etc.
                x_post_act = self.features[i+1]
                delta = act.backward(delta, x_post_act)

            x_in = self.features[i]

            # Weight gradient: dW = x_in^T @ delta
            dW = x_in.transpose(0,1) @ delta

            # Bias gradient: sum across batch dim
            db = torch.sum(delta,dim=0)

            # Average the batch
            dW /= self.batch_size
            db /= self.batch_size

            # Gradient descent updating
            self.weights[i] = self.weights[i] - self.learning_rate*dW
            self.biases[i] = self.biases[i] - self.learning_rate*db

            # Propagate delta to next layer (i-1)
            # ...only if i > 0
            if i > 0:
                delta = delta @ self.weights[i].transpose(0,1)

        return


def train_mlp(model: MLP, x_train: torch.Tensor, y_train: torch.Tensor) -> MLP:
    '''
    Train the MLP for 1 epoch using the provided data. 
    Uses random mini-batches of size model.batch_size.
    '''

    # Random sampling of data
    bs = model.batch_size
    N = x_train.shape[0]
    rng = np.random.default_rng()
    idx = rng.permutation(N)

    # Total Cross-Entropy Loss accumulated over the epoch
    L = 0.0

    # For each sample in the batch...
    for i in tqdm.tqdm(range(N // bs)):
        x = x_train[idx[i * bs:(i + 1) * bs], ...]
        y = y_train[idx[i * bs:(i + 1) * bs], ...]

        # Forward pass
        logits = model.forward(x)  # shape [bs, #classes]

        # Convert to probabilities
        p = torch.exp(logits)
        p /= torch.sum(p, dim=1, keepdim=True)  # shape [bs, #classes]

        # Cross-Entropy loss
        l = -torch.sum(y * torch.log(p + 1e-12))  # small epsilon for safety
        L += l.item()

        #backpropagate here

    trainingLoss = L / ((N // bs) * bs)
    print(f"Train Loss: {trainingLoss:.2f}" )


def test_mlp(model: MLP, x_test: torch.Tensor, y_test: torch.Tensor) -> tuple[float, float]:
    bs = model.batch_size
    N = x_test.shape[0]

    rng = np.random.default_rng()
    idx = rng.permutation(N)

    L = 0
    A = 0

    for i in tqdm.tqdm(range(N // bs)):
        x = x_test[idx[i * bs:(i + 1) * bs], ...]
        y = y_test[idx[i * bs:(i + 1) * bs], ...]

        y_hat = model.forward(x)
        p = torch.exp(y_hat)
        p /= torch.sum(p, dim = 1, keepdim = True)
        l = -1 * torch.sum(y * torch.log(p))
        L += l

        A += torch.sum(torch.where(torch.argmax(p, dim = 1) == torch.argmax(y, dim = 1), 1, 0))

    testLoss = L / ((N // bs) * bs)
    testAccuracy = 100 * A / ((N // bs) * bs)
    print(f"Test Loss: {testLoss}\nTest Accuracy: {testAccuracy:.2f}")

    return testLoss, testAccuracy

def normalize_mnist() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    This function loads the MNIST dataset, then normalizes 
    the "X" values to have zero mean, unit variance. 
    '''

    # Personal Path:
    base_path = "C:/Users/Luke/Courses/CSCI5922/Lab 1/MultilayerPerceptron/data/MNIST/"

    # Optional section providing access to the PyTorch data set.
    # Uncomment to use:
    ## trainset = datasets.MNIST('~/.pytorch/MNIST_data/', 
    ##                           download=True, 
    ##                           train=True, 
    ##                           transform=transforms.ToTensor())

    # Load files and obtain meta stat parameters
    (x_train, y_train) = load_and_preprocess_mnist(base_path + "train-images.idx3-ubyte",
                                                   base_path + "train-labels.idx1-ubyte")
    (x_test, y_test) = load_and_preprocess_mnist(base_path + "t10k-images.idx3-ubyte",
                                                 base_path + "t10k-labels.idx1-ubyte")
    x_mean = torch.mean(x_train, dim = 0, keepdim = True)
    x_std = torch.std(x_train, dim = 0, keepdim = True)

    # Normalize training data
    x_train -= x_mean
    x_train /= x_std

    # Normalize testing data
    x_test -= x_mean
    x_test /= x_std

    # Replace NaNs from dividing by zero
    x_train[x_train != x_train] = 0
    x_test[x_test != x_test] = 0

    return x_train, y_train, x_test, y_test

def main():
    '''
    This is an example of how to use the framework when completed.
    You can build off of this code to design your experiments for part 2.
    '''

    x_train, y_train, x_test, y_test = normalize_mnist()

    '''
    For the experiment, adjust the list [784,...,10] as desired to test other architectures.
    You are encouraged to play around with any of the following values if you so desire:
    E, lr, bs, activation
    '''

    model = MLP([784, 256, 10])
    model.initialize()
    model.set_hp(lr = 1e-6, bs = 512, activation = ReLU)

    E = 25
    for _ in range(E):
        train_mlp(model, x_train, y_train)
        test_mlp(model, x_test, y_test)

if __name__ == "__main__":
    main()