"""
Module defining artificial neuron activation functions.
Currently implemented:
    - ReLU
    - Leaky ReLU
"""
import torch

class ReLU():
    '''
    ReLU (Rectified Linear Unit) Function Class
    Converts any input signal to a nonlinear output.
    Any negative inputs are zeroed out.
    '''
    def __init__(self):
        pass

    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        ReLU(x) = if x>0 return x, else 0.0
        '''
        return torch.where(x > 0, x, 0.0)

    def backward(self, delta: torch.tensor, x: torch.tensor) -> torch.tensor:
        '''
        Derivative of ReLU is 1 if x>0, else 0.
        We multiply 'delta' by that mask.
        '''
        gradient_mask = torch.where(x > 0,                  # if x > 0
                                    torch.ones_like(x),     # then x_i = 1.0
                                    torch.zeros_like(x))    # else x_i = 0.0
        return delta * gradient_mask


class LeakyReLU():
    '''
    Leaky ReLU (Rectified Linear Unit) Function Class
    Leaky implies it allows a small percentage of failed inputs through.
    This model is designed at a rate of 10% negative leak.
    '''
    def __init__(self):
        pass

    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        LeakyReLU(x) = if x>0 return x, else 0.1*x
        '''
        return torch.where(x > 0, x, 0.1*x)

    def backward(self, delta: torch.tensor, x: torch.tensor) -> torch.tensor:
        '''
        Derivative of LeakyReLU is 1 if x>0, else 0.1.
        We multiply 'delta' by that factor.
        '''
        gradient_mask = torch.where(x > 0,                  # if x > 0
                                    torch.ones_like(x),     # then x_i = 1.0
                                    0.1*torch.ones_like(x)) # else x_i = 0.1
        return delta * gradient_mask
    