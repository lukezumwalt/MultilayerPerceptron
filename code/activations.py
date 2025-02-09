import torch

class ReLU():
    #Complete this class
    def forward(x: torch.tensor) -> torch.tensor:
        #implement ReLU(x) here
    
    def backward(delta: torch.tensor, x: torch.tensor) -> torch.tensor:
        #implement delta * ReLU'(x) here

class LeakyReLU():
    #Complete this class
    def forward(x: torch.tensor) -> torch.tensor:
        #implement LeakyReLU(x) here
    
    def backward(delta: torch.tensor, x: torch.tensor) -> torch.tensor:
        #implement delta * LeakyReLU'(x) here