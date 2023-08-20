import torch.nn as nn


NONLINEARITIES = {
    'tanh': nn.Tanh(),
    'softplus': nn.Softplus(),
    'relu': nn.ReLU(),
}
