# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:36:12 2020

@author: - Alycia N. Carey
"""

import numpy as np
import torch
import helper
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

def activation(x):
    """Sigmoid activation function
    
        Arguments
        ---------
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1,1)

# define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

# download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')

# flatten the batch of images 
inputs = images.view(images.shape[0], -1)
print(images.shape)

# build a multi-layer network with 784 input units, 256 hidden units, and 10 output units
# using random tensors for weights and biases, use sigmoid for activation for the hidden layer.
# leave the output layer without an activation
W1 = torch.randn(784, 256)
W2 = torch.randn(256, 10)

B1 = torch.randn(256)
B2 = torch.randn(10)

hidden = activation(torch.mm(inputs, W1) + B1)

out = torch.mm(hidden, W2) + B2
probabilities = softmax(out)

print(probabilities.shape)
print(probabilities.sum(dim=1))