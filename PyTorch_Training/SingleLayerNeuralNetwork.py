# -*- coding: utf-8 -*-
"""
Lesson One Part One
Created on Thu Sep 10 14:21:10 2020

@author: Alycia N. Carey
"""

import torch 

def activation(x):
    """Sigmoid activation function
    
        Arguments
        ---------
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))

#### Finding output for one node ####

### Generating Data
#torch.manual_seed(7)                    # Set the random seed so things are predictable
#features = torch.randn((1,5))           # Features are 5 random normal variables. Creates a tensor with shape (1,5), or 1 row 5 col. 
#weights = torch.randn_like(features)    # True weights for our data, random normal variables again
#bias = torch.randn((1,1))               # and a true bias term. Creates a single value from a normal distribution

### Calculate the output of this network using the weights and bias tensors
#weights = weights.view(5,1)             # resizing the weights tensor so we can multiply against features
#print(weights.shape)                    # checking the size of weights
#fw = torch.mm(features, weights)        # matrix multiplication of the features and the weights
#fwb = fw + bias

### Calculate the output of this network using matrix multiplication
#output = activation(fwb)
#print(output)

#### Finding output for multiple nodes ####

### Generate some data
torch.manual_seed(7)                    # Set the random seed so things are predictable
features = torch.randn((1,3))           # Features are 3 random normal variables

#Define the size of each layer in our network
n_input = features.shape[1]             # Number of input units, must match number of input features
n_hidden = 2                            # Number of hidden units
n_output = 1                            # Number of output units

W1 = torch.randn(n_input, n_hidden)     # Weights for inputs to hidden layer
W2 = torch.randn(n_hidden, n_output)    # Weights for hidden layer to output layer

B1 = torch.randn((1, n_hidden))         # bias terms for hidden layer
B2 = torch.randn((1, n_output))         # biad terms for output layers

###Calculate the output of the network using the weights and bias tensors
fw1 = torch.mm(features, W1) + B1
activation_one = activation(fw1)        # activate(torch.mm(features,W1) + B1) is a shorter way to write this

hidden = torch.mm(activation_one, W2) + B2 # activate(torch.mm(activation_one,W2) + B2) is a shorter way to write this
output = activation(hidden)

print(output)
