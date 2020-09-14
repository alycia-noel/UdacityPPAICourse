"""Better solution to problem"""

from torch import nn
import torch
import torch.nn.functional as F  
import numpy as np
import helper
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)

print(model[0])
model[0].weight

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

# download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0,:])
helper.view_classify(images[0].view(1, 28, 28), ps)
#from torch import nn

# class Network(nn.Module):                                       
#     def __init__(self):
#         super().__init__()
        
#         #Inputs to hidden layer linear transformation
#         self.hidden = nn.Linear(784, 256)
        
#         #Output layer, 10 units - one for each digit
#         self.output = nn.Linear(256,10)
        
#         #Define sigmoid activation and softmax output
#         self.sigmoid = nn.Sigmoid()
#         self.softmax == nn.Softmax(dim=1)
        
#     def forward(self,x):
#         #Pass the input tensor through each of our operations
#         x = self.hidden(x)
#         x = self.sigmoid(x)
#         x = self.output(x)
#         x = self.softmax(x)
        
#         return x

"""A More succinct way"""
# from torch import nn
# import torch.nn.functional as F

# class Network(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         #Inputs to hidden layer linear transformation
#         self.hidden = nn.Linear(784,256)
        
#         #Output layer, 10 units - one for each digit
#         self.output = nn.Linear(256, 10)
        
#     def forward(self, x):
#         #Hidden layer with sigmoid activation
#         x = F.sigmoid(self.hidden(x))
        
#         #Output layer with softmax activation
#         x = F.softmax(self.output(x), dim=1)
        
#         return x
 
# """Solution to problem"""
# from torch import nn
# import torch
# import torch.nn.functional as F  
# import numpy as np
# import helper
# import matplotlib.pyplot as plt

# from torchvision import datasets, transforms

# class Network(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         #define the layers
#         self.fc1 = nn.Linear(784,128)
#         self.fc2 = nn.Linear(128,64)
#         self.fc3 = nn.Linear(64, 10)
        
#     def forward(self,x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         x = F.softmax(x, dim=1)
        
#         return x

# model = Network()
# print(model)

# print(model.fc1.weight)
# print(model.fc1.bias)

# #set biases to all zeros
# model.fc1.bias.data.fill_(0)

# #sample from random normal with standard dev = 0.01
# model.fc1.weight.data.normal_(std=0.01)

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

# # download and load the training data
# trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# #Resize the images into a 1d vector, new shape is (batch size, color channels, image pixels)
# images.resize_(64, 1, 784)

# #Forward pass through the network
# img_idx = 0
# ps = model.forward(images[img_idx,:])

# img = images[img_idx]
# helper.view_classify(img.view(1, 28, 28), ps)