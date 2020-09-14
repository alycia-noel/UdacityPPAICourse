import torch
from torch import nn
import helper
from torchvision import datasets, transforms
from torch import optim

#Defina a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)),])

#Download the load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle=True )

#Build a feed-forward network
model = nn.Sequential(nn.Linear(784,128),
                      nn.ReLU(),
                      nn.Linear(128,64),
                      nn.ReLU(),
                      nn.Linear(64,10),
                      nn.LogSoftmax(dim=1))

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01)

#Define the loss
criterion = nn.NLLLoss()

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        #flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
helper.view_classify(img.view(1, 28, 28), ps)



# #Get our data
# images, labels = next(iter(trainloader))

# #Flatten images
# images = images.view(images.shape[0], -1)

# #Forward pass, get our logits
# output = model(images)

# #Calculate the loss with the logits and the labels
# loss = criterion(output, labels)

# print('Before backward pass: \n', model[0].weight.grad)

# loss.backward()

# print('After backward pass: \n', model[0].weight.grad)