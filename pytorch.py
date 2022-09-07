from asyncio.windows_events import NULL
from random import shuffle
from re import L
from typing_extensions import Self
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import transforms, datasets
import torch.optim as optim
train = datasets.MNIST("", train=True, download = True, transform=transforms.Compose([transforms.ToTensor()])) #we have to convert the data to tensor


test = datasets.MNIST("", train=False, download = True, transform=transforms.Compose([transforms.ToTensor()])) #we have to convert the data to tensor

trainSet = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True) #deep learning works best with millions of samples, so need to use a smaller
testSet = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)   #batch size as we cant pass in everything at once | Shuffle is important as if your data is in order, it will optimize for one value before 
                                                                           #processing another value
#how to define layers

class Net(nn.Module):
    def __init__(self):
        super().__init__() #super corresponds to nn.Module, and init will just run that function in the module
        self.layer1 = nn.Linear(784, 64) #images when flattened are 784 since the picture is 28x28 
        self.layer2 = nn.Linear(64, 64)#our output will be our first hidden layer which is 64 neurons
        self.layer3 = nn.Linear(64, 64)#nn.Linear just means flatly conencted
        self.layer4 = nn.Linear(64, 10)
 

    def forward(self, layer): #this runs automatically i GUESS? 
        layer = F.relu(self.layer1(layer))
        layer = F.relu(self.layer2(layer))
        layer = F.relu(self.layer3(layer))
        layer = F.softmax(self.layer4(layer),dim=1) #converts a vector of numbers into a vector of probabilities
        return layer                               #dim=1 not sure what it means but typically use 1

    def trainNetwork(self):
       return 



network = Net()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=0.001) #adam is one of many

for epoch in range(3):
    for data in trainSet:
        x, y = data
        network.zero_grad() #sets gradients to zero before loss calculation so it doesn't accumulate
        output = network(x.view(-1, 784)) #pass in reshaped batch which is our input -> AUTOMATICALLY calls forward within the nn.Module __call__ but includes hooks which can be important?
        print(output)
        loss = F.nll_loss(output, y) #use nloss if not one hot output, use meansqurederror otherwise
        loss.backward() #Literal backpropagation so easy
        optimizer.step() #adjusts the weights for us 
        
correct = 0
total = 0

with torch.no_grad():#This is just for testing, we do not want to adjust
    for data in trainSet:
        x, y = data
        output = network(x.view(-1, 784))
        for idx, i in enumerate(output): 
            if torch.argmax(i) == y[idx]: #if y[0] == x[index Of maxValue] 
                correct += 1
            total += 1
print(f"Accuracy: ", round(correct/total, 3)) 

#3 values to 2 values

# test = nn.Linear(3,1,bias=False)
# print(test.weight) #i think it transposes the weights to make matmult work
# print("--------------------")
# input = torch.randn(1,3)
# print(input)
# print("--------------------")
# output = test(input)
# print(output)
