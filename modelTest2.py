#import libraries
import numpy as np
import torch
import torchvision   
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable

if __name__ == '__main__':

    #define file paths
    train_path = 'C:/Users/adity/OneDrive/Documents/GitHub/dog-cat-full-dataset/data/lidarTest5' #fill in path to training data
    val_path = 'C:/Users/adity/OneDrive/Documents/GitHub/dog-cat-full-dataset/data/test' #fill in path to validation data

    #create dataset and dataloader
    #below is an example using the torchVision ImageFolder class
    #to use this, the path you pass in should be to a folder that contains folders of images separated by class 
    #   i.e one folder for images with clogs, one folder of images without clogs
    #alternatively, create a custom pytorch dataset class

    batch_size = 10 #hyperparameter you can modify!
    workers = 1 #number of processes running in parallel -- look up/ask me abt GPUs in Colab if you want to increase this number (can make training faster!)

    # define transforms
    # img_size = 224
    # if True:
    #     train_transform = transforms.Compose([
    #         transforms.Resize(img_size),
    #         transforms.RandomHorizontalFlip(0.3),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # else:
    #     train_transform = transforms.Compose([
    #         transforms.Resize(img_size),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])

    train_dataset = torchvision.datasets.ImageFolder(root=train_path, 
                                                    transform=torchvision.transforms.ToTensor())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                shuffle=True, num_workers=workers)

    val_dataset = torchvision.datasets.ImageFolder(root=val_path, 
                                                transform=torchvision.transforms.ToTensor())
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                                shuffle=True, num_workers=workers)

    #define model

    class Network(nn.Module):
        def __init__(self, input_channels, num_classes):
            super(Network, self).__init__()
            print ("hello")
            #create layers for model! 
            #here's an example of 1 convolutional layer follwed by 1 linear layer

            #convolutional layer
            #input_channels: dimension of each element of input. example: if you have an RBG image, input_channels=3
            #num_filters: number of output channels for this layer. same as number of 'filters' that are scanned over the data
            num_filters = 64
            self.layer1 = nn.Conv2d(input_channels, num_filters, kernel_size=3, stride=1, padding=1) #experiment with kernel_size and stride!

            # We'll apply max pooling with a kernel size of 2
            # self.pool = nn.MaxPool2d(kernel_size=2)
            
            # # A second convolutional layer takes 12 input channels, and generates 12 outputs
            # self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
            
            # # A third convolutional layer takes 12 inputs and generates 24 outputs
            # self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
            
            # # A drop layer deletes 20% of the features to help prevent overfitting
            # self.drop = nn.Dropout2d(p=0.2)

            #linear layer
            #this example produces an output of size num_classes, so the first element of the output corresponds 
            #to how likely it is that the output falls into the first class
            self.layer2 = nn.Linear(num_filters, num_classes)  #might have to modify dimensions 

            print ("hi again")

        def forward(self, x):
            output = self.layer1(x)
            output = self.layer2(output)

            print ("hi")
        
    print("CNN model class defined!")

    def train(model, data_loader, val_data_loader):
        #loop over epochs
        for epoch in range(num_epochs):
            print ("wat up")
            #loop over batches of data
            for batch_num, (data, target) in enumerate(data_loader):
            #1. pass input through model and get output
            #2. pass output and labels to loss function (criterion) to get loss
            #2.x print loss every so often to see how you're doing!
            #3. step with optimizer to update weights
                device = "cpu"
        #pass validation data through model and calculate validation loss 
                data, target = data.to(device), target.to(device)
                
                print ("Before reset optimizer")
                # Reset the optimizer
                optimizer.zero_grad()
                
                
                print ("Before push data")
                # Push the data forward through the model layers
                output = model(data)
                
                
                print ("Before get the loss")
                # Get the loss
                loss = loss_criteria(output, target)
                
                
                print ("Before keep total")
                # Keep a running total
                train_loss += loss.item()
                
                # Backpropagate
                loss.backward()
                optimizer.step()
                
            print ("shelly")

    def test(model, data_loader):
        #similar to train, but only pass data through model once
        #also can't calculate loss and update weights since you don't know real answers!
        #pass validation data through model and calculate validation loss 
        data, target = data.to(device), target.to(device)
        
        # Reset the optimizer
        optimizer.zero_grad()
        
        # Push the data forward through the model layers
        output = model(data)

    print ("It worked!")

    num_epochs = 12
    input_channels = 3
    num_classes = 2

    #create model by calling class we defined above 
    model = Network(input_channels, num_classes)
    print ("hi its ADi")
    #create criterion (loss function, I suggest CrossEntropyLoss)
    criterion = nn.CrossEntropyLoss()
    print ("its me again")
    #create optimizer (this updates weights automatically according to loss, I suggest SGD or Adam)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print ("sorry to bug you")
    #call training function to train model
    model.train() #sets model to training mode
    print ("ayo")
    train(model, train_dataloader, val_dataloader)

    print ("Model has been trained")