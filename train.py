# import statements and other necessary collections.
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import os
#import helper
#import fc_model
import argparse

# TODO: Define your transforms for the training, validation, and testing sets

def SetParams():
    
  parser = argparse.ArgumentParser(description='An application to Build and Train a Network.')
  parser.add_argument("data_dir", default="flowers",help="Directory of Data to process.")
  parser.add_argument("--arch" , const="vgg16", default="vgg16", nargs='?', type=str,\
                      help="architecture [densenet vgg].")
  parser.add_argument("--learning_rate", dest="lr", const=0.01, default=0.01, nargs='?', \
                      type=float, help="Learning rate for NN.")
  parser.add_argument("--hidden_units", dest="hu", const=64, default=64, nargs='?',\
                      type=int, help="Number of hidden layers.")
  parser.add_argument("--epochs", const=5, default=5, type=int, nargs='?', \
                      help="Number of times to run simulation.")
  parser.add_argument("--gpu" , help="cuda:0", action="store_true")
  parser.add_argument("--save_dir", const='./', default='./', nargs='?',\
                      help="Save the checkpoint in a specified directory.")
  args = parser.parse_args()
  if torch.cuda.is_available():
    if args.gpu:
        device = torch.device("cuda:0")
    else:
        
        device = torch.device("cpu")
  else:
        print('You have chosen --gpu but gpu is not enabled on your device.')
        device = "cpu"
        
  if args.save_dir:
    save_dir = args.save_dir
    print("save_dir={}".format(save_dir))

  return args.data_dir, args.arch, args.lr, args.hu, args.epochs, args.gpu, device, save_dir

def DefTransforms():
    print("Defining the train, valid, test Transforms")
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    
    return train_transforms, valid_transforms, test_transforms

# TODO: Load the datasets with ImageFolder
def LoadDataSets(data_dir, train_transforms, valid_transforms, test_transforms):
    if os.path.isdir(data_dir + '/train'):
        print("Loading the training data")
        train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    else:
        print(data_dir+'/train' + "directory does not exist")
    if os.path.isdir(data_dir + '/valid'):
        print("Loading the validating data")
        valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    else:
        print(data_dir+'/valid' + "directory does not exist")
    if os.path.isdir(data_dir + '/test'):
        print("Loading the test data")
        test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    else:
        print(data_dir+'/test' + "directory does not exist")
        
    return train_data, valid_data, test_data

# TODO: Using the image datasets and the trainforms, define the dataloaders
def DataLoaders(train_data, valid_data, test_data):
    print("Loading the trainloader, validloader and testloader")
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return trainloader, validloader, testloader



# TODO: Build and train your network
def Build(arch, hu, lr):
    print("Building the {} Network".format(arch))

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = getattr(models, arch)(pretrained=True)
    

    for param in model.parameters():
        param.requires_grad = False
#input layers are 224x224=50,176
#hidden layers = 64 (from original network from Part 1)
#output layers 64 x 102 classes
    model.classifier = nn.Sequential(nn.Linear(25088, hu),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hu, 102),
                                 nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    #
    #model
    #model.to(device);
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    return model, criterion, optimizer
    
def Train(device, model, criterion, lr, epochs, trainloader, validloader, optimizer):
    #optimizer = optim.Adam(model.classifier.parameters(), lr)
    model
    model.to(device);
    #epochs = 1
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
        # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)
                    
                        valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                    f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    print("finised!")
    
def SaveCheckPoint(train_data, model, optimizer, epochs, arch, hu, save_dir):
    
    class_to_idx = train_data.class_to_idx
    model.class_to_idx = { class_to_idx[k]: k for k in class_to_idx}

    checkpoint = {
        'input_size':25088,
        'output_size':102,
        'hidden_layers':hu,
        'optimizer': optimizer,
        'epoch': epochs,
        'arch': arch,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    
    checkpoint_file = save_dir + '/' + 'checkpoint.pth'
    torch.save(checkpoint, checkpoint_file)

    
def main():
    
  data_dir, arch, lr, hu, epochs, gpu, device, save_dir = SetParams()
  print("data_dir type={}".format(type(data_dir)))
  print("arch type={}".format(type(arch)))
  print("arch ={}".format(arch))
  train_tf, valid_tf, test_tf = DefTransforms()
  train_d, valid_d, test_d = LoadDataSets(data_dir, train_tf, valid_tf, test_tf)
  trainloader, validloader, testloader = DataLoaders(train_d, valid_d, test_d)
  model, criterion, optimizer = Build(arch, hu, lr)


  print("device ={}".format(device))
  
  Train(device, model, criterion, lr, epochs, trainloader, validloader, optimizer)

  if save_dir:
    SaveCheckPoint(train_d, model, optimizer, epochs, arch, hu, save_dir)  
  
  
  
if __name__== "__main__":
  main()   

