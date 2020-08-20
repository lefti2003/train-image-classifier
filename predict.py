#Program to predict the flower name given an image
# image to predict "flowers/train/1/image_06734.jpg"
import matplotlib.pyplot as plt
import numpy as np
import json
import math
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
from glob import glob
import os
#import helper
#import fc_model
import argparse

def SetParams():
    
  parser = argparse.ArgumentParser(description='An application to Build and Train a Network.')
  parser.add_argument("input", help="path to input image")
  parser.add_argument("checkpoint", const="checkpoint.pth", nargs='?', \
                      help="Load the checkpoint file into the model.")
  parser.add_argument("--top_k", const=5, default=5, nargs='?', \
                      type=int, help="top K probabilities.")
    
  parser.add_argument("--category_names", const="cat_to_name.json", default="cat_to_name.json", nargs='?',\
                      help="json category mapping file")

  parser.add_argument("--gpu" , help="cuda:0", action="store_true")
    
  

  args = parser.parse_args()
  input_img = args.input
  checkpoint = args.checkpoint
  top_k = args.top_k
  jsonfile = args.category_names
    
  if torch.cuda.is_available():
    if args.gpu:
        device = torch.device("cuda:0")
    else:
        
        device = torch.device("cpu")
  else:
        print('You have chosen --gpu but gpu is not enabled on your device.')
        device = "cpu"
  
  return input_img, checkpoint, top_k, jsonfile



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    model.arch = checkpoint['arch']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_dict'])

    for param in model.parameters():
        param.requires_grad = False
        
    return model


def process_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])])
    pil_image = Image.open(image)
    np_image = preprocess(pil_image)
    return np_image

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    image = process_image(image_path)
    model = load_checkpoint('checkpoint.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        image = image.to(device)
        image = image.unsqueeze(0)
        model.eval()
        output = model.forward(image)
        probabilities = torch.exp(output)
        top_k_probabilities, top_k_index = torch.topk(probabilities, topk)

        top_k_classes = list(map(lambda index: model.class_to_idx[index], np.array(top_k_index.cpu())[0]))

        top_k_probs = np.array(top_k_probabilities.cpu())[0]
        
    return top_k_probs, top_k_classes

def PrintFlower(img, model, jsonfile, topk):

    with open(jsonfile) as f:
      cat_to_name = json.load(f)
    image = process_image(img)
    probs, classes = predict(img, model, topk)
    new_probs = probs.flatten()
    names = [cat_to_name[i] for i in classes]
    
    return names

def truncate(n):
    return int(n*10000)/10000


img, checkpoint, top_k, jsonfile = SetParams()
model = load_checkpoint(checkpoint)

probs, classes = predict(img, model, top_k)

name = PrintFlower(img, model, jsonfile, top_k)
print("Flower:\t\t Probability:\t Percentage(%)")
print("-------\t\t ------------\t ----------")
for i in range(len(name)):
    prob_trunc = truncate(probs[i])
    prob_percent = round(100*prob_trunc,2)
    if prob_trunc != 0.0:
        if len(name[i]) >= 10:
            print("{}\t {}\t\t {}".format(name[i], prob_trunc, prob_percent))
        else:
            print("{}\t\t {}\t\t {}".format(name[i], prob_trunc, prob_percent))
    



