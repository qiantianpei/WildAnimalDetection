from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm

from WildAnimalDataset import WildAnimalDataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default='16', help='eval batch size')
parser.add_argument('--model1', type=str, default='densenet161', help='first model in the pipeline')
parser.add_argument('--model2', type=str, default='densenet161', help='second model in the pipeline')
parser.add_argument('--period', type=str, default='day', help='eval on day or night')
parser.add_argument('--split', type=str, default='dev', help='dataset split')

args = parser.parse_args()

data_transform = transforms.Compose([
        transforms.Resize((224,224)),
#         transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print('Preparing dataset...')
animal_dataset = WildAnimalDataset(csv_file = 'metadata/' + args.period + '_final_' + args.split +'.xlsx',
                                   oversample_list = None,
                                   oversample_times = 0,
                                   root_dir = 'data_' + args.period + '_final/' + args.split + '_signs', 
                                   transform = data_transform,
                                  prob = 'full')

print('Preparing data loader...')
dataloader = DataLoader(animal_dataset, args.batch_size, shuffle = False, num_workers = 4)
use_gpu = torch.cuda.is_available()

n_categories1 = 2
if args.period == 'day' or args.period == 'combined':
    n_categories2 = 23
if args.period == 'night':
    n_categories2 = 22

if args.model1 == 'densenet161':
    model1 = models.densenet161(pretrained=True)
    num_ftrs1 = model1.classifier.in_features
    model1.classifier = nn.Linear(num_ftrs1, n_categories1)
if args.model1 == 'resnet18':
    model1 = models.resnet18(pretrained=True)
    num_ftrs1 = model1.fc.in_features
    model1.fc = nn.Linear(num_ftrs1, n_categories1)

if args.model2 == 'densenet161':
    model2 = models.densenet161(pretrained=True)
    num_ftrs2 = model2.classifier.in_features
    model2.classifier = nn.Linear(num_ftrs2, n_categories2)
    #model2.fc = nn.Linear(num_ftrs2, 22)
if args.model2 == 'resnet18':
    model2 = models.resnet18(pretrained=True)
    num_ftrs2 = model2.fc.in_features
    model2.fc = nn.Linear(num_ftrs2, n_categories2)
    

model1.train(False)
model2.train(False)

if use_gpu:
    model1 = model1.cuda()
    model2 = model2.cuda()

print('Loading model1...')
model1.load_state_dict(torch.load('model_params/' + args.period + '_model1_' + args.model1 + '/best_model.pt'))
print('Loading model2...')
model2.load_state_dict(torch.load('model_params/' + args.period + '_model2_' + args.model2 + '/best_model.pt'))

corrects_pipeline = 0
corrects_model1 = 0
corrects_model2 = 0
contains = 0
dataset_size = 0

print('Evaluating on ' + args.period + '...')
for data in dataloader:
    inputs, labels = data
    labels = labels.view(-1)
    if use_gpu:
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)
    outputs1 = model1(inputs) # output: [N, 2]
    outputs2 = model2(inputs)
    _, preds1 = torch.max(outputs1.data, 1) # preds: [N]
    _, preds2 = torch.max(outputs2.data, 1) 
    preds2 += 1
    
    corrects_pipeline += torch.sum((preds1 == 0) * (labels.data == preds1) + (preds1 == 1) * (labels.data == preds2))
    corrects_model1 += torch.sum((preds1 == 0) * (labels.data == preds1) + (preds1 == 1) * (labels.data != 0))
    corrects_model2 += torch.sum((labels.data == preds2))
    contains += torch.sum(labels.data != 0)
    dataset_size += labels.size()[0]
    
print('Pipeline(' + args.model1 + '+' + args.model2 + '): {:.2f}%'.format(100 * corrects_pipeline / dataset_size) )
print('Model1(' + args.model1 + '): {:.2f}%'.format(100 * corrects_model1 / dataset_size))
print('Model2(' + args.model2 + '): {:.2f}%'.format(100 * corrects_model2 / contains))