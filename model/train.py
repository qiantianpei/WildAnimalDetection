from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
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
from multiClassHingeLoss import multiClassHingeLoss
from trainer import trainer
from evaluate import evaluate

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--prob', type=str, default = 'full', help='classification problem: full/model1(contains or not)/model2(species)/model3(major)/model4(rare)')
parser.add_argument('--model', type=str, default='densenet161', help='model type')
parser.add_argument('--period', type=str, default='combined', help='train on: day/night/combined')
parser.add_argument('--epoch', type=int, default=30, help='number of training epochs')
parser.add_argument('--topN', type=int, default=0, help='Evaluate top N results')
parser.add_argument('--norm', type=int, default=1, help='mean and std used for normalization')
parser.add_argument('--threshold', type=float, default=0, help='Confidence threshold for prediction')
parser.add_argument('--optimizer', type=str, default="SGD", help='optimizer choice')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lrdecay', type=int, default=1, help='learning rate decay')
parser.add_argument('--sample_list', type=list, default=None, help='oversample list')
parser.add_argument('--sample_N', type=int, default=0, help='number of times to oversample')
parser.add_argument('--loss', type=str, default='CE', help='loss type')

args = parser.parse_args()
print("learning rate: " + str(args.lr))

# mean and std from pytorch
if args.norm == 0:
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
# mean and std computed from training set 
if args.norm == 1:
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.31293, 0.30621,0.28194], [0.18229829,0.18250624,0.1759812])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.31293, 0.30621,0.28194], [0.18229829,0.18250624,0.1759812])
        ]),
    }
    
if args.norm == 2:
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ]),
    }
if args.norm == 3:
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.31293, 0.30621,0.28194], [0.18229829,0.18250624,0.1759812])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.31293, 0.30621,0.28194], [0.18229829,0.18250624,0.1759812])
        ]),
    }
if args.model == 'inceptionv3':
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.31293, 0.30621,0.28194], [0.18229829,0.18250624,0.1759812])
        ]),
        'val': transforms.Compose([
            transforms.Resize((299,299)),
            transforms.ToTensor(),
            transforms.Normalize([0.31293, 0.30621,0.28194], [0.18229829,0.18250624,0.1759812])
        ]),
    }


if args.prob in ['full', 'model1', 'model3', 'model4']:
    data_type = 'final'
if args.prob == 'model2':
    data_type = 'contain'

train_data_dir = 'data_' + args.period + '_' + data_type + '/train_signs'
dev_data_dir = 'data_' + args.period + '_' + data_type + '/dev_signs'

if args.prob == 'model2' and args.period == 'combined':
    train_data_dir = 'data_' + args.period + '_' + 'final/train_signs'
    dev_data_dir = 'data_' + args.period + '_' + 'final/dev_signs'
     
train_meta_dir = 'metadata/' + args.period + '_' + data_type + '_train.xlsx'
dev_meta_dir = 'metadata/' + args.period + '_' + data_type + '_dev.xlsx'

if args.prob == 'model3':
    train_meta_dir = 'metadata/combined_final_train_rareCombined.xlsx'
    dev_meta_dir = 'metadata/combined_final_dev_rareCombined.xlsx'
if args.prob == 'model4':
    train_meta_dir = 'metadata/combined_final_train_rare.xlsx'
    dev_meta_dir = 'metadata/combined_final_dev_rare.xlsx' 
    
   
print('Preparing dataset...')
animal_dataset_train = WildAnimalDataset(csv_file = train_meta_dir, 
                                         oversample_list = args.sample_list,
                                         oversample_times = args.sample_N,
                                         root_dir = train_data_dir, 
                                         transform = data_transforms['train'],
                                         prob = args.prob,
                                         merged_categories = False,
                                        norm = args.norm)

animal_dataset_val = WildAnimalDataset(csv_file = dev_meta_dir, 
                                       oversample_list = None,
                                       oversample_times = 0,
                                       root_dir = dev_data_dir, 
                                       transform = data_transforms['val'],
                                       prob = args.prob, 
                                       merged_categories = False,
                                      norm = args.norm)

print('Preparing data loader...')
dataloader_train = DataLoader(animal_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
dataloader_val = DataLoader(animal_dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4)
dataloaders = {'train':dataloader_train, 'val':dataloader_val}

use_gpu = torch.cuda.is_available()
if args.norm == 0:
    checkpoint = 'model_params/' + args.period + '_' + args.prob + '_' + args.model
elif args.norm == 1:
    checkpoint = 'model_params/' + args.period + '_' + args.prob + '_' + args.model + '_normUp'
elif args.norm == 3:
    checkpoint = 'model_params/' + args.period + '_' + args.prob + '_' + args.model + '_norm3'
else:
    checkpoint = 'model_params/' + args.period + '_' + args.prob + '_' + args.model + '_normMeanImage'

if args.optimizer == 'Adam':
    checkpoint += '_Adam'
if args.optimizer == 'Nesterov':
    checkpoint += '_Nesterov'
if args.lr != 1e-3:
    checkpoint += '_' + str(args.lr)
if args.lrdecay == 1:
    checkpoint += '_lrdecay'
if args.lrdecay == 2:
    checkpoint += '_lrdecay2'
if args.sample_list != None:
    checkpoint += '_oslist'+''.join(str(e) for e in args.sample_list)
if args.sample_N != 0:
    checkpoint += '_oversample' + str(args.sample_N)


if args.prob == 'full':
    n_categories = 24
    if args.period == 'night':
        n_categories -= 1
if args.prob == 'model1':
    n_categories = 2
if args.prob == 'model2':
    n_categories = 23
    if args.period == 'night':
        n_categories -= 1
if args.prob == 'model3':
    n_categories = 4
if args.prob == 'model4':
    n_categories = 21

if args.model == 'densenet161':
    model = models.densenet161(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, n_categories)
if args.model == 'resnet18':
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_categories)
if args.model == 'resnet152':
    model = models.resnet152(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_categories)
if args.model == 'inceptionv3':
    model = models.inception_v3(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_categories)
    

if use_gpu:
    model = model.cuda()

if args.loss == 'CE':
    criterion = nn.CrossEntropyLoss()
if args.loss == 'SVM':
    criterion = multiClassHingeLoss()
    checkpoint += 'SVMloss'

if args.optimizer == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
if args.optimizer == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
if args.optimizer == "Nesterov":
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    
if args.lrdecay == 1:
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) # Decay LR by a factor of 0.5 every 5 epochs
elif args.lrdecay == 2:
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
elif args.lrdecay == 3:
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)
    
if not os.path.isdir(checkpoint):
    os.makedirs(checkpoint)

if os.path.isfile(checkpoint + '/best_model.pt'):
    print('loaded')
    model.load_state_dict(torch.load(checkpoint + '/best_model.pt'))

# if args.model == 'inceptionv3':
#     model.fc = nn.Linear(num_ftrs, n_categories)
#     self.aux_logits = False
    
if args.topN == 0:  
    print('Counting data sizes...')
    dataset_sizes = {'train': len(pd.io.excel.read_excel(train_meta_dir)), 
                 'val':len(pd.io.excel.read_excel(dev_meta_dir))}
    print('Training starts')
    model = trainer(model, dataloaders, criterion, optimizer, exp_lr_scheduler, checkpoint, use_gpu, dataset_sizes, args.epoch)

else:
    print('Evaluating...')
    evaluate(model, dataloaders, use_gpu, args.topN, args.threshold)