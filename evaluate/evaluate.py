import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
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

def evaluate(model, dataloaders, use_gpu, topN, threshold):
    class_names = ["None", "Black-tailed deer",         
               "Human", "Black-tailed Jackrabbit",
               "Coyote", "Unidentified bird",         
               "Brush Rabbit", "Western scrub-jay",
               "Bobcat", "Other",
               "Unidentified mammal", "California quail",
               "Raccoon", "Mountain lion",
               "Striped skunk", "Wild turkey",
               "Gray Fox", "Virginia Opossum",
               "Stellers jay", "Western Gray Squirrel",
               "Dusky-footed Woodrat", "Great blue heron",
               "Fox Squirrel", "California Ground Squirrel"]
    model.train(False)  # Set model to evaluate mode
    running_corrects = 0
    num_pred = 0
    num_unpred = 0
    # Iterate over data.
    for data in dataloaders['val']:
        # get the inputs
        inputs, labels = data # inputs: [N, C, H, W], labels: [N]
        labels = labels.view(-1,1)
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # forward
        m = nn.Softmax(dim = 1)
        outputs = m(model(inputs)) # output: [N, 2]
        outputs = (outputs.data).cpu().numpy()
        preds = np.argsort(outputs, 1)[:,-topN:]
        
        for i in range(preds.shape[0]):
            label = (labels.data).cpu().numpy()[i,0]
#             if preds[i,-1] != label:
#                 print("True label: " + class_names[label])
#                 info = "Predicted Top {}: ".format(topN)
#                 for j in range(1, topN + 1):
#                     info += "{}. {} ({:.4f}) ".format(j, class_names[preds[i,-j]], outputs[i,preds[i,-j]])
#                 print(info)
#                 print()
            conf = outputs[i,preds[i,-1]]
    
#             print(outputs[i,:])
            
            if conf > threshold:
                #Unidf mammal = 10, unidf bird = 5
#                 if preds[i, -1] == 5 or preds[i, -1]==10:
#                     continue
#                 else:
                running_corrects += (preds[i,-1] == label)
                num_pred += 1
            else:
                num_unpred += 1
            
        #running_corrects += np.sum(preds == (labels.data).cpu().numpy())

    #epoch_acc = running_corrects / dataset_sizes['val']
    epoch_acc = running_corrects / num_pred
    frac_pred = num_pred / (num_pred + num_unpred)

    print('Acc: {:.4f}'.format(epoch_acc))
    print('Frac of data predicted: {:.4f}'.format(frac_pred))
