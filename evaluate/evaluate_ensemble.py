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

from WildAnimalDataset_Ensemble import WildAnimalDataset
from trainer import trainer


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--period', type=str, default='combined', help='train on: day/night/combined/day_plus_night')
parser.add_argument('--norm', type=int, default=0, help='mean and std used for normalization')
args = parser.parse_args()


# def obtain_inputs_parameters(args):

#     csv_path = 'metadata/' + args.period + '_final_' + 'dev.xlsx'
#     data_path = 'data_' + args.period + '_final/' + 'dev_signs'
    
#     if args.norm == 0:
#         data_transform = transforms.Compose([
#             transforms.Resize((224,224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
    
#     if args.norm == 1:
#         data_transform = transforms.Compose([
#             transforms.Resize((224,224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.31293, 0.30621,0.28194], [0.1965, 0.1965, 0.19075])
#         ])
        
    
#     animal_dataset = WildAnimalDataset(csv_file= csv_path, root_dir= data_path, transform = data_transform, merged_categories = False, prob = "full") # always use full to make sure the correct true labels are extracted
#     dataloader = DataLoader(animal_dataset, args.batch_size, shuffle=False, num_workers=4)
    
#     return dataloader
    


# def obtain_true_labels(dataloader):
    
#     full_labels = []
#     img_name_list = []

#     for data in tqdm(dataloader):
#         inputs, labels, img_names = data 
#         labels = list(labels.view(-1).numpy())
#         img_names =  list(img_names)
#         full_labels += labels
#         img_name_list += img_names 
    
#     df = pd.DataFrame({'Img_Name': img_name_list, 'True_Label': full_labels})
#     df.to_csv("combined_final_dev_true_labels.csv", index = False)
#     return full_labels

    
    
def main():
    true_label_df = pd.read_csv("combined_final_dev_true_labels.csv")
    
    junk1 = pd.read_csv("predictions/combined_full_densenet161.csv")
    junk2 = pd.read_csv("predictions/combined_full_resnet152_normUp.csv")
    junk3 = pd.read_csv("predictions/combined_pipeline1_densenet161.csv")
    junk4 = pd.read_csv("predictions/day_plus_night_full_densenet161.csv")
    
    assert junk1.shape == junk2.shape == junk3.shape == junk4.shape == (11487, 25)
    assert junk1["Img_Names"].tolist() == junk2["Img_Names"].tolist() == junk3["Img_Names"].tolist() == junk4["Img_Names"].tolist() == true_label_df["Img_Names"].tolist()
    junk = junk1.iloc[:,1:25] + junk2.iloc[:,1:25] + junk3.iloc[:,1:25] + junk4.iloc[:,1:25]
    junk = junk.as_matrix()
    predictions = np.array(list(np.argmax(junk, axis=1)))
    
    true_labels = np.array(true_label_df["True_Label"].tolist())
    
#     if args.period == "day_plus_night": # difference between combined vs day_plus_night?
#         pass
#     else:
#         dataloader = obtain_inputs_parameters(args)
#         true_labels = np.array(obtain_true_labels(dataloader))
        
        
    print("Total Accuracy: " + str(sum(true_labels == predictions)/len(predictions)))

#-------------------------------
if __name__ == "__main__":
    main()