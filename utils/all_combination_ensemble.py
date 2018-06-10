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
import itertools

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

true_label_df = pd.read_csv("combined_final_dev_true_labels.csv")
full_model_list = os.listdir("predictions")
full_model_list.remove('.ipynb_checkpoints')
full_model_list.remove('pipeline_individual_model')
max_num_element = len(full_model_list)
true_labels = np.array(true_label_df["True_Label"].tolist())
result = []
for num_model in range(1, 7):
    for single_combination in tqdm(itertools.combinations(full_model_list, num_model)):
        for index in range(num_model):
            temp_probs = pd.read_csv(os.path.join("predictions", single_combination[index]))
            assert temp_probs.shape == (11487, 25)
            assert temp_probs["Img_Names"].tolist() == true_label_df["Img_Names"].tolist()
            if index == 0:
                probs = temp_probs.iloc[:,1:25]
                name = single_combination[index].split(".")[0]
            else:
                probs = probs + temp_probs.iloc[:,1:25]
                name = name + "_" + single_combination[index].split(".")[0]
        probs = probs.as_matrix()
        predictions = np.array(list(np.argmax(probs, axis=1)))
        total_accuracy = sum(true_labels == predictions)/len(predictions)
        result.append(tuple([name, num_model, total_accuracy]))
        #print(name + " Total Accuracy: " + str(sum(true_labels == predictions)/len(predictions)))  

Ensemble_Result = pd.DataFrame.from_records(result, columns = ["Model", "Num_Model", "Accuracy"])
Ensemble_Result.to_csv('Ensemble_Result.csv', index = False)    
