# Without modification for combined categories and adjusted loss
# on dev set only
# random crop not turned off yet
# haven't check logic for day/night yet

from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision import datasets, models, transforms

from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

from WildAnimalDataset import WildAnimalDataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--photo_path', type=str, default='/data/cs341-bucket/camera_B2', help='photo path')
parser.add_argument('--meta_path', type=str, default='/data/cs341-bucket/camera_B2/camera_B2.xlsx', help='metadata path')

# parser.add_argument('--photo_path', type=str, default='/data/data_combined_final/dev_signs', help='photo path')
# parser.add_argument('--meta_path', type=str, default='/data/metadata/combined_final_dev.xlsx', help='metadata path')
parser.add_argument('--threshold', type=float, default=0, help='threshold for settling prediction')


def obtain_inputs_parameters(args):

    meta_path = args.meta_path
    photo_path = args.photo_path

    data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.31293, 0.30621,0.28194], [0.18229829,0.18250624,0.1759812])
    ])
        
    animal_dataset = WildAnimalDataset(meta_path, photo_path, data_transform) 
    dataloader = DataLoader(animal_dataset, args.batch_size, shuffle=False, num_workers=4)
    return dataloader

def read_trained_single_model(args, use_gpu, n_total_categories):
        
    model = models.densenet161(pretrained = True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, n_total_categories)
    model.train(False)
    
    if use_gpu:
        model = model.cuda()
    model_checkpoint_path = 'best_model.pt'
    print("model_checkpoint_path: " + model_checkpoint_path)
    model.load_state_dict(torch.load(model_checkpoint_path))
    return model

def create_single_model_confusion(model, n_total_categories, dataloader, use_gpu, args):
    
    confusion = torch.zeros(n_total_categories, n_total_categories)
    model.train(False) 

    for data in tqdm(dataloader):
        inputs, labels = data 
        labels = labels.view(-1)
        
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs) 
        _, preds = torch.max(outputs.data, 1)

        for i, j in zip(preds, labels.data):
            confusion[i,j] += 1
    return confusion

    
def write_pred(args, model, dataloader, use_gpu):
    meta = pd.io.excel.read_excel(args.meta_path)
    
    pred_settled = pd.DataFrame(columns = list(meta.columns.values) + ['pred'])
    pred_unsettled = pd.DataFrame(columns = list(meta.columns.values) + ['pred'])
    settled = 0
    unsettled = 0
    
    model.train(False) 
    
    for inputs in tqdm(dataloader):
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        m = nn.Softmax(dim = 1)
        outputs = m(model(inputs)) 
        outputs = (outputs.data).cpu().numpy()
        preds = np.argsort(outputs, 1)[:,-1:]

        for i in range(preds.shape[0]):
            curr = meta.loc[settled + unsettled]
            curr['pred'] = preds[i,-1]

            conf = outputs[i, preds[i,-1]]           
            if conf > args.threshold:
                settled += 1
                pred_settled.loc[settled] = curr 
            else:
                unsettled += 1
                pred_unsettled.loc[unsettled] = curr
                
    pred_settled.to_csv('settled.csv', index = False)
    pred_unsettled.to_csv('unsettled.csv', index = False)
    
    
def main():
    args = parser.parse_args()
    n_total_categories = 24
    use_gpu = torch.cuda.is_available()
    class_names = ["None", "Black-tailed deer", "Human", "Black-tailed Jackrabbit", "Coyote", 
                   "Big Category", "Brush Rabbit", "Western scrub-jay","Bobcat", "blank1","blank2", 
                   "California quail","Raccoon", "Mountain lion","Striped skunk", "Wild turkey", 
                   "Gray Fox", "Virginia Opossum", "Stellers jay", "Western Gray Squirrel", 
                   "Dusky-footed Woodrat", "Great blue heron", "Fox Squirrel", "California Ground Squirrel"]
    
    dataloader = obtain_inputs_parameters(args)
    model = read_trained_single_model(args, use_gpu, n_total_categories)
    
    write_pred(args, model, dataloader, use_gpu)

if __name__ == "__main__":
    main()