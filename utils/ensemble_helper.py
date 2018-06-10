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
parser.add_argument('--prob', type=str, help='pipeline1(contain or not + species)/pipeline2 (none/people/deer/other + rest species)/full(end-to-end one model)')
# parser.add_argument('--prob', type=str, help='classification problem: full/model1(contains or not)/model2(species)')
parser.add_argument('--model', type=str, default='resnet18', help='model type')
parser.add_argument('--period', type=str, default='day', help='train on: day/night/combined/day_plus_night')
parser.add_argument('--merged_categories', type=bool, default='False', help = 'whether combine unidentified bird, unidentified mammal and other into a large category')
parser.add_argument('--norm', type=int, default=0, help='mean and std used for normalization')
parser.add_argument('--optimizer', type=str, default="SGD", help='optimizer choice')
parser.add_argument('--split', type=str, default="train", help='train/dev/test')
parser.add_argument('--sample_list', type=list, default=None, help='oversample list')
parser.add_argument('--sample_N', type=int, default=0, help='number of times to oversample')
parser.add_argument('--lrdecay', type=int, default= -99, help='learning rate decay')
parser.add_argument('--loss', type=str, default='CE', help='loss type')
args = parser.parse_args()


def obtain_inputs_parameters(args):

    csv_path = 'metadata/' + args.period + '_final_' + args.split + '.xlsx'
    data_path = 'data_' + args.period + '_final/' + args.split + '_signs'
    
    if args.norm == 0:
        data_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    if args.norm == 1:
        data_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
#             transforms.Normalize([0.31293, 0.30621,0.28194], [0.1965, 0.1965, 0.19075])
            transforms.Normalize([0.31293, 0.30621,0.28194], [0.18229829,0.18250624,0.1759812])
        ])
    
    if args.norm == 2:
        data_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
    if args.norm == 3: 
        data_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.31293, 0.30621,0.28194], [0.18229829,0.18250624,0.1759812])
        ])
        
    if args.norm == 4: # for inception
        data_transform = transforms.Compose([
            transforms.Resize((299,299)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        
    
    animal_dataset = WildAnimalDataset(csv_file= csv_path, root_dir= data_path, transform = data_transform, merged_categories = False, prob = "full",oversample_list = None, oversample_times = 0, norm = args.norm) # always use full to make sure the correct true labels are extracted
    dataloader = DataLoader(animal_dataset, args.batch_size, shuffle=False, num_workers=4)
    
    return dataloader
    

# if prob = pipeline1 or pipeline2
def read_trained_pipeline_model(args, use_gpu):
    if args.prob == "pipeline1":   
        n_categories1 = 2
        n_categories2 = 23
    else:
        n_categories1 = 4
        n_categories2 = 21   # to be modified
    
    if args.period == "night":
        n_categories2 -= 1
            
    if args.model == "densenet161":
        model1 = models.densenet161(pretrained=True)
        model2 = models.densenet161(pretrained=True)
        num_ftrs1 = model1.classifier.in_features
        num_ftrs2 = model2.classifier.in_features
        model1.classifier = nn.Linear(num_ftrs1, n_categories1)
        model2.classifier = nn.Linear(num_ftrs2, n_categories2)
    else:
        model1 = models.resnet18(pretrained=True)
        model2 = models.resnet18(pretrained=True)
        num_ftrs1 = model1.fc.in_features
        num_ftrs2 = model2.fc.in_features
        model1.fc = nn.Linear(num_ftrs1, n_categories1)
        model2.fc = nn.Linear(num_ftrs2, n_categories2)
        
    model1.train(False)
    model2.train(False)
    
    if use_gpu:
        model1 = model1.cuda()
        model2 = model2.cuda()
        
    if args.prob == "pipeline1":
        model1_checkpoint_path = 'model_params/' + args.period + '_model1_' + args.model + '/best_model.pt'
        model2_checkpoint_path = 'model_params/' + args.period + '_model2_' + args.model + '/best_model.pt'
    else:
        model1_checkpoint_path = 'model_params/' + args.period + '_model3_' + args.model + '/best_model.pt'
        model2_checkpoint_path = 'model_params/' + args.period + '_model4_' + args.model + '/best_model.pt'
    
    print("model1_checkpoint_path: " + model1_checkpoint_path)
    print("model1_checkpoint_path: " + model2_checkpoint_path)
    model1.load_state_dict(torch.load(model1_checkpoint_path))
    model2.load_state_dict(torch.load(model2_checkpoint_path))
    return model1, model2

# if prob = full
def read_trained_single_model(args, use_gpu, n_total_categories):
    if args.period == "night":
        n_total_categories -= 1
        
    if args.model == "densenet161":
        model = models.densenet161(pretrained = True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, n_total_categories)
    elif args.model == "resnet152":
        model = models.resnet152(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_total_categories)  
    elif args.model == "resnet18":
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_total_categories)
    elif args.model == "inceptionv3":
            model = models.inception_v3(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, n_total_categories)

    model.train(False)
    
    if use_gpu:
        model = model.cuda()
        
    model_checkpoint_path = 'model_params/' + args.period + '_' + args.prob + '_' + args.model
    if args.norm == 1:
        model_checkpoint_path += "_normUp"
    if args.norm == 3:
        model_checkpoint_path += "_norm3"
    if args.optimizer == "Adam":
        model_checkpoint_path += "_Adam"
    if args.optimizer == "Nesterov":
        model_checkpoint_path += "_Nesterov"
    if args.model == "inceptionv3":
        model_checkpoint_path += "_lrdecay2"
    if args.norm == 2:
        model_checkpoint_path += "_normMeanImage_lrdecay"
    if args.lrdecay == 1:
        model_checkpoint_path += '_lrdecay'
    if args.sample_list != None:
        model_checkpoint_path += '_oslist'+''.join(str(e) for e in args.sample_list)
    if args.sample_N != 0:
        model_checkpoint_path += '_oversample' + str(args.sample_N)
    if args.loss == 'SVM':
        model_checkpoint_path += 'SVMloss'
        
    model_checkpoint_path += '/best_model.pt'
    print("model_checkpoint_path: " + model_checkpoint_path)
    model.load_state_dict(torch.load(model_checkpoint_path))
    
    return model, model_checkpoint_path

# if prob = pipeline1 or pipeline2
def record_pipeline_prediction(model1, model2, n_total_categories, dataloader, use_gpu, args, class_names):
    
    confusion = torch.zeros(n_total_categories, n_total_categories)
    img_name_list = []
    index = 0
    softmax = nn.Softmax(dim = 1)
    model1.train(False)  
    model2.train(False)

    for data in tqdm(dataloader):
        inputs, labels, img_names = data 
        img_names =  list(img_names)
        labels = labels.view(-1)
        
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs1 = model1(inputs)
        outputs2 = model2(inputs)
        outputs1 = softmax(outputs1).data
        outputs2 = softmax(outputs2).data
        
        _, preds1 = torch.max(outputs1, 1) 
        _, preds2 = torch.max(outputs2, 1) 
        
        if args.prob == 'pipeline1':
            preds2 += 1 
            final_preds = preds2
            final_preds[preds1 == 0] = 0
        else:
            preds2 += 3
            final_preds = preds2
            final_preds[preds1 != 3] = preds1[preds1 != 3]
            
        for i, j in zip(final_preds, labels.data):
            confusion[i,j] += 1
        
        outputs1 = outputs1.cpu().numpy()
        outputs2 = outputs2.cpu().numpy()
        
        if args.prob == "pipeline1":
            temp_matrix1 = outputs1[:,0].reshape((-1,1))
            temp_matrix2 = outputs1[:,1].reshape((-1,1)) * outputs2
            temp_matrix = np.concatenate((temp_matrix1, temp_matrix2), axis=1)
        if args.prob == "pipeline2":
            temp_matrix1 = outputs1[:,[0,1,2]]
            temp_matrix2 = outputs1[:,3].reshape((-1,1)) * outputs2
            temp_matrix = np.concatenate((temp_matrix1, temp_matrix2), axis=1)
 
        if index == 0:
            prediction_matrix = temp_matrix
        else:
            prediction_matrix = np.concatenate((prediction_matrix, temp_matrix))        
        img_name_list = img_name_list + img_names 
        index += 1
        
    # create dataframe
    if args.period ==  "night":
        final_predictions = pd.DataFrame(prediction_matrix, columns = class_names[0:23])
        final_predictions.insert(loc = 23, column = class_names[23], value = 0)
        print(final_predictions.shape)
    else:
        final_predictions = pd.DataFrame(prediction_matrix, columns = class_names)
    final_predictions.insert(loc=0, column='Img_Names', value = img_name_list) 
    
    return confusion, final_predictions

# if prob = full
def create_single_model_confusion(model, n_total_categories, dataloader, use_gpu, args, class_names):
    
    confusion = torch.zeros(n_total_categories, n_total_categories)
    img_name_list = []
    index = 0
    model.train(False) 
    softmax = nn.Softmax(dim = 1)

    for data in tqdm(dataloader):
        inputs, labels, img_names = data 
        img_names =  list(img_names)
        labels = labels.view(-1)
        
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[-1]
        outputs = softmax(outputs).data
        #outputs = model(inputs).data
        _, preds = torch.max(outputs, 1)
        for i, j in zip(preds, labels.data):
            confusion[i,j] += 1
        
        outputs = outputs.cpu().numpy()
        if index == 0:
            prediction_matrix = outputs
        else:
            prediction_matrix = np.concatenate((prediction_matrix, outputs))
        
        img_name_list = img_name_list + img_names 
        index += 1
        
    # create dataframe
    if args.period ==  "night":
        final_predictions = pd.DataFrame(prediction_matrix, columns = class_names[0:23])
        final_predictions.insert(loc = 23, column = class_names[23], value = 0)
        print(final_predictions.shape)
    else:
        final_predictions = pd.DataFrame(prediction_matrix, columns = class_names)
    final_predictions.insert(loc=0, column='Img_Names', value = img_name_list)
    
    return confusion, final_predictions

# functions checked until this point
def write_predictions_csv(final_predictions, model_checkpoint_path, args):
    
    if args.prob == "full":
        predictions_output_name = model_checkpoint_path.split("/")[1] + ".csv"
    else:
        predictions_output_name = args.period + "_" + args.prob + "_" + args.model + ".csv"
    
    if args.split == "train":
        predictions_output_name = "predictions_train/" + predictions_output_name
    
    if args.split == "test":
        predictions_output_name = "test_set_predictions/" + predictions_output_name
    
    if args.split == "dev":
        predictions_output_name = "predictions/" + predictions_output_name
        
        
    
    print("predictions_output_name: " + predictions_output_name)
    final_predictions.to_csv(predictions_output_name, index = False)

def create_confusion_day_plus_night(args):
    day_model = "day_" + args.prob + "_" + args.model
    night_model = "night" + args.prob + "_" + args.model
    day_model_path = "../confusion_matrix/" + day_model + ".csv"
    night_model_path = "../confusion_matrix/" + night_model + ".csv"
    day_confusion = pd.read_csv(day_model_path)
    night_confusion = pd.read_csv(night_model_path)
    combined_confusion = day_confusion + night_confusion
    return combined_confusion
    
    
def main():
    n_total_categories = 24
    use_gpu = True
    class_names = ["None", "Black-tailed deer", "Human", "Black-tailed Jackrabbit", "Coyote", "Unidentified Bird", "Brush Rabbit", "Western scrub-jay","Bobcat", "Other","Unidentified Mammal", "California quail","Raccoon", "Mountain lion","Striped skunk", "Wild turkey", "Gray Fox", "Virginia Opossum", "Stellers jay", "Western Gray Squirrel", "Dusky-footed Woodrat", "Great blue heron", "Fox Squirrel", "California Ground Squirrel"]
    
    if args.period == "day_plus_night": # difference between combined vs day_plus_night?
        confusion = create_confusion_day_plus_night(args)
    else:
        dataloader = obtain_inputs_parameters(args)
        if args.prob == "full":
            model, model_checkpoint_path = read_trained_single_model(args, use_gpu, n_total_categories)
            confusion, final_predictions = create_single_model_confusion(model, n_total_categories, dataloader, use_gpu, args, class_names)
        else:
            model1, model2 = read_trained_pipeline_model(args, use_gpu)
            confusion, final_predictions = record_pipeline_prediction(model1, model2, n_total_categories, dataloader, use_gpu, args, class_names)
        
    print("Total Accuracy: " + str(torch.sum(confusion.diag())/confusion.sum()))
    if args.prob == "full":
        write_predictions_csv(final_predictions, model_checkpoint_path, args)
    else:
        write_predictions_csv(final_predictions, "none", args)
            
#-------------------------------
if __name__ == "__main__":
    main()