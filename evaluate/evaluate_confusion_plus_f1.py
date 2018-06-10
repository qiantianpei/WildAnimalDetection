# Without modification for combined categories and adjusted loss
# on dev set only
# random crop not turned off yet
# haven't check logic for day/night yet

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
parser.add_argument('--epoch', type=int, default=20, help='number of training epochs')
parser.add_argument('--merged_categories', type=bool, default='False', help = 'whether combine unidentified bird, unidentified mammal and other into a large category')
parser.add_argument('--norm', type=int, default=0, help='mean and std used for normalization')
parser.add_argument('--optimizer', type=str, default="SGD", help='optimizer choice')
parser.add_argument('--sample_list', type=list, default=None, help='oversample list')
parser.add_argument('--sample_N', type=int, default=0, help='number of times to oversample')
parser.add_argument('--lrdecay', type=int, default= -99, help='learning rate decay')
parser.add_argument('--split', type=str, default= "dev", help='dev/train/test')
args = parser.parse_args()


def obtain_inputs_parameters(args):

    csv_path = 'metadata/' + args.period + '_final_' + args.split + '.xlsx'
    data_path = 'data_' + args.period + '_final/' + args.split + '_signs'
    
    if args.norm == 0:
        data_transform = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    if args.norm == 1:
        data_transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.31293, 0.30621,0.28194], [0.18229829,0.18250624,0.1759812])
        ])
    if args.norm == 3: 
        data_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.31293, 0.30621,0.28194], [0.18229829,0.18250624,0.1759812])
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
    else:
        model = models.resnet18(pretrained=True)
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
        
    model_checkpoint_path += '/best_model.pt'
    print("model_checkpoint_path: " + model_checkpoint_path)
    model.load_state_dict(torch.load(model_checkpoint_path))
    
    return model

# if prob = pipeline1 or pipeline2
def create_pipeline_confusion(model1, model2, n_total_categories, dataloader, use_gpu, args):
    
    confusion = torch.zeros(n_total_categories, n_total_categories)
    
    model1.train(False)  
    model2.train(False)

    for data in tqdm(dataloader):
        inputs, labels = data 
        labels = labels.view(-1)
        
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        
        sm = torch.nn.Softmax(1)
        
        outputs1 = sm(model1(inputs)) 
        outputs2 = sm(model2(inputs))
        
        outputs = torch.zeros(outputs1.size(0), outputs2.size(1) + 1)
        outputs[:,0] = outputs1.data[:,0]
        outputs[:,1:] = outputs1.data[:,1:] * outputs2.data

#         _, preds1 = torch.max(outputs1.data, 1) 
#         _, preds2 = torch.max(outputs2.data, 1) 
        _, preds = torch.max(outputs, 1) 
        
        if args.prob == 'pipeline1':
#             preds2 += 1 
#             final_preds = preds2
#             final_preds[preds1 == 0] = 0
            final_preds = preds
        else:
            preds2 += 3
            final_preds = preds2
            final_preds[preds1 != 3] = preds1[preds1 != 3]
            
        for i, j in zip(final_preds, labels.data):
            confusion[i,j] += 1
    return confusion

# if prob = full
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

# functions checked until this point
def write_confusion_csv(confusion, class_names, args):
    confusion_df = pd.DataFrame(confusion.numpy(), columns = class_names)
    confusion_output_name = "confusion_matrix/" + args.period + "_" + args.prob + "_" + args.model
    if args.norm == 1:
        confusion_output_name += "_normUp"
    if args.norm == 3:
        confusion_output_name += "_norm3"
    if args.optimizer == "Adam":
        confusion_output_name += "_Adam"
    if args.optimizer == "Nesterov":
        confusion_output_name += "_Nesterov"
    if args.model == "inceptionv3":
        confusion_output_name += "_lrdecay2"
    if args.norm == 2:
        confusion_output_name += "_normMeanImage_lrdecay"
    if args.lrdecay == 1:
        confusion_output_name += '_lrdecay'
    if args.sample_list != None:
        confusion_output_name += '_oslist'+''.join(str(e) for e in args.sample_list)
    if args.sample_N != 0:
        confusion_output_name += '_oversample' + str(args.sample_N)
        
    confusion_output_name += '.csv'
    if args.split == "test":
        confusion_output_name = "test_" + confusion_output_name
    confusion_df.to_csv(confusion_output_name, index = False)

def calculateRecall(class_index, confusion):
    true_positive = confusion[class_index][class_index]
    false_negative = torch.sum(confusion[:, class_index]) - true_positive
    if true_positive + false_negative == 0:
        return None
    else:
        return true_positive / (true_positive + false_negative)

def calculatePrecision(class_index, confusion):
    true_positive = confusion[class_index][class_index]
    false_positive = torch.sum(confusion[class_index,:]) - true_positive
    if true_positive + false_positive == 0:
        return None
    else:
        return true_positive / (true_positive + false_positive)

def calculateF1score(recall, precision):
    if recall == None or precision == None:
        return None
    else: 
        return 2 * ((recall * precision) / (recall + precision))

def model_eval(confusion, n_total_categories, class_names):
    full_info = [] # [(species, recall, precision, f1_score), (species, recall, precision, f1_score), ...]
    for class_index in range(n_total_categories):
        recall = calculateRecall(class_index, confusion)
        precision = calculatePrecision(class_index, confusion)
        f1_score = calculateF1score(recall, precision)
        value = tuple([class_names[class_index], recall, precision, f1_score])
        full_info.append(value)
    
    model_eval_result = pd.DataFrame.from_records(full_info, columns = ['species', 'recall', 'precision', 'f1score'])
    return model_eval_result   

def write_metrics_to_csv(model_eval_result, args):
    eval_result_output_path = "evaluation_metrics/" + args.period + "_" + args.prob + "_" + args.model
    if args.norm == 1:
        eval_result_output_path += "_normUp"
    if args.norm == 3:
        eval_result_output_path += "_norm3"
    if args.optimizer == "Adam":
        eval_result_output_path += "_Adam"
    if args.optimizer == "Nesterov":
        eval_result_output_path += "_Nesterov"
    if args.model == "inceptionv3":
        eval_result_output_path += "_lrdecay2"
    if args.norm == 2:
        eval_result_output_path += "_normMeanImage_lrdecay"
    if args.lrdecay == 1:
        eval_result_output_path += '_lrdecay'
    if args.sample_list != None:
        eval_result_output_path += '_oslist'+''.join(str(e) for e in args.sample_list)
    if args.sample_N != 0:
        eval_result_output_path += '_oversample' + str(args.sample_N)
        
    eval_result_output_path += '.csv'
    if args.split == "test":
        eval_result_output_path = "test_" + eval_result_output_path
    model_eval_result.to_csv(eval_result_output_path, index = False)

def create_confusion_day_plus_night(args):
    day_model = "day_" + args.prob + "_" + args.model
    night_model = "night_" + args.prob + "_" + args.model
    day_model_path = "confusion_matrix/" + day_model + ".csv"
    night_model_path = "confusion_matrix/" + night_model + ".csv"
    day_confusion = pd.read_csv(day_model_path)
    night_confusion = pd.read_csv(night_model_path)
    combined_confusion = day_confusion + night_confusion
    return combined_confusion
    
    
def main():
    n_total_categories = 24
    use_gpu = True
    class_names = ["None", "Black-tailed deer", "Human", "Black-tailed Jackrabbit", "Coyote", "Unidentified Bird", "Brush Rabbit", "Western scrub-jay","Bobcat", "Other","Unidentified mammal", "California quail","Raccoon", "Mountain lion","Striped skunk", "Wild turkey", "Gray Fox", "Virginia Opossum", "Stellers jay", "Western Gray Squirrel", "Dusky-footed Woodrat", "Great blue heron", "Fox Squirrel", "California Ground Squirrel"]
    
    if args.period == "day_plus_night": # difference between combined vs day_plus_night?
        confusion = torch.from_numpy(create_confusion_day_plus_night(args).values)
    else:
        dataloader = obtain_inputs_parameters(args)
        if args.prob == "full":
            model = read_trained_single_model(args, use_gpu, n_total_categories)
            confusion = create_single_model_confusion(model, n_total_categories, dataloader, use_gpu, args)
        else:
            model1, model2 = read_trained_pipeline_model(args, use_gpu)
            confusion = create_pipeline_confusion(model1, model2, n_total_categories, dataloader, use_gpu, args)
        
#     print("Total Accuracy: " + str( (confusion.diag().sum() - confusion[5,5] - confusion[10,10]) / (confusion.sum() - confusion[5,:].sum() - confusion[10,:].sum() - confusion[:,5].sum() - confusion[:,10].sum() + confusion[5,5] + confusion[10,10] ) )) 
    print("Total Accuracy: " + str(torch.sum(confusion.diag())/confusion.sum())) 
    write_confusion_csv(confusion, class_names, args)
    model_eval_result = model_eval(confusion, n_total_categories, class_names)
    write_metrics_to_csv(model_eval_result, args)    

#-------------------------------
if __name__ == "__main__":
    main()