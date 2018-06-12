# Without modification for combined categories and adjusted loss
# on dev set only
# random crop not turned off yet
# haven't check logic for day/night yet

from __future__ import print_function, division
import argparse
import io
import os
import fnmatch
import requests
from PIL import Image
import cv2
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision import datasets, models, transforms

from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image, ImageFile, ImageFont, ImageDraw
from tqdm import tqdm

from WildAnimalDataset import WildAnimalDataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--photo_path', type=str, default='/data/cs341-bucket/camera_B2', help='photo path')
# parser.add_argument('--photo_path', type=str, default='/test_data', help='photo path')
parser.add_argument('--meta_path', type=str, default='/data/cs341-bucket/camera_B2/camera_B2.xlsx', help='metadata path')
parser.add_argument('--output_path', type=str, default='unsettled/', help='unsettled output path')

# parser.add_argument('--photo_path', type=str, default='/data/data_combined_final/dev_signs', help='photo path')
# parser.add_argument('--meta_path', type=str, default='/data/metadata/combined_final_dev.xlsx', help='metadata path')
parser.add_argument('--threshold', type=float, default=0, help='threshold for settling prediction')


def obtain_inputs_parameters(args, meta_path, photo_path):

#     meta_path = args.meta_path
#     photo_path = args.photo_path

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

def trans(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1) * 255
    return inp

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (224, 224)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def write_pred(args, meta_path, folder, model, dataloader, use_gpu, class_names):
    meta = pd.io.excel.read_excel(meta_path)
    
    pred_settled = pd.DataFrame(columns = list(meta.columns.values) + ['pred'])
    pred_unsettled = pd.DataFrame(columns = list(meta.columns.values) + ['pred'])
    settled = 0
    unsettled = 0
    
    model.train(False)
    model.eval()
    
    # hook the feature extractor
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
    
    for data in tqdm(dataloader):
        inputs, image = data
        features_blobs = []
        model._modules.get('features').register_forward_hook(hook_feature)
        if use_gpu:
            inputs = Variable(inputs.cuda())
            image = Variable(image.cuda())
        else:
            inputs = Variable(inputs)
            image = Variable(image)

        m = nn.Softmax(dim = 1)
        outputs = m(model(inputs)) 
        outputs = (outputs.data).cpu().numpy()
        preds = np.argsort(outputs, 1)[:,-1:]

        for i in range(preds.shape[0]):
            curr = meta.loc[settled + unsettled]
            curr['pred'] = class_names[preds[i,-1]]

            conf = outputs[i, preds[i,-1]]           
            if conf > args.threshold:
                settled += 1
                pred_settled.loc[settled] = curr 
            else:
                unsettled += 1
                pred_unsettled.loc[unsettled] = curr
                
                # generate CAM with caption
                CAMs = returnCAM(features_blobs[0][i], weight_softmax, [preds[i]])
                img = inputs.cpu().data[i]
                heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(224, 224)), cv2.COLORMAP_JET)
                img = trans(img)
                org=img.astype(np.uint8)
                org = Image.fromarray(org)
                draw = ImageDraw.Draw(org)
#                 fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 15)
                draw.text((0, 0), "Predicted: {}".format(class_names[curr['pred']]))                
                org_fig = cv2.cvtColor(np.array(org), cv2.COLOR_RGB2BGR)
                result = heatmap * 0.3 + org_fig * 0.6
                
                # original image
                photo = image.cpu().data[i].numpy()
                photo = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)
                
                # combine images
                total_width = result.shape[1] + photo.shape[1]
                max_height = max(result.shape[0], photo.shape[0])
                comb_img = np.zeros((max_height, total_width, 3), np.uint8)
                comb_img[:, 0:photo.shape[1]] = photo
                comb_img[0:result.shape[0], photo.shape[1]:total_width] = result
                
                #### change later based on file dir ####
                out_path = args.output_path + cam + '_' + str(unsettled) + '.jpg'
                
                ### add file name to excel
                pred_unsettled.loc[unsettled]['file_name'] = out_path
                
                cv2.imwrite(out_path, comb_img)

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
    
    model = read_trained_single_model(args, use_gpu, n_total_categories)
    
    # get all related folders
    keep = []
    for file in os.listdir(args.photo_path + '/'):
        if fnmatch.fnmatch(file, 'camera*'):
            keep.append(file)
    
    # photo path: directory above all camera folders
    for folder in keep:
        meta_path = args.photo_path + '/metadata/' + folder + '.xlsx'
        photo_path = args.photo_path + '/' + folder
        dataloader = obtain_inputs_parameters(args, meta_path, photo_path)    
        write_pred(args, meta_path, folder, model, dataloader, use_gpu, class_names)

if __name__ == "__main__":
    main()