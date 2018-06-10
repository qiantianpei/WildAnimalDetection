from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

class WildAnimalDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, oversample_list, oversample_times, root_dir, transform, prob, merged_categories = False, norm = 0):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label = pd.io.excel.read_excel(csv_file)
        # oversample
        if oversample_list==None:
            oversample_list=[]
        else:
            sample = pd.DataFrame()
            dup = (self.label).loc[self.label['Species'].isin(oversample_list)]
            for i in range(oversample_times):
                sample = pd.concat([dup, sample], ignore_index=True)
            self.label = pd.concat([self.label, sample], ignore_index=True) 

        self.root_dir = root_dir
        self.transform = transform
        self.prob = prob
        self.merged_categories = merged_categories
        self.norm = norm
        
        self.read_mean_images()
        
    def read_mean_images(self):
        self.mean_images = {}
        for image in os.listdir('composite_images_large'):
            self.mean_images[image] = np.array(Image.open('composite_images_large/' + image)) / 255

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.label.up_name[idx])
        image = Image.open(img_name)
        if self.prob == 'full':
            if self.merged_categories:
                label = self.label.Species_Combined_Unidentified[idx]
            else:
                label = self.label.Species[idx]
        if self.prob == 'model1':
            label = self.label.Contain[idx]
        if self.prob == 'model2':
            label = self.label.Species[idx] - 1
        if self.prob == 'model3':
            label = self.label.major[idx]
        if self.prob == 'model4':
            label = self.label.rare[idx]
        
        if self.norm == 2: 
            if self.label.Light[idx] > 0:
                key = 'day_'
            else:
                key = 'night_'
            key += self.label.CameraPath[idx].replace('/','_') + '.png'
            image_transformed = np.array(transforms.Resize((1000,1500))(image)) / 255 - self.mean_images[key]
            image_transformed = Image.fromarray(np.uint8(np.clip(image_transformed, 0, 1) * 255))
            image_transformed = self.transform(image_transformed)
        else:
            image_transformed = self.transform(image)
        
        sample = [image_transformed, torch.from_numpy(np.array([int(label)]))]
        
        return sample