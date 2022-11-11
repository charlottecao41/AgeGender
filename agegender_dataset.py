
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import json
from PIL import Image

PATH = 'aligned/'

class AgeGenderDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        filepath, meta = self.items[idx]
        img = Image.open(PATH+filepath)
        if self.transform is not None:
          img = self.transform(img)
        age_id, gender_id = torch.tensor(meta['age_id']), torch.tensor(meta['gender_id'])
        age_id_one_hot=F.one_hot(torch.tensor(int(meta['age_id'])),8)
        gender_id_one_hot = F.one_hot(torch.tensor(int(meta['gender_id'])),2)
        return img, (age_id, gender_id),(age_id_one_hot,gender_id_one_hot)
    
    
class SplitAgeGenderDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        filepath, meta = self.items[idx]
        img = Image.open(PATH+filepath)
        if self.transform is not None:
          img = self.transform(img)
        age_id, gender_id = int(meta['age_id']), int(meta['gender_id'])
        if gender_id == 1:
            split_id=torch.tensor((age_id+8))
        else:
            split_id=torch.tensor(age_id)
        split_id_one_hot=F.one_hot(torch.tensor(split_id),16)
        return img, split_id, split_id_one_hot
    
    
class OrdinalAgeGenderDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        filepath, meta = self.items[idx]
        img = Image.open(PATH+filepath)
        if self.transform is not None:
          img = self.transform(img)
        age_id, gender_id = int(meta['age_id']), int(meta['gender_id'])
        gender_onehot=F.one_hot(torch.tensor(gender_id),2)
        age_onehot=[]
        for i in range(8):
            if age_id >= i:
                age_onehot.append(torch.tensor([1,0]))
            else:
                age_onehot.append(torch.tensor([0,1]))
                
        
        return img, (torch.tensor(age_id), torch.tensor(gender_id)),(age_onehot,gender_onehot)