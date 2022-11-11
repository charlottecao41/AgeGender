import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import json

class Gender(nn.Module):
    def __init__(self, encoder, encoder_channels, 
                 gender_classes):
        super().__init__()
        self.encoder = encoder
        self.conv1 = nn.Conv2d(encoder_channels, 96,(7,7),stride=4,padding=1)
        self.conv2 = nn.Conv2d(96, 256, (5,5),stride=1,padding=2)
        self.conv3 = nn.Conv2d(256, 384, (3,3),stride=4,padding=1)
        self.pool1 = nn.MaxPool2d(3, stride = 1, padding = 1)
        self.pool2 = nn.MaxPool2d(3, stride = 1, padding = 1)
        self.pool3 = nn.MaxPool2d(3, stride = 1, padding = 1)
        self.fc1=nn.Linear(75264,512)
        self.fc2=nn.Linear(512,512)
        self.layernorm1=nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)
        self.layernorm2=nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)  
        self.layernorm3=nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)
        self.gender_head = nn.Linear(512, gender_classes)
        self.dropout1=nn.Dropout(0.5)
        self.dropout2=nn.Dropout(0.5)
    
    def forward(self, x):
        if self.encoder is not None:
            x = self.encoder(x)
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.layernorm1(x)
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = self.layernorm2(x)
        x = self.pool3(F.leaky_relu(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        gender_logits = self.gender_head(x)
        return gender_logits
    
    
class Gender_reduced_conv(nn.Module):
    def __init__(self, encoder, encoder_channels, 
                 gender_classes):
        super().__init__()
        self.encoder = encoder
        self.conv1 = nn.Conv2d(encoder_channels, 96,(7,7),stride=4,padding=1)
        self.conv2 = nn.Conv2d(96, 256, (5,5),stride=1,padding=2)
        self.pool1 = nn.MaxPool2d(3, stride = 1, padding = 1)
        self.pool2 = nn.MaxPool2d(3, stride = 1, padding = 1)
        self.fc1=nn.Linear(802816,512)
        self.fc2=nn.Linear(512,512)
        self.layernorm1=nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)  
        self.layernorm2=nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)  
        self.gender_head = nn.Linear(512, gender_classes)
        self.dropout1=nn.Dropout(0.5)
        self.dropout2=nn.Dropout(0.5)
    
    def forward(self, x):
        if self.encoder is not None:
            x = self.encoder(x)
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.layernorm1(x)
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = self.layernorm2(x)
        x = x.view(x.size(0),-1)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        gender_logits = self.gender_head(x)
        return gender_logits
    
    
class Gender_reduced_both(nn.Module):
    def __init__(self, encoder, encoder_channels, 
                 gender_classes):
        super().__init__()
        self.encoder = encoder
        self.conv1 = nn.Conv2d(encoder_channels, 96,(7,7),stride=4,padding=1)
        self.conv2 = nn.Conv2d(96, 256, (5,5),stride=1,padding=2)
        self.pool = nn.MaxPool2d(3, stride = 1, padding = 1)
        self.fc1=nn.Linear(802816,512)
        self.layernorm=nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)  
        self.gender_head = nn.Linear(512, gender_classes)
        self.dropout=nn.Dropout(0.5)
    
    def forward(self, x):
        if self.encoder is not None:
            x = self.encoder(x)
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.layernorm(x)
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        gender_logits = self.gender_head(x)
        return gender_logits
    
class Gender_minimalist(nn.Module):
    def __init__(self, encoder, encoder_channels, 
                 gender_classes):
        super().__init__()
        self.encoder = encoder
        self.conv1 = nn.Conv2d(encoder_channels, 96,(7,7),stride=4,padding=1)
        self.pool = nn.MaxPool2d(3, stride = 1, padding = 1)
        self.fc1=nn.Linear(301056,512)
        self.layernorm=nn.LocalResponseNorm(size = 5, alpha = 0.0001, beta = 0.75)  
        self.gender_head = nn.Linear(512, gender_classes)
        self.dropout=nn.Dropout(0.5)
    
    def forward(self, x):
        if self.encoder is not None:
            x = self.encoder(x)
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.layernorm(x)
        x = x.view(x.size(0),-1)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        gender_logits = self.gender_head(x)
        return gender_logits

