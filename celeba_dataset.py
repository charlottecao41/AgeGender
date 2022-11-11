import torch
import torchvision
import os
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms



class CelebADataset(Dataset):
  def __init__(self, dir, transform=None, train = True):
    self.root_dir = dir
    self.gender_attr_index=20
    with open(f'{dir}/partition.txt') as f:
      lines=f.readlines()
    
    train_test_info={}
    for i in lines:
        r=i.split()
        train_test_info.update({r[0]:r[1:]})
    
    with open(f'{dir}/attr.txt') as f:
      lines=f.readlines()

    label=lines[1].split()
    print("total label: ",label)
    self.gender_attr_index=label.index('Male')
    print(f"Male is number {self.gender_attr_index}th label.")
        
    final=lines[2:]
    info=[]
    for i in range(len(final)):
      j=final[i].split()
      if train_test_info[j[0]][-1] == '0' and train:
        info.append(j)
      if train_test_info[j[0]][-1] != '0' and train==False:
        info.append(j)
    
    self.info=info
    self.transform=transform

  def __len__(self): 
    return len(self.info)

  def __getitem__(self, idx):
    # Get the path to the image 

    img_path = os.path.join(f'{self.root_dir}/img_align_celeba', self.info[idx][0])
    # Load image and convert it to RGB
    img = Image.open(img_path).convert('RGB')
    # Apply transformations to the image
    if self.transform:
      img = self.transform(img)

    img_info=self.info[idx][1:]
    l=img_info[self.gender_attr_index]
    if int(l)==int(-1):
        l=0
    else:
        l=1
        
    gender_one_hot=F.one_hot(torch.tensor(l),2)

    return img,torch.tensor(l),gender_one_hot
