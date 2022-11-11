import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import json
from agegender_dataset import AgeGenderDataset
import matplotlib.pyplot as plt
from gender_model import Gender_minimalist
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(42)

train_transforms = transforms.Compose(
    [transforms.Resize(256),
    transforms.RandomCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()]
)

valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.ToTensor()
])


BATCH_SIZE=64
DEVICE="cuda"
EPOCH = 10
LR = 1e-4

train_label=np.array(list(json.load(open("aligned/train.json")).items()))
valid_label=np.array(list(json.load(open("aligned/valid.json")).items()))
test_label=np.array(list(json.load(open("aligned/test.json")).items()))

train_set=AgeGenderDataset(train_label,train_transforms)
valid_set=AgeGenderDataset(valid_label,valid_transforms)
test_set=AgeGenderDataset(test_label,valid_transforms)

train_loader = DataLoader(
    train_set,
    BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    valid_set,
    BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_set,
    BATCH_SIZE,
    shuffle=False
)

def accuracy(pred,target):
  pred=torch.argmax(pred,dim=1)
  return torch.sum(pred==target).cpu().numpy()


def test_metric(pred,target):
    pred=np.array(pred).astype(int)
    target=np.array(target).astype(int)
    a=np.sum(pred==target)/len(pred)
    tp=0
    for i in range(len(pred)):
        if int(pred[i])==int(target[i]) and int(pred[i])==1:
            tp=tp+1
            
    tn=np.sum(pred==target)-tp
    fp=np.sum(pred)-tp
    fn=(len(pred)-np.sum(pred))-tn
    p=tp/(tp+fp)
    r=tp/(tp+fn)
    f1=p*r/(p+r)*2
    print(f"accuracy: {a}, f1: {f1}, precision: {p}, recall: {r}")


def save_network(net, subfolder,filename):
    net.eval()
    torch.save(net.state_dict(), f'gender_model/{subfolder}/{filename}.pt')
    
def validate(model, loader,gender_criterion,metric_fn=accuracy,device=DEVICE,ds_length=len(valid_set)):
    model.eval()
    gender_acc_list = []
    loss_list=[]
    for image, (age_gt, gender_gt),(age_id_one_hot,gender_id_one_hot) in loader:
        with torch.no_grad():
            image, _, gender_gt = image.to(device), age_gt.to(device), gender_gt.to(device)
            gender_logits = model(image)
            loss_gender = gender_criterion(gender_logits, gender_id_one_hot.to(DEVICE).float())
            loss = loss_gender.cpu().numpy()
            gender_acc_list.append(metric_fn(gender_logits,gender_gt))
            loss_list.append(loss)
    gender_acc = np.sum(gender_acc_list)/ds_length
    loss=np.mean(loss_list)
    print(f"valid: gender acc {gender_acc:.2%}")
    return loss,gender_acc

def test(model, loader,gender_criterion,metric_fn=accuracy,device=DEVICE,ds_length=len(test_set)):
    model.eval()
    result=[]
    target=[]
    for image, (age_gt, gender_gt),(age_id_one_hot,gender_id_one_hot) in loader:
        with torch.no_grad():
            image, _, gender_gt = image.to(device), age_gt.to(device), gender_gt.to(device)
            gender_logits = model(image)
            gender_result=torch.argmax(gender_logits,dim=1)
            result.extend(gender_result.cpu().numpy().tolist())
            target.extend(gender_gt.cpu().numpy().tolist())
    print("accuracy: ")
    metric_fn(result,target)



def train(epoch,model,loader,gender_criterion,subfolder,sch,accuracy=accuracy):
  for batch_idx, (image, (age_gt, gender_gt),(age_id_one_hot,gender_id_one_hot)) in enumerate(train_loader):
        optimizer.zero_grad()
        #sch.step()
        image, _, gender_gt = image.to(DEVICE), age_gt.to(DEVICE), gender_gt.to(DEVICE)
        gender_logits = model(image)
        loss_gender = gender_criterion(gender_logits, gender_id_one_hot.to(DEVICE).float())  # softmax+ce
        loss_gender.backward()
        optimizer.step()
        sch.step()
        if batch_idx % 10 == 0:
          print(loss_gender.cpu().detach().numpy())

  gender_acc = accuracy(gender_logits, gender_gt)/gender_logits.shape[0]
  print(f"epoch:{epoch}, gender accuracy of the last batch: {gender_acc}")
  #save_network(model,subfolder,epoch)
  return loss_gender.cpu().detach().numpy(),gender_acc 

model_dict={'minimalist':Gender_minimalist(None,3,2).to(DEVICE)}

for gender_model in model_dict:
    print(gender_model)
    model = model_dict[gender_model]
    optimizer = torch.optim.Adam(model.parameters(), LR)
    gender_criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000])
    
    loss_list=[]
    train_loss=[]
    v_acc=[]
    train_acc=[]
    best_loss,a=validate(model,valid_loader,gender_criterion)
    print("init_best_loss:",best_loss)
    for i in range(EPOCH):
        l,a=train(i,model,train_loader,gender_criterion,gender_model,sch=scheduler)
        train_loss.append(l)
        train_acc.append(a)
        loss,a=validate(model, valid_loader,gender_criterion,metric_fn=accuracy,device=DEVICE)
        loss_list.append(loss)
        v_acc.append(a)
        print(f"valid loss: {loss}")
        if loss<best_loss:
            best_loss=loss
            save_network(model,gender_model,'best')

    model_name = gender_model
    plt.figure(figsize=(10,8))
    plt.plot([i for i in range((len(loss_list)))],train_loss, label = "train")
    plt.plot([i for i in range(len(loss_list))],loss_list, label = "valid")
    plt.legend()
    plt.savefig(f'graph/{model_name}loss.png')
    
    plt.figure(figsize=(10,8))
    plt.plot([i for i in range((len(loss_list)))],train_acc, label = "train")
    plt.plot([i for i in range(len(loss_list))],v_acc, label = "valid")
    plt.legend()
    plt.savefig(f'graph/{model_name}acc.png')
    
    best_model = model_dict[gender_model]
    best_model.load_state_dict(torch.load(f'gender_model/{gender_model}/best.pt'))
    test(best_model,test_loader,gender_criterion,metric_fn=test_metric,device=DEVICE)
    
