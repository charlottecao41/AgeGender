import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import json
from agegender_dataset import SplitAgeGenderDataset
import matplotlib.pyplot as plt
from gender_model import Gender,Gender_reduced_conv,Gender_reduced_both
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
EPOCH = 20
LR = 1e-4

train_label=np.array(list(json.load(open("/home/jupyter/shared/aligned/train.json")).items()))
valid_label=np.array(list(json.load(open("/home/jupyter/shared/aligned/valid.json")).items()))
test_label=np.array(list(json.load(open("/home/jupyter/shared/aligned/test.json")).items()))
train_set=SplitAgeGenderDataset(train_label,train_transforms)
valid_set=SplitAgeGenderDataset(valid_label,valid_transforms)
test_set=SplitAgeGenderDataset(test_label,valid_transforms)

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
    
def validate(model, loader,criterion,metric_fn=accuracy,device=DEVICE,ds_length=len(valid_set)):
    model.eval()
    acc_list = []
    loss_list=[]
    for image, gt,one_hot in loader:
        with torch.no_grad():
            image, gt, one_hot = image.to(device), gt.to(device), one_hot.to(device)
            logits = model(image)
            loss = criterion(logits, one_hot.to(DEVICE).float())
            loss = loss.cpu().numpy()
            acc_list.append(metric_fn(logits,gt))
            loss_list.append(loss)
    acc = np.sum(acc_list)/ds_length
    loss=np.mean(loss_list)
    print(f"valid: acc {acc:.2%}")
    return loss,acc

def test(model, loader,gender_criterion,metric_fn=accuracy,device=DEVICE,ds_length=len(test_set)):
    model.eval()
    result=[]
    target=[]
    acc=[]
    for image, gt,one_hot in loader:
        with torch.no_grad():
            image, gt, one_hot = image.to(device), gt.to(device), one_hot.to(device)
            logits = model(image)
            acc.append(accuracy(logits,gt))
            split_result=torch.argmax(logits,dim=1)
            result.extend(split_result.cpu().numpy().tolist())
            target.extend(gt.cpu().numpy().tolist())
    print("overall accuracy: ",np.sum(acc)/ds_length)
    print("gender accuracy: ")
    gender_pred=[]
    gender_target=[]
    for i in range(len(target)):
        if target[i]>=8:
            gender_target.append(1)
        else:
            gender_target.append(0)
            
        if result[i]>=8:
            gender_pred.append(1)
        else:
            gender_pred.append(0)
    metric_fn(gender_pred,gender_target)
    print("age accuracy: ")
    age_pred=[]
    age_target=[]
    for i in range(len(target)):
        if target[i]>=8:
            age_target.append(target[i]-8)
        else:
            age_target.append(target[i])
            
        if result[i]>=8:
            age_pred.append(result[i]-8)
        else:
            age_pred.append(result[i])
    age_pred=np.array(age_pred).astype(int)
    age_target=np.array(age_target).astype(int)
    print(np.sum(np.equal(age_pred,age_target))/ds_length)



def train(epoch,model,loader,gender_criterion,subfolder,sch,accuracy=accuracy,device=DEVICE):
  for batch_idx, (image, gt, one_hot) in enumerate(train_loader):
        optimizer.zero_grad()
        sch.step()
        image, gt, one_hot = image.to(device), gt.to(device), one_hot.to(device)
        logits = model(image)
        loss = gender_criterion(logits, one_hot.to(DEVICE).float())  # softmax+ce
        loss.backward()
        optimizer.step()
        sch.step()
        if batch_idx % 10 == 0:
          print(loss.cpu().detach().numpy())

  acc = accuracy(logits, gt)/logits.shape[0]
  print(f"epoch:{epoch}, gender accuracy of the last batch: {acc}")
  return loss.cpu().detach().numpy(), acc

model_dict={'split':Gender(None,3, gender_classes=16).to(DEVICE)}

for gender_model in model_dict:
    print(gender_model)
    model = model_dict[gender_model]
    optimizer = torch.optim.Adam(model.parameters(), LR)
    gender_criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000])
    loss_list=[]
    acc_list=[]
    train_loss=[]
    train_acc=[]
    best_loss,a=validate(model,valid_loader,gender_criterion)
    best_model=0
    print("init_best_loss:",best_loss)
    for i in range(EPOCH):
        tl,ta=train(i,model,train_loader,gender_criterion,gender_model,sch=scheduler)
        train_loss.append(tl)
        train_acc.append(ta)
        loss,va=validate(model, valid_loader,gender_criterion,metric_fn=accuracy,device=DEVICE)
        loss_list.append(loss)
        acc_list.append(va)
        print(f"valid loss: {loss}")
        if loss<best_loss:
            print("best loss found: ", loss)
            best_loss=loss
            save_network(model,gender_model,'best')

    model_name = gender_model
    plt.figure(figsize=(10,8))
    plt.plot([i for i in range(len(loss_list))],train_loss, label = "train")
    plt.plot([i for i in range(len(loss_list))],loss_list, label = "valid")
    plt.legend()
    plt.savefig(f'graph/{model_name}loss.png')
    
    plt.figure(figsize=(10,8))
    plt.plot([i for i in range(len(train_acc))],train_acc, label = "train")
    plt.plot([i for i in range(len(acc_list))],acc_list, label = "valid")
    plt.legend()
    plt.savefig(f'graph/{model_name}acc.png')
    
    best_model = model_dict[gender_model]
    best_model.load_state_dict(torch.load(f'gender_model/{gender_model}/best.pt'))
    test(best_model,test_loader,gender_criterion,metric_fn=test_metric,device=DEVICE)
