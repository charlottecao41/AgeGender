import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from celeba_dataset import CelebADataset
import matplotlib.pyplot as plt
from gender_model import Gender
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(42)
BATCH_SIZE=64
DEVICE="cuda"
EPOCH = 10 #or 3
LR = 1e-4

img_folder = f'/home/jupyter/shared/celeba'

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


# Load the dataset from file and apply transformations
train_set = CelebADataset(img_folder, train_transforms)
valid_set = CelebADataset(img_folder, valid_transforms, False)

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

def accuracy(pred,target):
  pred=torch.argmax(pred,dim=1)
  return torch.sum(pred==target).cpu().numpy()

def test_metric(pred,target):
    pred=np.argmax(np.array(pred),axis=1)
    target=np.array(target)
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
    torch.save(net.state_dict(), f'celeba_model/{filename}.pt')
    
def validate(model, loader,gender_criterion,metric_fn=accuracy,device=DEVICE,ds_length=len(valid_set)):
    model.eval()
    gender_acc_list = []
    loss_list=[]
    for image, gender_gt,gender_id_one_hot in loader:
        with torch.no_grad():
            image, gender_gt = image.to(device), gender_gt.to(device)
            gender_logits = model(image)
            loss_gender = gender_criterion(gender_logits, gender_id_one_hot.to(DEVICE).float())
            loss = loss_gender.cpu().numpy()
            gender_acc_list.append(metric_fn(gender_logits,gender_gt))
            loss_list.append(loss)
    gender_acc = np.sum(gender_acc_list)/ds_length
    loss=np.mean(loss_list)
    print(f"valid: gender acc {gender_acc:.2%}")
    return loss

def test(model, loader,gender_criterion,metric_fn=accuracy,device=DEVICE,ds_length=len(valid_set)):
    model.eval()
    result=[]
    target=[]
    for image, gender_gt,gender_id_one_hot in loader:
        with torch.no_grad():
            image, gender_gt = image.to(device), gender_gt.to(device)
            gender_logits = model(image)
            result.extend(gender_logits.cpu().numpy().tolist())
            target.extend(gender_gt.cpu().numpy().tolist())
    metric_fn(result,target)



def train(epoch,model,loader,gender_criterion,subfolder,accuracy=accuracy):
  for batch_idx, (image, gender_gt, gender_id_one_hot) in enumerate(train_loader):
        optimizer.zero_grad()
        #sch.step()
        image, gender_gt = image.to(DEVICE), gender_gt.to(DEVICE)
        gender_logits = model(image)
        loss_gender = gender_criterion(gender_logits, gender_id_one_hot.to(DEVICE).float())  # softmax+ce
        loss_gender.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
          print(loss_gender.cpu().detach().numpy())

  gender_acc = accuracy(gender_logits, gender_gt)/gender_gt.shape[0]
  print(f"epoch:{epoch}, gender accuracy of the last batch: {gender_acc}")
  #save_network(model,subfolder,epoch)

model_dict={'original_on_celeba':Gender(None,3, gender_classes=2).to(DEVICE)}

for gender_model in model_dict:
    print(gender_model)
    model = model_dict[gender_model]
    optimizer = torch.optim.Adam(model.parameters(), LR)
    gender_criterion = nn.CrossEntropyLoss()

    loss_list=[]
    best_loss=validate(model,valid_loader,gender_criterion)
    print("init_best_loss:",best_loss)
    for i in range(EPOCH):
        train(i,model,train_loader,gender_criterion,gender_model)
        loss=validate(model, valid_loader,gender_criterion,metric_fn=accuracy,device=DEVICE)
        loss_list.append(loss)
        print(f"valid loss: {loss}")
        if loss<best_loss:
            save_network(model,'celeba_model','best')
            best_loss=loss

    model_name = gender_model
    plt.plot([i for i in range(len(loss_list))],loss_list)
    plt.savefig(f'graph/{model_name}.png')
    
    best_model = model_dict[gender_model]
    best_model.load_state_dict(torch.load(f'celeba_model/best.pt'))
    test(best_model,valid_loader,gender_criterion,metric_fn=test_metric,device=DEVICE)