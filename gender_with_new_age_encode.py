import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from agegender_dataset import OrdinalAgeGenderDataset
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import resnet34, ResNet34_Weights
from copy import deepcopy

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

train_label=np.array(list(json.load(open("aligned/train.json")).items()))
valid_label=np.array(list(json.load(open("aligned/valid.json")).items()))
test_label=np.array(list(json.load(open("aligned/test.json")).items()))
train_set=OrdinalAgeGenderDataset(train_label,train_transforms)
valid_set=OrdinalAgeGenderDataset(valid_label,valid_transforms)
test_set=OrdinalAgeGenderDataset(test_label,valid_transforms)

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
    pred=np.array(pred)
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
    net.eval()
    result=deepcopy(net.state_dict())
    torch.save(net.state_dict(), f'{subfolder}/{filename}.pt')
    
def validate(model, loader,gender_criterion,age_criterion,metric_fn=accuracy,device=DEVICE,ds_length=len(valid_set)):
    model.eval()
    gender_acc_list = []
    age_accuracy = []
    age_accuracy_true = []
    both_loss_list=[]
    gender_loss=[]
    age_loss=[]
    for image, (age_gt, gender_gt),(age_id_one_hot,gender_id_one_hot) in loader:
        with torch.no_grad():
            image, age_gt, gender_gt = image.to(device), age_gt.to(device), gender_gt.to(device)
            gender_logits, age_logits = model(image)
            loss_age=[]
            loss_weights=[1,1,1,1.3,1.5,1.5,1.3,1,1]
            for i in range(8):
                loss_age.append(age_criterion[i](age_logits[i],age_id_one_hot[i].to(DEVICE).float())*loss_weights[i])
            loss_gender = gender_criterion(gender_logits, gender_id_one_hot.to(DEVICE).float())
            loss = (sum(loss_age) + loss_gender) / sum(loss_weights)
            gender_loss.append(loss_gender.cpu().numpy())
            age_loss.append(sum(loss_age).cpu().detach().numpy())
            gender_acc_list.append(metric_fn(F.softmax(gender_logits,dim=1), gender_gt).item())
            
            inner_age_acc=[]
            one_hot_list=[]
            age_id_one_hot_to_numpy=[]
            age_pred_to_numpy=[]
            for i in range(8):
                one_hot = torch.argmax(age_id_one_hot[i],dim=1).to(DEVICE)
                age_id_one_hot_to_numpy.append(one_hot.cpu().numpy())
                age_pred_to_numpy.append(torch.argmax(F.softmax(age_logits[i],dim=1),dim=1).cpu().numpy())  
                inner_age_acc.append(accuracy(F.softmax(age_logits[i],dim=1),one_hot.to(DEVICE)).item())
                one_hot_list.append(one_hot.cpu().numpy())
            age_accuracy.append(sum(inner_age_acc))
        
            age_id_one_hot_to_numpy=np.array(age_id_one_hot_to_numpy)
            age_id_one_hot_to_numpy=np.transpose(age_id_one_hot_to_numpy)
            age_pred_to_numpy=np.array(age_pred_to_numpy)
            age_pred_to_numpy=np.transpose(age_pred_to_numpy)
            
            acc=0
            for i in range(len(age_id_one_hot_to_numpy)):
                r = np.sum(age_id_one_hot_to_numpy[i]==age_pred_to_numpy[i])
                if r==8:
                    acc=acc+1
                
            age_accuracy_true.append(acc)

            both_loss_list.append(loss.cpu().detach().numpy())
    gender_acc = np.sum(gender_acc_list)/ds_length
    both_loss=np.mean(both_loss_list)
    age_loss=np.mean(age_loss)
    gender_loss=np.mean(gender_loss)
    print("valid gender results: ", gender_acc)
    print(f"valid age accuracy: ", np.sum(np.array(age_accuracy))/ds_length/8)
    #print(age_accuracy_true)
    print(f"valid age accuracy true: ", np.sum(np.array(age_accuracy_true))/ds_length)
    return both_loss,age_loss,gender_loss

def test(model, loader,gender_criterion,age_criterion,metric_fn,device=DEVICE,ds_length=len(test_set),accuracy=accuracy):
    model.eval()
    result=[]
    age_accuracy=[]
    age_accuracy_true=[]
    target=[]
    for image, (age_gt, gender_gt),(age_id_one_hot,gender_id_one_hot) in loader:
        with torch.no_grad():
            image, age_gt, gender_gt = image.to(device), age_gt.to(device), gender_gt.to(device)
            gender_logits,age_logits = model(image)
            
            gender_pred=torch.argmax(F.softmax(gender_logits, dim=1),dim=1)
            result.extend(gender_pred.cpu().numpy().tolist())
            target.extend(gender_gt.cpu().numpy().tolist())
            inner_age_acc=[]
            one_hot_list=[]
            age_id_one_hot_to_numpy=[]
            age_pred_to_numpy=[]
            for i in range(8):
                one_hot = torch.argmax(age_id_one_hot[i],dim=1).to(DEVICE)
                age_id_one_hot_to_numpy.append(one_hot.cpu().numpy())
                age_pred_to_numpy.append(torch.argmax(F.softmax(age_logits[i],dim=1),dim=1).cpu().numpy())  
                inner_age_acc.append(accuracy(F.softmax(age_logits[i],dim=1),one_hot.to(DEVICE)).item())
                one_hot_list.append(one_hot.cpu().numpy())
            age_accuracy.append(sum(inner_age_acc))
        
        age_id_one_hot_to_numpy=np.array(age_id_one_hot_to_numpy)
        age_id_one_hot_to_numpy=np.transpose(age_id_one_hot_to_numpy)
        age_pred_to_numpy=np.array(age_pred_to_numpy)
        age_pred_to_numpy=np.transpose(age_pred_to_numpy)
        
        acc=0
        for i in range(len(age_id_one_hot_to_numpy)):
            r=np.sum(age_id_one_hot_to_numpy[i]==age_pred_to_numpy[i])
            if r==8:
                acc=acc+1
                
        age_accuracy_true.append(acc)
            
    print("gender results: ")
    metric_fn(result,target)
    print(f"age accuracy: ", np.sum(np.array(age_accuracy))/ds_length/8)
    print(f"age accuracy true: ", np.sum(np.array(age_accuracy_true))/ds_length)



def train(epoch,model,loader,gender_criterion,age_criterion,sch,accuracy=accuracy,subfolder='resnet34'):
  for batch_idx, (image, (age_gt, gender_gt),(age_id_one_hot,gender_id_one_hot)) in enumerate(train_loader):
        optimizer.zero_grad()
        #optimizer.zero_grad()
        image, age_gt, gender_gt = image.to(DEVICE), age_gt.to(DEVICE), gender_gt.to(DEVICE)
        gender_logits, age_logits = model(image)
        #print(age_logits)
        loss_age=[]
        if epoch > -1:
            loss_weights=[1,1,1,1.3,1.5,1.5,1.3,1,9.6]
        else:
            loss_weights=[1,1,1,1.3,1.5,1.5,1.3,1,1]
        for i in range(8):
            loss_age.append(age_criterion[i](age_logits[i], age_id_one_hot[i].to(DEVICE).float())*loss_weights[i])
        
        loss_gender = gender_criterion(gender_logits, gender_id_one_hot.to(DEVICE).float())  # softmax+ce
        loss = (sum(loss_age) + loss_gender*loss_weights[-1]) / sum(loss_weights)
        loss.backward()
        optimizer.step()
        sch.step()
        
        inner_age_acc=[]
        one_hot_list=[]
        age_id_one_hot_to_numpy=[]
        age_pred_to_numpy=[]
        for i in range(8):
            one_hot = torch.argmax(age_id_one_hot[i],dim=1).to(DEVICE)
            age_id_one_hot_to_numpy.append(one_hot.cpu().numpy())
            age_pred_to_numpy.append(torch.argmax(F.softmax(age_logits[i]),dim=1).cpu().numpy())  
            inner_age_acc.append(accuracy(F.softmax(age_logits[i]),one_hot.to(DEVICE)).item()/age_logits[i].size(0))
            one_hot_list.append(one_hot.cpu().numpy())
        age_acc = sum(inner_age_acc)/8
        
        age_id_one_hot_to_numpy=np.array(age_id_one_hot_to_numpy)
        age_id_one_hot_to_numpy=np.transpose(age_id_one_hot_to_numpy)
        age_pred_to_numpy=np.array(age_pred_to_numpy)
        age_pred_to_numpy=np.transpose(age_pred_to_numpy)
        
        acc=0
        for i in range(len(age_id_one_hot_to_numpy)):
            r=np.sum(age_id_one_hot_to_numpy[i]==age_pred_to_numpy[i])
            if r==8:
                acc=acc+1
            
        if batch_idx % 10 == 0:
          print(loss.cpu().detach().numpy())


  gender_acc = accuracy(F.softmax(gender_logits), gender_gt)
  print(f"last_batch: epoch:{epoch}, gender: {gender_acc/gender_logits.shape[0]}, age: {age_acc}, age_true: {acc/gender_logits.shape[0]}")
  #save_network(model,subfolder,epoch)
  
class AgeGender(nn.Module):
    def __init__(self, encoder, encoder_channels, 
                 age_classes, gender_classes):
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
        self.gender_head = nn.Linear(512, gender_classes)
        self.dropout1=nn.Dropout(0.5)
        self.dropout2=nn.Dropout(0.5)
        
        self.age_head0 = nn.Linear(512, age_classes)
        self.age_head1 = nn.Linear(512, age_classes)
        self.age_head2 = nn.Linear(512, age_classes)
        self.age_head3 = nn.Linear(512, age_classes)
        self.age_head4 = nn.Linear(512, age_classes)
        self.age_head5 = nn.Linear(512, age_classes)
        self.age_head6 = nn.Linear(512, age_classes)
        self.age_head7 = nn.Linear(512, age_classes)
        
    
    def forward(self, x):
        if self.encoder is not None:
            x = self.encoder(x)
            x = self.resize(x)
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
        age_0=self.age_head0(x).view(x.size(0),-1)
        age_1=self.age_head1(x).view(x.size(0),-1)
        age_2=self.age_head2(x).view(x.size(0),-1)
        age_3=self.age_head3(x).view(x.size(0),-1)
        age_4=self.age_head4(x).view(x.size(0),-1)
        age_5=self.age_head5(x).view(x.size(0),-1)
        age_6=self.age_head6(x).view(x.size(0),-1)
        age_7=self.age_head7(x).view(x.size(0),-1)
        return gender_logits, [age_0,age_1,age_2,age_3,age_4,age_5,age_6,age_7]

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False
            
            
model = AgeGender(None,3, age_classes=2, gender_classes=2).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000])
gender_criterion = nn.CrossEntropyLoss()
age_criterion = [nn.CrossEntropyLoss() for i in range (8)]
both_loss_list = []
gender_loss_list=[]
age_loss_list=[]

best_both_loss=10000
best_age_loss=10000
best_gender_loss=10000

for i in range(EPOCH):
  train(i,model,train_loader,gender_criterion,age_criterion,scheduler)
  both_loss,age_loss,gender_loss=validate(model, valid_loader,gender_criterion,age_criterion,metric_fn=accuracy,device=DEVICE)
  if both_loss<best_both_loss:
      save_network(model, 'gender_with_new_age_encoding','both')
      best_both_loss=both_loss
  if age_loss<best_age_loss:
      save_network(model, 'gender_with_new_age_encoding','age')
      best_age_loss=age_loss
  if gender_loss<best_gender_loss:
      save_network(model, 'gender_with_new_age_encoding','gender')
      best_gender_loss=gender_loss  
           
  both_loss_list.append(both_loss)
  gender_loss_list.append(gender_loss)
  age_loss_list.append(age_loss)
  print(f"valid loss: {both_loss}")
  
print(both_loss_list)
print(gender_loss_list)
print(age_loss_list)

loss_dict={'age': age_loss_list, 'gender':gender_loss_list,'both':both_loss_list}
for i in loss_dict:
  best_model_idx=i
  print("Best loss regarding : ", i)
  best_model = AgeGender(None,3, age_classes=2, gender_classes=2).to(DEVICE)
  best_model.load_state_dict(torch.load(f'gender_with_new_age_encoding/{best_model_idx}.pt'))
  test(best_model, test_loader,gender_criterion,age_criterion,metric_fn=test_metric,device=DEVICE)

#print
model_name='gender_with_new_age_encoding'
for loss_list in [both_loss_list,gender_loss_list,age_loss_list]:
    plt.plot([i for i in range(len(loss_list))],loss_list)
    
plt.savefig(f'graph/{model_name}.png')