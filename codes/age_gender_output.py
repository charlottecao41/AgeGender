import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from agegender_dataset import AgeGenderDataset
import matplotlib.pyplot as plt
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
    torch.save(net.state_dict(), f'{subfolder}/{filename}.pt')
    
def validate(model, loader,gender_criterion,age_criterion,metric_fn=accuracy,device=DEVICE,ds_length=len(valid_set)):
    model.eval()
    gender_acc_list = []
    age_acc_list = []
    both_loss_list=[]
    gender_loss=[]
    age_loss=[]
    for image, (age_gt, gender_gt),(age_id_one_hot,gender_id_one_hot) in loader:
        with torch.no_grad():
            image, age_gt, gender_gt = image.to(device), age_gt.to(device), gender_gt.to(device)
            age_logits, gender_logits = model(image)
            loss_age = age_criterion(age_logits, age_id_one_hot.to(DEVICE).float())  # bce
            loss_gender = gender_criterion(gender_logits, gender_id_one_hot.to(DEVICE).float())  # softmax+ce
            loss = ((loss_age + loss_gender) / 2).cpu().numpy()
            gender_loss.append(loss_gender.cpu().numpy())
            age_loss.append(loss_age.cpu().numpy())
            gender_acc_list.append(metric_fn(gender_logits, gender_gt).item())
            age_acc_list.append(metric_fn(age_logits, age_gt).item())
            #print(loss)
            both_loss_list.append(loss)
    gender_acc = np.sum(gender_acc_list)/ds_length
    age_acc = np.sum(age_acc_list)/ds_length
    both_loss=np.mean(both_loss_list)
    age_loss=np.mean(age_loss)
    gender_loss=np.mean(gender_loss)
    print(f"valid: gender acc {gender_acc:.2%}, age acc {age_acc:.2%}")
    return both_loss,age_loss,gender_loss

def test(model, loader,gender_criterion,age_criterion,metric_fn,device=DEVICE,ds_length=len(test_set),accuracy=accuracy):
    model.eval()
    result=[]
    age_accuracy=[]
    target=[]
    for image, (age_gt, gender_gt),(age_id_one_hot,gender_id_one_hot) in loader:
        with torch.no_grad():
            image, _, gender_gt = image.to(device), age_gt.to(device), gender_gt.to(device)
            age_logits,gender_logits = model(image)
            gender_result=torch.argmax(gender_logits,dim=1)
            result.extend(gender_result.cpu().numpy().tolist())
            target.extend(gender_gt.cpu().numpy().tolist())
            age_accuracy.append(accuracy(age_logits,age_gt.to(DEVICE)).item())
            
    print("gender results: ")
    metric_fn(result,target)
    print(f"age accuracy: ", np.sum(np.array(age_accuracy))/ds_length)



def train(epoch,model,loader,gender_criterion,age_criterion,sch,accuracy=accuracy,subfolder='agegender_model'):
  for batch_idx, (image, (age_gt, gender_gt),(age_id_one_hot,gender_id_one_hot)) in enumerate(train_loader):
        optimizer.zero_grad()
        #optimizer.zero_grad()
        image, age_gt, gender_gt = image.to(DEVICE), age_gt.to(DEVICE), gender_gt.to(DEVICE)
        age_logits, gender_logits = model(image)
        loss_age = age_criterion(age_logits, age_id_one_hot.to(DEVICE).float())  # bce
        loss_gender = gender_criterion(gender_logits, gender_id_one_hot.to(DEVICE).float())  # softmax+ce
        loss = (loss_age + loss_gender) / 2
        loss.backward()
        optimizer.step()
        sch.step()
        if batch_idx % 10 == 0:
          print(loss.cpu().detach().numpy())


  gender_acc = accuracy(gender_logits, gender_gt)
  age_acc = accuracy(age_logits, age_gt)
  print(f"last_batch: epoch:{epoch}, gender: {gender_acc/gender_logits.shape[0]}, age: {age_acc/age_logits.shape[0]}")
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
        self.age_head = nn.Linear(512, age_classes)
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
        age_logits = self.age_head(x)
        gender_logits = self.gender_head(x)
        return age_logits, gender_logits
    
model = AgeGender(None,3, age_classes=8, gender_classes=2).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000])
gender_criterion = nn.CrossEntropyLoss()
age_criterion = nn.CrossEntropyLoss()
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
      save_network(model, 'agegender_model','both')
      best_both_loss=both_loss
  if age_loss<best_age_loss:
      save_network(model, 'agegender_model','age')
      best_age_loss=age_loss
  if gender_loss<best_gender_loss:
      save_network(model, 'agegender_model','gender')
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
  best_model = AgeGender(None,3, age_classes=8, gender_classes=2).to(DEVICE)
  best_model.load_state_dict(torch.load(f'agegender_model/{best_model_idx}.pt'))
  test(best_model, test_loader,gender_criterion,age_criterion,metric_fn=test_metric,device=DEVICE)

#print
model_name='agegender'
for loss_list in [both_loss_list,gender_loss_list,age_loss_list]:
    plt.plot([i for i in range(len(loss_list))],loss_list)
    
plt.savefig(f'graph/{model_name}.png')