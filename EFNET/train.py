import torch 
import torchvision 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import argparse
import json
import numpy as np
from tqdm import tqdm
class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path



transforms = transforms.Compose([
    transforms.ToTensor()
    #transforms.Resize((224, 224)) # H, W
])

parser = argparse.ArgumentParser(description="train efficientnet-b0")
parser.add_argument("--dataset", default="./dataset", type=str, help="dataset folder")
#parser.add_argument("--test", default="dataset/test", type=str, help="test folder")
parser.add_argument("--model", default="eff_net.pt", type=str, help="model name to save")
args = parser.parse_args()

trainset = ImageFolderWithPaths(root=args.dataset, transform=transforms, target_transform=None)

#data_size = len(dataset)
#n_train = int(data_size * 0.9)
#n_valid = data_size
#split_idx = np.random.choice(data_size, data_size, replace=False)
#train_idx = split_idx
#val_idx = split_idx[n_train:n_valid]

#trainset = torch.utils.data.Subset(dataset, train_idx)
#valset = torch.utils.data.Subset(dataset, val_idx)


from torch.utils.data import DataLoader
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)

#valloader = DataLoader(valset, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)

import torch.nn as nn 
import torch.nn.functional as F
import torchvision.models as models
import time
import torch.optim as optim
import copy

from sklearn.metrics import f1_score


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def build_net(num_classes):
    net = models.efficientnet_b0(pretrained=True)
    #print(net.classifier[1].)
    num_ftrs = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return net
#print(dir(trainset))
#net = build_net(len(trainset.classes))
net = build_net(3)
net = net.to(device)
#net.train()
import os
if args.model in os.listdir():
    try:
        net.load_state_dict(torch.load(args.model))
        print(f"model loaded from {args.model}")
    except:
        print("failed to load model, creating new one")
        
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(net.parameters(), lr=0.0001)

def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0
        
        # Iterate over data.
        for batch_idx, (inputs, labels, paths) in enumerate(tqdm(trainloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            with torch.set_grad_enabled(False):
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            #print("running corrects", running_corrects)
            #print(labels.data)
            #print(preds)
                current_f1_score = f1_score(labels.data.cpu(), preds.cpu(), average="micro")

                epoch_loss = running_loss / len(trainset)
                epoch_acc = running_corrects.double() / len(trainset)
            
                if batch_idx == len(trainloader)-1:
                    print('Epoch {} Loss: {:.4f} Acc: {:.4f} F1 Score: {:.4f}'.format(epoch, epoch_loss, epoch_acc, current_f1_score))
            # deep copy the model
                if epoch_acc > best_acc and batch_idx == len(trainloader)-1:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, "eff_net.pt")
                    print("model saved to eff_net.pt")
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(model):
    logs = {}
    model.eval()

    total_f1_score = 0.0
    total_acc = 0.0
    counts = 0
    for inputs, labels, paths in testloader:
        counts += 1
        inputs = inputs.to(device)
        labels = labels.to(device)
       
        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i, path in enumerate(paths):
                #현재 파일에 대한 파일경로
                logs[path] = {"pred":preds[i].item(), "true": labels[i].item()}

        # statistics
        running_corrects += torch.sum(preds == labels.data)

        current_f1_score = f1_score(labels.data.cpu(), preds.cpu())
        total_f1_score += current_f1_score

        epoch_acc = running_corrects.double() / len(testset)
        total_acc += epoch_acc
        print('Acc: {:.4f} F1 Score: {:.4f}'.format(epoch_acc, current_f1_score))

    json.dump(logs, open("efnet_logs.json", "w"))
    print('Final average Acc and F1: {:4f} {:4f}'.format(total_acc/counts, total_f1_score/counts))
    return model
#%% 
model = train_model(model=net, criterion=criterion, optimizer=optimizer, num_epochs=25)
#model = test_model(model)
print('Finished Training')
#torch.save(model.state_dict(), args.model)
