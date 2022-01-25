from calendar import c
import torch 
import torchvision 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)) # H, W
])

"""
example root_dir
    | --- cat/
    |      |-- 0001.jpg
    |      |-- 0002.jpg
    |      |-- 0003.jpg
    |      |-- ...
    |
    | --- dog/
    |      |-- 0001.jpg
    |      |-- 0002.jpg
    |      |-- 0003.jpg
    |      |-- ...
    |
    | --- rabbit/
    |      |--...
"""
train_dir = 'dataset/train'
test_dir  = 'dataset/val'

trainset = ImageFolder(root=train_dir, transform=transforms, target_transform=None)
testset = ImageFolder(root=test_dir, transform=transforms, target_transform=None)
print("class to idx")
print(trainset.class_to_idx)

from torch.utils.data import DataLoader
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, pin_memory=True, num_workers=4)
testloader = DataLoader(testset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)

# test image
dataiter =iter(trainloader)
images, labels = dataiter.next()
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
imshow(torchvision.utils.make_grid(images))
print(trainset.classes[label] for label in labels)

for i, (images, labels) in enumerate(trainloader):
    print(f'batch {i} labels {labels}')


import torch.nn as nn 
import torch.nn.functional as F
import torchvision.models as models
import time
import torch.optim as optim
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def build_net(num_classes):
    net = models.efficientnet_b0(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_classes)
    return net

net = build_net(len(trainset.classes))
net = net.to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
        for inputs, labels in trainloader:
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

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(trainset)
            epoch_acc = running_corrects.double() / len(trainset)

            print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
#%% 
model = train_model(model=net, criterion=criterion, optimizer=optimizer, num_epochs=25)
print('Finished Training')
PATH = './my_net.pth' 
torch.save(model.state_dict(), PATH)

#%%
# test 



