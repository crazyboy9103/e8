from re import template
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn 
import argparse


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

parser = argparse.ArgumentParser(description="train efficientnet-b0")
parser.add_argument("--model", default="eff_net.pt", type=str, help="model name to load from")
args = parser.parse_args()

transforms = transforms.Compose([
    transforms.ToTensor()
])

test_dir  = './dataset'
testset = ImageFolderWithPaths(root=test_dir, transform=transforms, target_transform=None)
testloader = DataLoader(testset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)

PATH = args.model
dataiter = iter(testloader) 


def build_net(num_classes):
    net = models.efficientnet_b0(pretrained=True)
    num_ftrs = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return net

net = build_net(len(testset.classes))
net.load_state_dict(torch.load(PATH)) 
net.eval()


seen_labels = []
for image, label, path in testloader:
    if len(seen_labels) == 3:
        break
    if label not in seen_labels:
        output = net(image)
        _, predicted = torch.max(output, 1) 
        pred = testset.classes[predicted.item()]
        seen_labels.append(label)
        label = testset.classes[label]
        npimg = image.numpy()
        plt.figure()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig(f"label_{label}_pred_{pred}.png", dpi=600)
        plt.clf()



#outputs = net(images)
#_, predicted = torch.max(outputs, 1) 
#print('Predicted: ', ' '.join('%5s' %  testset.classes[predict] for predict in predicted))
