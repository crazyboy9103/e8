from re import template
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn 
import argparse
import json
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


def getTimestamp():
    import time, datetime
    timezone = 60*60*9 # seconds * minutes * utc + 9
    utc_timestamp = int(time.time() + timezone)
    date = datetime.datetime.fromtimestamp(utc_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return utc_timestamp
parser = argparse.ArgumentParser(description="train efficientnet-b0")
parser.add_argument("--model", default="eff_net.pt", type=str, help="model name to load from")
args = parser.parse_args()

transforms = transforms.Compose([
    transforms.ToTensor()
])

test_dir  = './dataset'
testset = ImageFolderWithPaths(root=test_dir, transform=transforms, target_transform=None)
class_to_idx = testset.class_to_idx
idx_to_class = {v:k for k, v in class_to_idx.items()}
testloader = DataLoader(testset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)

dataiter = iter(testloader) 
images, labels, paths = dataiter.next() # 실험용 데이터와 결과 출력 
def imsave(img):
    npimg = img.numpy()
    plt.figure(1, figsize=(12, 12))
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.savefig("evaluate.png", dpi=600)
    plt.clf()
imsave(torchvision.utils.make_grid(images)) 
print('GroundTruth: ', ' '.join('%5s' % testset.classes[label] for label in labels)) # 학습한 모델로 예측값 뽑아보기 

def build_net(num_classes):
    net = models.efficientnet_b0(pretrained=True)
    num_ftrs = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return net

net = build_net(len(testset.classes))
PATH = args.model
net.load_state_dict(torch.load(PATH)) 
#outputs = net(images)
#_, predicted = torch.max(outputs, 1) 
#print('Predicted: ', ' '.join('%5s' %  testset.classes[predict] for predict in predicted))

logs = {"start":getTimestamp()}
stats_by_class = {idx_to_class[i]:{"correct":0, "total":0} for i in range(3)} #11 classes
labels_by_class = {idx_to_class[i]:[] for i in range(3)}
preds_by_class = {idx_to_class[i]:[] for i in range(3)}

net.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
with torch.no_grad(): 
    for img_idx, (image, label, path) in enumerate(tqdm(testloader)): 
        output = net(image.to(device))
        #print(output)
        _, predicted = torch.max(output.cpu(), 1) 
        
        img_name = path
        temp_label = idx_to_class[label.item()]
        temp_predict = idx_to_class[predicted.item()]
        is_correct = temp_label == temp_predict

        #print("predict", temp_predict)
        #print("label", temp_label)

    
        labels_by_class[temp_label].append(temp_label)
        preds_by_class[temp_label].append(temp_predict)

        
    """    if img_idx % 20 == 0:
            str_buffer = f"== Image Index {img_idx}=="
            for i in range(3):
                labels = labels_by_class[i]
                preds = labels_by_class[i]
                try:2
                    acc = accuracy_score(labels, preds)
                    f1 = f1_score(labels, preds)
                    str_stats = f"Class {i}, Accuracy:{acc}, F1: {f1}"
                    str_buffer = f"{str_buffer}\n{str_stats}\n"
                except: 
                    str_stats = f"Class {i}, Accuracy:NaN, F1: NaN"
                    str_buffer = f"{str_buffer}\n{str_stats}\n"
            
            print(str_buffer)"""
    
    for i in range(3):
        labels = labels_by_class[idx_to_class[i]]
        preds = labels_by_class[idx_to_class[i]]
        logs["class_stats"][idx_to_class[i]] = {"acc": accuracy_score(labels, preds), "f1":f1_score(labels, preds, average="micro")}
    
    mean_acc = np.mean(list(logs["class_stats"][idx_to_class[i]]["acc"] for i in range(3)))
    mean_f1 = np.mean(list(logs["class_stats"][idx_to_class[i]]["f1"] for i in range(3)))

    logs["final_stats"] = {"mean_acc":mean_acc, "mean_f1":mean_f1}

logs["end"] = getTimestamp()

json.dump(logs, open("detailed_metrics.json", "w"))
