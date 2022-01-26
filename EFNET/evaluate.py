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
def getTimestamp():
    import time, datetime
    timezone = 60*60*9 # seconds * minutes * utc + 9
    utc_timestamp = int(time.time() + timezone)
    date = datetime.datetime.fromtimestamp(utc_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return utc_timestamp
parser = argparse.ArgumentParser(description="train efficientnet-b0")
parser.add_argument("--model", default="eff_net.pt", type=str, help="model name to load from")
parser.add_argument("--batch", default=1, type=int, help="batch size")
args = parser.parse_args()

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)) # H, W
])

test_dir  = '../dataset/val'
testset = ImageFolder(root=test_dir, transform=transforms, target_transform=None)
testloader = DataLoader(testset, batch_size=args.batch, shuffle=False, pin_memory=True, num_workers=4)

PATH = args.model
dataiter = iter(testloader) 
images, labels = dataiter.next() # 실험용 데이터와 결과 출력 
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
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_classes)
    return net

net = build_net(len(testset.classes))
net.load_state_dict(torch.load(PATH)) 
#outputs = net(images)
#_, predicted = torch.max(outputs, 1) 
#print('Predicted: ', ' '.join('%5s' %  testset.classes[predict] for predict in predicted))

logs = {"start":getTimestamp()}

correct = 0 
total = 0 
f1 = 0

from sklearn.metrics import f1_score
image_names = list(map(lambda img: img[0], testset.imgs))
with torch.no_grad(): 
    for batch_idx, (images, labels) in enumerate(testloader): 
        outputs = net(images) 
        images_names = [image_names[j] for j in range(batch_idx * args.batch, (batch_idx + 1) * args.batch) if j < len(image_names)]
        _, predicted = torch.max(outputs.data, 1) 
        for img_name, predict, label in zip(images_names, predicted, labels):
            logs[img_name] = {"predict":predict.item(), "gt": label.item(), "correct":predict.item()==label.item()}
            
        f1 += f1_score(labels.numpy(), predicted.numpy())
        total += labels.size(0) 
        correct += (predicted == labels).sum().item() 
        print('Accuracy of the network on the test images: %d %%' % ( 100 * correct / total))
        print("Average F1 score : %f %%" % (f1 / total))
    
    logs["stats"] = {"acc":correct / total, "f1": f1 / total}

logs["end"] = getTimestamp()

json.dump(logs, open("detailed_metrics.json", "w"))


