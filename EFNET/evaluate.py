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
import os
import csv

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
    return date

parser = argparse.ArgumentParser(description="train efficientnet-b0")
parser.add_argument("--dataset", default="./dataset", type=str, help="dataset path")
parser.add_argument("--model", default="eff_net.pt", type=str, help="model name to load from")
args = parser.parse_args()

transforms = transforms.Compose([
    transforms.ToTensor()
])

test_dir  = args.dataset
dataset = ImageFolderWithPaths(root=test_dir, transform=transforms, target_transform=None)
model_name = args.model.strip(".pt")
idx_filename = f"{model_name}_idx.npy"

if idx_filename not in os.listdir():
    test_idx = np.random.choice(len(dataset), len(dataset)//10, replace=False)
    np.save(idx_filename, test_idx)
else:
    test_idx = np.load(idx_filename)

testset = torch.utils.data.Subset(dataset, test_idx)
f = open(f"test_{model_name}.csv", "w", newline='', encoding="utf-8-sig")
csv_writer = csv.writer(f)
testloader = DataLoader(testset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)

full_image_names = dataset.imgs
filenames = [full_image_names[idx] for idx in test_idx]
for row in filenames:
    csv_writer.writerow([row[0]])
f.close()

PATH = args.model


def build_net(num_classes):
    net = models.efficientnet_b0(pretrained=True)
    num_ftrs = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return net

net = build_net(3)
net.load_state_dict(torch.load(PATH))
device = torch.device("cuda:0")
net.to(device)
net.eval()

logs = {"start":getTimestamp()}

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
# Run inference image by image, accumulating image names, predictions, times 
with torch.no_grad():
    for img_idx, (image, label, path) in enumerate(tqdm(testloader, desc="evaluating")):
        output = net(image.to(device))

        _, predicted = torch.max(output, 1)
        path = path[0]
        img_name = path
        temp_label = dataset.classes[label.item()]
        if temp_label not in logs:
            logs[temp_label] = {}
            logs[temp_label]["image_names"] = []
            logs[temp_label]["pred_labels"] = []
            logs[temp_label]["times"] = []
        

        temp_predict = dataset.classes[predicted.item()]

        logs[temp_label]["image_names"].append(img_name)    
        logs[temp_label]["pred_labels"].append(temp_predict)
        logs[temp_label]["times"].append(getTimestamp())

logs["end"] = getTimestamp()

from openpyxl import Workbook
from sklearn.metrics import multilabel_confusion_matrix as mcm
from itertools import accumulate

def write_to_json(logs):
    # Open workbook
    # wb = Workbook(write_only=True)
    # ws = wb.create_sheet(model_name)

    model_result = {"image_names": [], "gt_labels":[], "pred_labels": [], "times": [], "totals": [], "corrects": [], "cum_corrects": []}

    labels = []
    preds = []
    for key, item in logs.items():
        if key not in ["start", "end"]:
            image_names = item["image_names"]
            N = len(image_names)

            label = key
            gt_labels = [label for _ in range(N)]
            pred_labels = item["pred_labels"]
            
            # Accumulate them for multilabel confusion matrix
            labels.extend(gt_labels)
            preds.extend(pred_labels)

            times = item["times"]
            totals = [i+1 for i in range(N)]
            corrects = [label == pred_label for pred_label in pred_labels]
            cum_corrects = list(accumulate([int(correct) for correct in corrects]))
            if model_result["totals"]:
                last = model_result["totals"][-1]
                totals = [total + last for total in totals]
            
            if model_result["cum_corrects"]:
                last = model_result["cum_corrects"][-1]
                cum_corrects = [cum_correct + last for cum_correct in cum_corrects]
            

            model_result["image_names"].extend(image_names)
            model_result["gt_labels"].extend(gt_labels)
            model_result["pred_labels"].extend(pred_labels)
            
            model_result["times"].extend(times)
            model_result["totals"].extend(totals)
            
            model_result["corrects"].extend(corrects)
            
            model_result["cum_corrects"].extend(cum_corrects)

        else:
            if key == "start":
                model_result["eval_start"] = item
            elif key == "end":
                model_result["eval_end"] = item
    
    # Multi confusion matrix
    multi_confusion = mcm(labels, preds, labels=["best", "normal", "faulty"])
    best_conf, norm_conf, fault_conf = multi_confusion


    b_tn, b_fp, b_fn, b_tp = best_conf.ravel()
    n_tn, n_fp, n_fn, n_tp = norm_conf.ravel()
    f_tn, f_fp, f_fn, f_tp = fault_conf.ravel()

    b_prec, b_recall = b_tp/(b_tp+b_fp), b_tp/(b_tp+b_fn)
    b_f1 = 2 * (b_prec * b_recall)/(b_prec + b_recall)
    b_acc = (b_tp+b_tn) / (b_tp+b_tn+b_fp+b_fn)

    n_prec, n_recall = n_tp/(n_tp+n_fp), n_tp/(n_tp+n_fn)
    n_f1 = 2 * (n_prec * n_recall)/(n_prec + n_recall)
    n_acc = (n_tp+n_tn) / (n_tp+n_tn+n_fp+n_fn)

    f_prec, f_recall = f_tp/(f_tp+f_fp), f_tp/(f_tp+f_fn)
    f_f1 = 2 * (f_prec * f_recall)/(f_prec + f_recall)
    f_acc = (f_tp+f_tn) / (f_tp+f_tn+f_fp+f_fn)

    stats = [[b_tn, b_fp, b_fn, b_tp, b_prec, b_recall, b_f1, b_acc],\
             [n_tn, n_fp, n_fn, n_tp, n_prec, n_recall, n_f1, n_acc],\
             [f_tn, f_fp, f_fn, f_tp, f_prec, f_recall, f_f1, f_acc]]

    for i, label in enumerate(["best", "normal", "faulty"]):
        for j, c in enumerate(["tn", "fp", "fn", "tp", "prec", "recall", "f1", "acc"]):
            value = stats[i][j]
            key = label + "_" + c
            
        
            if j <= 3:
                value = int(value)
            else:
                value = float(value)
            model_result[key] = value
    
    avg_f1, avg_acc = (b_f1 + n_f1 + f_f1) / 3, (b_acc + n_acc + f_acc) / 3

    model_result["avg_f1"] = float(avg_f1)
    model_result["avg_acc"] = float(avg_acc)

    model_name = args.model.strip(".pt")
    json_filename = model_name + "_eval.json"

    with open(json_filename, "w") as f:
        json.dump(model_result, f)

write_to_json(logs)