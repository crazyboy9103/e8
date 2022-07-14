from MRCNN import *
import json
import torchvision 
dic = json.load(open("dic.json","r"))
label_map = json.load(open("labels.json", "r"))
decode = {}
for k, v in label_map.items():
    decode[v] = k

import os 
import numpy as np
import torch
import utils
from sklearn.metrics import average_precision_score, confusion_matrix
import csv
import time, datetime
class MyModel(Model):
    def test(self, dataset):
        # print("dataset len", len(dataset))
        # if "test_idx.npy" not in os.listdir():
        #     test_idx = np.random.choice(len(dataset), len(dataset)//10, replace=False)
        #     np.save("test_idx.npy", test_idx)
        # else:
        #     test_idx = np.load("test_idx.npy")

        # test_set = torch.utils.data.Subset(dataset, test_idx)
        # print("subset len", len(test_set))
        f = open("test_mrcnn.csv", "w", newline='', encoding="utf-8-sig")
        csv_writer = csv.writer(f)
        testloader = DataLoader(dataset = dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)
        full_image_names = dataset.images
        filenames = full_image_names
        for row in filenames:
            csv_writer.writerow([row])
        f.close()
        print("test dataset list saved 'test_mrcnn.csv'")
        evaluate(self.model, filenames, testloader)

def getTimestamp():
    timezone = 60*60*9 # seconds * minutes * utc + 9
    utc_timestamp = int(time.time() + timezone)
    date = datetime.datetime.fromtimestamp(utc_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return date

def compute_iou(cand_mask, gt_mask):
    gt_mask = gt_mask.bool().numpy()
    mask = cand_mask.bool().numpy()

    intersection = np.logical_and(mask, gt_mask)
    union = np.logical_or(mask, gt_mask)

    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
    
# def convert_mask_to_poly(mask):
#     if len(mask.shape) == 3:
#         mask = torch.squeeze(mask)
#     mask = mask.numpy()
#     coords = np.column_stack(np.where(mask > 0))
#     coords = coords.tolist()
#     return coords

import sys
def evaluate(model, image_names, data_loader):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')    
    model.eval()
    model.to(device)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    logs = {"start":getTimestamp()}

    for batch_idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        images = list(img.to(device) for img in images)
        with torch.no_grad():
            preds = model(images)
        images_names = [image_names[j] for j in range(batch_idx * len(images), (batch_idx+1) * len(images)) if j < len(image_names)]
        for i, (image, target) in enumerate(zip(images, targets)):
            pred = preds[i]
            masks = pred['masks'].detach().cpu()
            labels = pred['labels'].detach().cpu()
            scores = pred['scores'].detach().cpu()
            image_id = images_names[i]

            if len(labels) == 0: # if no label, nothing to evaluate	
                continue	

            if image_id not in logs:
                logs[image_id] = {}

            target = targets[i]
            gt_masks = target['masks']
            gt_labels = target['labels']
            gt_label = gt_labels[0].item()
            class_name = decode[gt_label]

            if class_name not in logs[image_id]:
                logs[image_id][class_name] = {}
                logs[image_id][class_name]['gt_label'] = []
                logs[image_id][class_name]['label'] = []
                logs[image_id][class_name]['conf'] = []
                logs[image_id][class_name]['iou'] = []
                logs[image_id][class_name]['correct'] = []
                logs[image_id][class_name]["time"] = []
                logs[image_id][class_name]["FP"] = []
                logs[image_id][class_name]["FN"] = []
                logs[image_id][class_name]["TP"] = []
            
            for label in gt_labels:
                logs[image_id][class_name]["gt_label"].append(decode[label.item()])
            
            
            for j, gt_mask in enumerate(gt_masks):
                best_iou = -9999
                best_idx = None
                for k, mask in enumerate(masks):
                    iou = compute_iou(mask, gt_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = k

                logs[image_id][class_name]['gt_label'].append(class_name)
                logs[image_id][class_name]['label'].append(decode[labels[best_idx].item()])
                logs[image_id][class_name]['conf'].append(float(scores[best_idx]))
                logs[image_id][class_name]['iou'].append(best_iou)

                correct = gt_label == labels[best_idx].item() if best_iou > 0.2 else False

                logs[image_id][class_name]["correct"].append(correct)
                logs[image_id][class_name]["time"].append(getTimestamp())

                FP = (labels[best_idx].item() == gt_label) == True and best_iou < 0.2
                FN = (labels[best_idx].item() == gt_label) == False
                TP = (labels[best_idx].item() == gt_label) == True and best_iou > 0.2

                logs[image_id][class_name]["FP"].append(int(FP))
                logs[image_id][class_name]["FN"].append(int(FN))
                logs[image_id][class_name]["TP"].append(int(TP))

                    

                
    logs["end"]=getTimestamp()
    import json
    with open(f"detailed_metrics.json", "w") as f:
        json.dump(logs, f, ensure_ascii=False)


import argparse
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--data', default="mrcnn_data.pt", type=str, help="dataset.pt filename")
parser.add_argument('--model', default="mrcnn_model_85.pt", type=str, help="mrcnn_model.pt filename")
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     
dataset = CustomDataset("/dataset/48g_dataset", args.data)
myModel = MyModel(num_classes=dataset.num_classes, device = device, model_name = args.model, batch_size=32, parallel=False) # if there is no ckpt to load, pass model_name=None 
myModel.test(dataset)
