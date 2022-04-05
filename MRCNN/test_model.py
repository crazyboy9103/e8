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
class MyModel(Model):
    def test(self, dataset):
        print("dataset len", len(dataset))
        if "test_idx.npy" not in os.listdir():
            test_idx = np.random.choice(len(dataset), len(dataset)//10, replace=False)
            np.save("test_idx.npy", test_idx)
        else:
            test_idx = np.load("test_idx.npy")

        test_set = torch.utils.data.Subset(dataset, test_idx)
        print("subset len", len(test_set))
        f = open("test_mrcnn.csv", "w", newline='')
        csv_writer = csv.writer(f)
        testloader = DataLoader(dataset = test_set, batch_size=self.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)
        full_image_names = dataset.images
        filenames = [full_image_names[idx] for idx in test_idx]
        for row in filenames:
            csv_writer.writerow([row])
        f.close()
        print("test dataset list saved 'test_mrcnn.csv'")
        evaluate(self.model, filenames, 1, testloader, device=self.device)
def getTimestamp():
    import time, datetime
    timezone = 60*60*9 # seconds * minutes * utc + 9
    utc_timestamp = int(time.time() + timezone)
    date = datetime.datetime.fromtimestamp(utc_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return date
def compute_iou(cand_mask, gt_mask):
    #gt_mask = torch.zeros((H_new, W_new), dtype=torch.bool)
    #mask = torch.zeros((H_new, W_new), dtype=torch.bool)
    #gt_poly = np.array(gt_poly).reshape(len(gt_poly)//2, 2)
    #cand_poly = np.array(cand_poly).reshape(len(cand_poly)//2, 2)
    #cv2.fillPoly(img=gt_mask, pts=[gt_poly], color=(1,1,1))
    #cv2.fillPoly(img=mask, pts=[cand_poly], color=(1,1,1))

    gt_mask = gt_mask.bool().numpy()
    mask = cand_mask.bool().numpy()

    intersection = np.logical_and(mask, gt_mask)
    union = np.logical_or(mask, gt_mask)

    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
def convert_mask_to_poly(mask):
    if len(mask.shape) == 3:
        mask = torch.squeeze(mask)
    mask = mask.numpy()
    #print("mask", mask.shape)
    coords = np.column_stack(np.where(mask > 0))
    #print("coords", coords.shape)
    coords = coords.tolist()
    return coords

import sys
def evaluate(model, image_names, epoch, data_loader, device):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')    
    model.eval()
    model.to(device)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    logs = {"start":getTimestamp()}
    IOUs = {}
    APs = {}
    #accs = {}
    for batch_idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        images = list(img.to(device) for img in images)
        preds = model(images)
        images_names = [image_names[j] for j in range(batch_idx * 8, (batch_idx+1) * 8) if j < len(image_names)]
        for i, image in enumerate(images):
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
                #logs[image_id][class_name]["gt_polys"] = []
                logs[image_id][class_name]['gt_label'] = []
                logs[image_id][class_name]['label'] = []
                #logs[image_id][class_name]['polys'] = []
                logs[image_id][class_name]['conf'] = []
                logs[image_id][class_name]['iou'] = []
                logs[image_id][class_name]['correct'] = []

            #for label in gt_labels:
            #    logs[image_id][class_name]['gt_label'].append(decode[label.item()])
            
            #for mask in gt_masks:
                #poly = convert_mask_to_poly(mask)
                #logs[image_id][class_name]["gt_polys"].append(convert_mask_to_poly(mask))
            
            for j, gt_mask in enumerate(gt_masks):
                for k, mask in enumerate(masks):
                    iou = compute_iou(mask, gt_mask)
                    if iou > 0.3:
                       logs[image_id][class_name]['iou'].append(iou)
                       logs[image_id][class_name]['correct'].append(gt_label == labels[k].item())
                       logs[image_id][class_name]['gt_label'].append(decode[gt_label])
                       logs[image_id][class_name]['label'].append(decode[labels[k].item()])
                       logs[image_id][class_name]['conf'].append(float(scores[k]))

            #pred_result = {}
            # 같은 label 끼리 묶음
            #for j, label in enumerate(labels):
            #    label = label.item()
            #    logs[image_id][class_name]['label'].append(decode[label])
                #logs[image_id][class_name]['polys'].append(convert_mask_to_poly(masks[j]))
            #    logs[image_id][class_name]['conf'].append(float(scores[j]))
                    
        
            #del masks, gt_masks
        #del images, preds
                
    logs["end"]=getTimestamp()
    import json
    with open(f"detailed_metrics.json", "w") as f:
        json.dump(logs, f, ensure_ascii=False)
    
    # gather the stats from all processes
    #metric_logger.synchronize_between_processes()

import argparse
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--data', default="mrcnn_data.pt", type=str, help="dataset.pt filename")
parser.add_argument('--model', default="mrcnn_model_75.pt", type=str, help="mrcnn_model.pt filename")
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     
dataset = CustomDataset("/dataset", args.data)
#dataset.labels = {i:dataset.labels[i] for i in range(1000)}
#dataset.images = {i:dataset.images[i] for i in range(1000)}
myModel = MyModel(num_classes=dataset.num_classes, device = device, model_name = args.model, batch_size=8, parallel=False) # if there is no ckpt to load, pass model_name=None 
myModel.test(dataset)
