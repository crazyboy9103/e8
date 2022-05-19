from SSD import *
import json
import torchvision 
dic = json.load(open("dic.json","r"))
label_map = json.load(open("labels.json", "r"))
decode = {}
for k, v in label_map.items():
    decode[v] = k

import numpy as np
import torch
import utils
from sklearn.metrics import average_precision_score, confusion_matrix
import csv
class MyModel(Model):
    def test(self, dataset):
        #if "test_idx.npy" not in os.listdir():
        #    test_idx = np.random.choice(len(dataset), len(dataset)//10, replace=False)
        #    np.save("test_idx.npy", test_idx)
        #else:
        #    test_idx = np.load("test_idx.npy")

        #test_set = torch.utils.data.Subset(dataset, test_idx)
        f = open("test_ssd.csv", "w", newline='', encoding="utf-8-sig")
        csv_writer = csv.writer(f)
        testloader = DataLoader(dataset = dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)
        full_image_names = dataset.images
        filenames = full_image_names#[full_image_names[idx] for idx in test_idx]
        for row in filenames:
            csv_writer.writerow([row])
        f.close()
        print("test dataset list saved 'test_ssd.csv'")
        evaluate(self.model, filenames, testloader)

def getTimestamp():
    import time, datetime
    timezone = 60*60*9 # seconds * minutes * utc + 9
    utc_timestamp = int(time.time() + timezone)
    date = datetime.datetime.fromtimestamp(utc_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return date
def evaluate(model, image_names, data_loader):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')    
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    logs = {"start":getTimestamp()}
    for batch_idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        images = list(img.to(device) for img in images)
        with torch.no_grad():
            preds = model(images)
        images_names = [image_names[j] for j in range(batch_idx * len(images), (batch_idx+1) * len(images)) if j < len(image_names)]
        for i, (image, target) in enumerate(zip(images, targets)):
            pred = preds[i]
            boxes = pred['boxes'].detach().cpu()
            labels = pred['labels'].detach().cpu()
            scores = pred['scores'].detach().cpu()
            image_id = images_names[i]

            if len(labels) == 0: # if no label, nothing to evaluate	
                continue	

            if image_id not in logs:
                logs[image_id] = {}

            gt_boxes = target['boxes']
            gt_labels = target['labels']
            gt_label = gt_labels[0].item()
            class_name = decode[gt_label]

            if class_name not in logs[image_id]:
                logs[image_id][class_name] = {}
                logs[image_id][class_name]["gt_bbox"] = []
                logs[image_id][class_name]['gt_label'] = []
                logs[image_id][class_name]['label'] = []
                logs[image_id][class_name]['bbox'] = []
                logs[image_id][class_name]['conf'] = []

            for label in gt_labels:
                logs[image_id][class_name]['gt_label'].append(decode[label.item()])
            
            for box in gt_boxes:
                logs[image_id][class_name]["gt_bbox"].append(box.tolist())
            
            for j, label in enumerate(labels):
                label = label.item()
                logs[image_id][class_name]['label'].append(decode[label])
                logs[image_id][class_name]['bbox'].append(boxes[j].tolist())
                logs[image_id][class_name]['conf'].append(float(scores[j]))
		

    logs["end"]=getTimestamp()
    
    with open(f"detailed_metrics.json", "w") as f:
        json.dump(logs, f, ensure_ascii=False)
    

import argparse
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--data', default="ssd_data.pt", type=str, help="dataset.pt filename")
parser.add_argument('--model', default="ssd_model_140.pt", type=str, help="ssd_model.pt filename")
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     
dataset = CustomDataset("/dataset/48g_dataset", args.data)
myModel = MyModel(num_classes=dataset.num_classes, device = device, model_name = args.model, batch_size=128, parallel=False) # if there is no ckpt to load, pass model_name=None 
myModel.test(dataset)
