from MRCNN import *
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
        test_idx = np.random.choice(len(dataset), len(dataset)//10, replace=False)
        test_data = torch.utils.data.Subset(dataset, test_idx)
        f = open("test_mrcnn.csv", "w")
        csv_writer = csv.writer(f)
        #print(dir(test_data.dataset))
        #print(test_data.labels)
        csv_writer.writerows(test_data.dataset.images.values())
        f.close()
        print("test dataset list saved 'test_mrcnn.csv'")
        testloader = DataLoader(dataset = test_data.dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)
        evaluate(self.model, test_data.dataset.images, 1, testloader, device=self.device)
def getTimestamp():
    import time, datetime
    timezone = 60*60*9 # seconds * minutes * utc + 9
    utc_timestamp = int(time.time() + timezone)
    date = datetime.datetime.fromtimestamp(utc_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return date
def evaluate(model, image_names, epoch, data_loader, device):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')    
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    logs = {"start":getTimestamp()}
    IOUs = {}
    APs = {}
    #accs = {}
    for batch_idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        images = list(img.to(device) for img in images)
        preds = model(images)
        images_names = [image_names[j] for j in range(batch_idx * 8, (batch_idx+1) * 8) if j in image_names]
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
                logs[image_id][class_name]["gt_masks"] = []
                logs[image_id][class_name]['gt_label'] = []
                logs[image_id][class_name]['label'] = []
                logs[image_id][class_name]['masks'] = []
                logs[image_id][class_name]['conf'] = []

            for label in gt_labels:
                logs[image_id][class_name]['gt_label'].append(decode[label.item()])
            
            for mask in gt_masks:
                logs[image_id][class_name]["gt_masks"].append(mask)
                
            pred_result = {}
            # 같은 label 끼리 묶음
            for j, label in enumerate(labels):
                label = label.item()
                logs[image_id][class_name]['label'].append(decode[label])
                logs[image_id][class_name]['masks'].append(masks[j].tolist())
                logs[image_id][class_name]['conf'].append(float(scores[j]))


#                if label in pred_result:
#                    pred_result[label]['scores'].append(float(scores[j]))
#                    pred_result[label]['masks'].append(masks[j])
#                else:
#                    pred_result[label]={'scores':[float(scores[j])], 'masks':[masks[j]]}

         # 마스크 N * W * H 텐서로 묶음
            # for label, output in pred_result.items():
            #     N = len(output['scores'])             
            #     temp_masks = torch.zeros((N, 480, 360), dtype=torch.bool)

            #     for k, mask in enumerate(output['masks']):
            #         temp_masks[k, :, :] = mask.bool()

            #     pred_result[label]['masks'] = temp_masks
            
            # 
            # for j, (label, output) in enumerate(pred_result.items()):
            #     temp_masks = [gt_mask for gt_mask, gt_label in zip(gt_masks, gt_labels) if label == gt_label]
                
            #     #skip if no gt mask 
            #     if not temp_masks:
            #         continue
                
            #     gt_label_masks = torch.zeros((len(temp_masks), 480, 360), dtype=torch.bool)
            #     for k, mask in enumerate(temp_masks):
            #         gt_label_masks[k, :, :] = mask

                
            
            #     for mask in temp_masks:
            #         mask = mask.bool().numpy()
            #         for gt_mask in output['masks'].numpy():
            #             intersection = np.logical_and(mask, gt_mask)
            #             union = np.logical_or(mask, gt_mask)

            #             iou_score = np.sum(intersection) / np.sum(union)
                        
            #             logs[image_id][class_name]['maskiou'] = []

            #             if iou_score > 0.2:
            #                 logs[]

    logs["end"]=getTimestamp()
    import json
    with open(f"detailed_metrics.json", "w") as f:
        json.dump(logs, f, ensure_ascii=False)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

import argparse
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--data', default="mrcnn_data.pt", type=str, help="dataset.pt filename")
parser.add_argument('--model', default="mrcnn_model_75.pt", type=str, help="mrcnn_model.pt filename")
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     
dataset = CustomDataset("/dataset/data_1230", args.data)
#dataset.labels = {i:dataset.labels[i] for i in range(1000)}
#dataset.images = {i:dataset.images[i] for i in range(1000)}
myModel = MyModel(num_classes=dataset.num_classes, device = device, model_name = args.model, batch_size=8, parallel=False) # if there is no ckpt to load, pass model_name=None 
myModel.test(dataset)
