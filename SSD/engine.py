import torch
import warnings
warnings.filterwarnings("ignore")
import math
import sys
import numpy as np
import torch
import utils
from sklearn.metrics import average_precision_score, confusion_matrix

import json
dic = json.load(open("dic.json","r"))
label_map = json.load(open("labels.json", "r"))

decode = {}
for k, v in label_map.items():
    decode[v] = k

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.001 * (0.97 ** epoch)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        model.train()
        losses = sum(loss for loss in loss_dict.values())
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    return metric_logger

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def evaluate(model, epoch, data_loader, device):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    logs = {}
    maskIOUs = {}
    IOUs = {}
    APs = {}
    accs = {}
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        preds = model(images)
        for i, image in enumerate(images):
            pred = preds[i]
            boxes = pred['boxes'].detach().cpu()
            labels = pred['labels'].detach().cpu()
            scores = pred['scores'].detach().cpu()
            image_id = pred['image_id'].detach().cpu()

            if len(labels) == 0: # if no label, nothing to evaluate
                continue

            if image_id in logs:
                pass
            else:
                logs[image_id] = {}
            
            target = targets[i]
            gt_boxes = target['boxes']
            gt_labels = target['labels']
            gt_label = gt_labels[0].item()
            class_name = decode[gt_label]
            if class_name in logs[image_id]:
                pass
            else:
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

            pred_result = {}
            for j, label in enumerate(labels):
                label = label.item()
                logs[image_id][class_name]['label'].append(decode[label])
                logs[image_id][class_name]['bbox'].append(boxes[j].tolist())
                logs[image_id][class_name]['conf'].append(float(scores[j]))

                if label in pred_result:
                    pred_result[label]['scores'].append(float(scores[j]))
                    pred_result[label]['boxes'].append(boxes[j])
                else:
                    pred_result[label]={'scores':[float(scores[j])], 'boxes':[boxes[j]]}

        
            
            for label, output in pred_result.items():
                N = len(output['scores'])
                temp_boxes = torch.zeros((N, 4))

                for k, box in enumerate(output['boxes']):
                    temp_boxes[k, :] = box

                pred_result[label]['boxes'] = temp_boxes
                
            for j, (label, output) in enumerate(pred_result.items()):
                temp_boxes = [gt_box for gt_box, gt_label in zip(gt_boxes, gt_labels) if label == gt_label]
                if not temp_boxes: # if there are no gt boxes, no need to evaluate
                    continue
                
                # stack gt boxes
                gt_label_boxes = torch.zeros((len(temp_boxes), 4))
                for k, box in enumerate(temp_boxes):
                    gt_label_boxes[k, :] = box

                # calculate box iou between dt boxes and gt boxes
                box_iou = torchvision.ops.box_iou(output['boxes'], gt_label_boxes)
                box_iou_d1 = torch.max(box_iou, dim=1) # tensor of maximum ious with each gt box 
                pred_classes = []

                
                # for each iou, take 
                # if iou > 0.5: correct detection
                # else: incorrect
                for temp_iou in box_iou_d1.values:
                    temp_iou = float(temp_iou)
                    if temp_iou > 0.3:
                        pred_classes.append(True)
                        logs[image_id][class_name][j]["IOU"] = temp_iou

                        if label in IOUs:
                            IOUs[label].append(temp_iou)
                        else:
                            IOUs[label] = [temp_iou]
                    else:
                        pred_classes.append(False)

                #tp, fn, fp, tn = confusion_matrix([True for _ in range(len(pred_classes))], pred_classes).ravel()
                
                #logs[image_id][class_name][j]['tp'] = tp
                #logs[image_id][class_name][j]['fp'] = fp
                #logs[image_id][class_name][j]['fn'] = fn
                #logs[image_id][class_name][j]['tn'] = tn

                AP = average_precision_score(pred_classes, output['scores'])
                # if all pred_classes are False, AP is nan
                if not np.isnan(AP):
                    logs[image_id][class_name][j]["AP"] = AP
                    
                    if label in APs:
                        APs[label].append(AP)
                        
                    else:
                        APs[label] = [AP]
                
                # if nothing was detected, 0 % 
                if not pred_classes:
                    acc = 0
                
                else:
                    acc = sum([int(correct) for correct in pred_classes])/len(pred_classes)

                # append to accs Dict{label:[List]}
                if label in accs:
                    accs[label].append(acc)
               
                else:
                    accs[label] = [acc]
            if i == 0:
                print(logs)
    mIOU = {decode[int(k)]:np.mean(v) for k, v in IOUs.items()}
    mAP = {decode[int(k)]:np.mean(v) for k, v in APs.items()}
    meanAcc = {decode[int(k)]:np.mean(v) for k, v in accs.items()}
    
    metrics = {"mIOU": mIOU, "mAP": mAP, "meanAcc":meanAcc}
    import json
    with open(f"metrics_{epoch}.json", "w") as f:
        json.dump(metrics, f)
    del metrics

    with open(f"detailed_metrics_{epoch}.json", "w") as f:
        json.dump(logs, f)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    del mIOU, mAP, meanAcc, images
