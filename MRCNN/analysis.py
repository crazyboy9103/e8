import numpy as np 
import json
from torch import R
from tqdm import tqdm
import torch
import cv2 


W_new, H_new = 360, 480
def compute_iou(cand_poly, gt_poly):
    gt_mask = torch.zeros((H_new, W_new), dtype=torch.bool)
    mask = torch.zeros((H_new, W_new), dtype=torch.bool)
    
    cv2.fillPoly(img=gt_mask, pts=[gt_poly], color=(1,1,1))
    cv2.fillPoly(img=mask, pts=[cand_poly], color=(1,1,1))

    gt_mask = gt_mask.bool().numpy()
    mask = mask.bool().numpy()

    intersection = np.logical_and(mask, gt_mask)
    union = np.logical_or(mask, gt_mask)

    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

print("Reading detailed_metrics.json...")
f = open("detailed_metrics.json", "r")
metrics = json.load(f)
f.close()
print("Finished reading")

from sklearn.metrics import average_precision_score
def analysis(metrics):
    logs = {}
    print(f"Eval started : {metrics['start']}, Eval ended : {metrics['end']}")
    for image_name, result in tqdm(metrics.items(), desc="reading image result"):
        if isinstance(result, int):
            continue
        if image_name not in logs:
            logs[image_name] = {}
            
        for class_name, stats in result.items():
            if class_name not in logs[image_name]:
                logs[image_name][class_name] = {}
                logs[image_name][class_name]["gt_label"] = []
                logs[image_name][class_name]["label"] = []
                logs[image_name][class_name]["gt_polys"] = []
                logs[image_name][class_name]["polys"] = []
                logs[image_name][class_name]["conf"] = []
                logs[image_name][class_name]["iou"] = []
                logs[image_name][class_name]["correct"] = []
                
            gt_polys = stats['gt_polys']
            gt_label = stats['gt_label']
            pred_polys = stats['polys']
            pred_label = stats['label']
            conf = stats['conf']
            for i in range(len(gt_polys)):
                for j in range(len(pred_polys)):
                    iou = compute_iou(gt_polys[i], pred_polys[j])
                    if iou > 0.3:
                        logs[image_name][class_name]["gt_label"].append(gt_label[i])
                        logs[image_name][class_name]["label"].append(pred_label[j])
                        logs[image_name][class_name]["gt_polys"].append(gt_polys[i])
                        logs[image_name][class_name]['polys'].append(pred_polys[j])
                        logs[image_name][class_name]['conf'].append(conf[j])
                        logs[image_name][class_name]["iou"].append(iou)
                        logs[image_name][class_name]["correct"].append(pred_label[j] == gt_label[i])
                                                

    return logs

def convert_mask_to_poly(mask):
    coords = np.column_stack(np.where(mask > 0))
    return coords
def write_to_excel(metrics):
    from openpyxl import Workbook
    is_correct_by_class = {}
    conf_by_class = {}
    iou_by_class = {}

    wb = Workbook()
    ws = wb.active
    ws.append(["image_name", "correct", "gt_label", "gt_poly", "label", "poly", "conf", "iou"])
    for image_name, result in tqdm(analysis(metrics).items(), desc="image analysis"):
        if isinstance(result, int):
            continue

        for class_name, stats in result.items():
            if class_name not in is_correct_by_class:
                is_correct_by_class[class_name] = []
            
            if class_name not in conf_by_class:
                conf_by_class[class_name] = []
            
            if class_name not in iou_by_class:
                iou_by_class[class_name] = []

            for i in range(len(stats['label'])):
                try:
                    is_correct_by_class[class_name].append(stats['correct'][i])
                    conf_by_class[class_name].append(stats["conf"][i])
                    iou_by_class[class_name].append(stats["iou"][i])

                    ws.append([image_name, str(stats["correct"][i]), str(stats["gt_label"][i]), str(stats['gt_polys'][i]), str(stats['label'][i]), str(stats['polys'][i]), str(stats['conf'][i]), str(stats['iou'][i])])

                except:
                    continue
    
    
    mAP = 0
    mIoU = 0
    count = 0

    wb.create_sheet(index=1, title="mAPandmIOU")
    ws = wb[wb.sheetnames[1]]

    for class_name in is_correct_by_class:
        is_cor = is_correct_by_class[class_name]
        conf = conf_by_class[class_name]
        ious = iou_by_class[class_name]

        class_average_iou = np.mean(ious)
        class_average_ap = average_precision_score(is_cor, conf)

        ws.append([class_name, class_average_iou, class_average_ap])
        print(f"Class: {class_name}, Average IoU: {class_average_iou}, Average Precision: {class_average_ap}")
        mIoU += class_average_iou
        mAP += class_average_ap
        count += 1
    ws.append(["Final mAP", mAP/count, "Final maskIou", mIoU/count])
    print(f"Final mAP :{mAP/count}, Final mIoU : {mIoU/count}")
    wb.save("test.xlsx")

write_to_excel(metrics)
