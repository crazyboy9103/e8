import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from openpyxl import Workbook
import time, datetime

# Load logs
f = open("detailed_metrics.json", "r")
logs = json.load(f)
f.close()

# Labels
labels = json.load(open("labels.json", "r"))
labels = list(labels.keys())
# Open workbook
wb = Workbook()
assert "Sheet" in wb.sheetnames
worksheets = {}
header = ["image_name", "time", "correct", "gt_label", "label", "conf", "iou", "cum_TP", "cum_FN", "cum_FP", "recall", "precision", "average_precision", "AP", "avg_iou"]
for label in labels:
    ws = wb.create_sheet(label)
    worksheets[label] = ws
    ws.append(header)


analysis_result = {label: [] for label in labels}

#analysis_result.append(["image_name", "correct", "gt_label", "label", "conf", "iou", "", "class_name", "Average IoU", "Average Precision", "mAP", "mIoU"])
for image_name, result in tqdm(logs.items(), desc=f"image analysis"):
    if isinstance(result, int) or isinstance(result, str):
        continue

    for class_name, stats in result.items():
        for i in range(len(stats['label'])):
            analysis_result[class_name].append([image_name, stats["time"][i], stats["correct"][i], stats["gt_label"][i], stats["label"][i], stats["conf"][i], stats["iou"][i], stats["FP"][i], stats["FN"][i], stats["TP"][i]])

APs = {}
mious = {}   

epsilon = 1e-6
for label, result in tqdm(analysis_result.items(), desc="writing to excel"):
    ws = worksheets[label]
    result = sorted(result, key=lambda x: x[5], reverse=True)
    ious = [item[-4] for item in result]
    mean_iou = sum(ious)/len(ious)
    TP = [item[-1] for item in result]
    FN = [item[-2] for item in result]
    FP = [item[-3] for item in result]

    cum_TP = list(accumulate(TP))
    cum_FN = list(accumulate(FN))
    cum_FP = list(accumulate(FP))

    recalls = [tp/(cum_TP[-1] + cum_FN[-1] + epsilon) for tp in cum_TP]
    precisions = [tp/(cum_TP[-1] + cum_FP[-1] + epsilon) for tp in cum_TP]
    average_precisions = integrate.cumtrapz([1]+precisions, [0]+recalls)

    for i, (res, cum_tp, cum_fn, cum_fp, rec, prec, avg_prec) in enumerate(zip(result, cum_TP, cum_FN, cum_FP, recalls, precisions, average_precisions)):
        line = res[:7] + [cum_tp, cum_fn, cum_fp, rec, prec, avg_prec]
        if i == 0:
            APs[label] = average_precisions[-1]
            mious[label] = mean_iou
            line = line + [average_precisions[-1], mean_iou]
        line = list(map(str, line))
        ws.append(line)

ws = wb["Sheet"]
ws.title = "Stats"
ws.append(["Class", "AP", "average IoU", "mAP", "mIoU"])
for i, label in enumerate(labels):
    AP = APs[label]
    miou = mious[label]
    line = [label, AP, miou]
    if i == 0:
        line = line + [sum(APs.values())/len(APs), sum(mious.values())/len(mious)]
    line = list(map(str, line))
    ws.append(line) 


wb.save("mrcnn_test.xlsx")
wb.close()