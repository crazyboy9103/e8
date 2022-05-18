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

<<<<<<< HEAD
def write_to_excel(metrics):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    
    ws.append(['start', 'end'])
    ws.append([metrics['start'], metrics['end']])

    labels = json.load(open("labels.json", "r"))

    counts_per_class = {}
    
    mAPs = {}
    mIoUs = {}

    for label in labels:
        counts_per_class[label] = 0
        cum_TP = []
        cum_FP = []
        cum_Prec = []
        cum_Rec = []
        cum_area = []
        image_names = []
        corrects = []
        gt_labels = []
        pred_labels = []
        pred_ious = []
        
        ws_label = wb.create_sheet(label)
    #ws.append(["image_name", "correct", "gt_label", "label", "conf", "iou"])
        for image_name, result in tqdm(metrics.items(), desc="image analysis"):
            if isinstance(result, int) or isinstance(result, str):
                continue

            for class_name, stats in result.items():
                if class_name == label: 
                    counts_per_class[label] += len(stats['label'])
                    for i in range(len(stats['label'])):
                        correct = stats["correct"][i]
                        gt_label = stats["gt_label"][i]
                        
                        pred_label = stats["label"][i]
                        iou = stats['iou'][i]

                        if correct:
                            cum_TP.append(1)
                            cum_FP.append(0)
                        else:
                            cum_TP.append(0)
                            cum_FP.append(1)
        
                         
                        image_names.append(image_name)
                        corrects.append(correct)
                        gt_labels.append(gt_label)
                        pred_labels.append(pred_label)
                        pred_ious.append(iou)
        cumul_TP = 0
        cumul_FP = 0
        cumul_TPs = []
        cumul_FPs = []
        for i, (TP, FP) in enumerate(zip(cum_TP, cum_FP)):
            cumul_TP += TP
            cumul_FP += FP

            prec = cumul_TP/(cumul_TP+cumul_FP)
            rec = cumul_TP/counts_per_class[label]
            
            cumul_TPs.append(cumul_TP)
            cumul_FPs.append(cumul_FP)

            cum_Prec.append(prec)
            cum_Rec.append(rec)
            if i == 0:
                cum_area.append(0)
            else:
                area = cum_Prec[i] * (cum_Rec[i]-cum_Rec[i-1])
                cum_area.append(area + cum_area[-1])
        
        ws_label.append(["image_name", "correct", "gt_label", "label", "iou", "cumul_TP", "cumul_FP", "cumul_Prec", "cumul_Rec", "cumul_area"])

        for image_name, correct, gt_label, pred_label, pred_iou, cumul_TP, cumul_FP, cumul_Prec, cumul_Rec, cumul_area in zip(image_names, corrects, gt_labels, pred_labels, pred_ious, cumul_TPs, cumul_FPs, cum_Prec, cum_Rec, cum_area):
            ws_label.append([image_name, correct, gt_label, pred_label, pred_iou, cumul_TP, cumul_FP, cumul_Prec, cumul_Rec, cumul_area])
        
        class_mAP, class_mIoU = cum_area[-1], sum(pred_ious)/len(pred_ious)
        ws_label.append(["mAP", class_mAP, "mIoU", class_mIoU])
        mAPs[label] =  class_mAP
        mIoUs[label] =  class_mIoU

    mAP = sum(list(mAPs.values())) / len(labels)
    mIoU = sum(list(mIoUs.values())) / len(labels)
    ws.append(["class", "mIoU", "AP"])
    for label in mAPs:
        ws.append([label, mIoUs[label], mAPs[label]])
        print(f"Class: {label}, Average IoU: {mIoUs[label]}, Average Precision: {mAPs[label]}")
    ws.append(["final_mIoU", "final_mAP"])
    ws.append([mIoU, mAP])
    print(f"Final mAP :{mAP}, Final mIoU : {mIoU}")
        
    wb.save("test.xlsx")
=======
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

>>>>>>> b0fb9406a7590190cd75b01d59fabfec9968d853

wb.save("mrcnn_test.xlsx")
wb.close()