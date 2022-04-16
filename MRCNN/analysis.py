import numpy as np 
import json
from tqdm import tqdm


print("Reading detailed_metrics.json...")
f = open("detailed_metrics.json", "r")
metrics = json.load(f)
f.close()
print("Finished reading")

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

write_to_excel(metrics)
