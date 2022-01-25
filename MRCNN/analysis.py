import numpy as np 
import json
def compute_iou(cand_box, gt_box):
    # Calculate intersection areas
    x1 = np.maximum(cand_box[0], gt_box[0])
    y1 = np.maximum(cand_box[1], gt_box[1])
    x2 = np.minimum(cand_box[2], gt_box[2])
    y2 = np.minimum(cand_box[3], gt_box[3])
    
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    
    cand_box_area = (cand_box[2] - cand_box[0]) * (cand_box[3] - cand_box[1])
    gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = cand_box_area + gt_box_area - intersection
    
    iou = intersection / union
    return iou

metrics = json.load(open("detailed_metrics.json", "r", encoding="utf-8"))

from sklearn.metrics import average_precision_score
def analysis(metrics):
    ious, aps = {}, {}
    logs = {}
    for image_name, result in metrics.items():
        if image_name not in logs:
            logs[image_name] = {}
            
        for class_name, stats in result.items():
            if class_name not in logs[image_name]:
                logs[image_name][class_name] = {}
                logs[image_name][class_name]["gt_label"] = []
                logs[image_name][class_name]["label"] = []
                logs[image_name][class_name]["gt_bbox"] = []
                logs[image_name][class_name]["bbox"] = []
                logs[image_name][class_name]["conf"] = []
                logs[image_name][class_name]["iou"] = []
                logs[image_name][class_name]["correct"] = []
                logs[image_name][class_name]["ap"] = []
                
            gt_bbox = stats['gt_bbox']
            gt_label = stats['gt_label']
            pred_bbox = stats['bbox']
            pred_label = stats['label']
            conf = stats['conf']
            for i in range(len(gt_bbox)):
                for j in range(len(pred_bbox)):
                    iou = compute_iou(gt_bbox[i], pred_bbox[j])
                    if iou > 0.5:
                        logs[image_name][class_name]["correct"].append(pred_label[j] == gt_label[i])
                        logs[image_name][class_name]["gt_label"].append(gt_label[i])
                        logs[image_name][class_name]["gt_bbox"].append(gt_bbox[i])
                        logs[image_name][class_name]["label"].append(pred_label[j])
                        logs[image_name][class_name]['bbox'].append(pred_bbox[j])
                        logs[image_name][class_name]['conf'].append(conf[j])
                        logs[image_name][class_name]["iou"].append(iou)
                        
                        if class_name not in ious:
                            ious[class_name] = [iou]
                        else:
                            ious[class_name].append(iou)
                            
            n = len(logs[image_name][class_name]["correct"])
            if n != 0:
                AP = average_precision_score(logs[image_name][class_name]["correct"], logs[image_name][class_name]['conf'])

                if not np.isnan(AP):
                    for _ in range(n):
                        logs[image_name][class_name]['ap'].append(AP)
                        
                    if class_name not in aps:
                        aps[class_name] = [AP]

                    else:
                        aps[class_name].append(AP)
    for class_name, iou in ious.items():
        print(f"mIoU for {class_name}: {np.mean(iou)}")

    for class_name, ap in aps.items():
        print(f"AP for {class_name}: {np.mean(ap)}")

    return logs

def write_to_excel(metrics):
    from openpyxl import Workbook
    miou = 0
    counts = 0 

    wb = Workbook()
    ws = wb.active
    ws.append(["image_name", "correct", "gt_label", "gt_bbox", "label","bbox", "conf", "iou", "ap"])
    for image_name, result in analysis(metrics).items():
        for class_name, stats in result.items():
            for i in range(len(stats['label'])):
                try:
                    ws.append([image_name, str(stats["correct"][i]), str(stats["gt_label"][i]), str(stats['gt_bbox'][i]), str(stats['label'][i]), str(stats['bbox'][i]), str(stats['conf'][i]), str(stats['iou'][i]), str(stats['ap'][i])])
                except:
                    continue
            average_iou = np.mean(stats["iou"])
            if not np.isnan(average_iou):
                miou += average_iou
                counts += 1


    wb.save("test.xlsx")
    print("mean iou", miou/counts)

write_to_excel(metrics)
