import numpy as np 
import json
from tqdm import tqdm
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
        #print(result)
        if isinstance(result, int) or isinstance(result, str):
            continue
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
                
            gt_bbox = stats['gt_bbox']
            gt_label = stats['gt_label']
            pred_bbox = stats['bbox']
            pred_label = stats['label']
            conf = stats['conf']
            for i in range(len(gt_bbox)):
                best_iou = -99
                best_idx = None
                for j in range(len(pred_bbox)):
                    iou = compute_iou(gt_bbox[i], pred_bbox[j])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j

                logs[image_name][class_name]["gt_label"].append(gt_label[i])
                logs[image_name][class_name]["gt_bbox"].append(gt_bbox[i])
                logs[image_name][class_name]["label"].append(pred_label[best_idx])
                logs[image_name][class_name]['bbox'].append(pred_bbox[best_idx])
                logs[image_name][class_name]['conf'].append(conf[best_idx])
                logs[image_name][class_name]["iou"].append(best_iou)
                if best_iou > 0.5:
                    logs[image_name][class_name]["correct"].append(pred_label[best_idx] == gt_label[i])

                else:
                    logs[image_name][class_name]["correct"].append(False)
                                                

    return logs

def write_to_excel(metrics):
    from openpyxl import Workbook
    is_correct_by_class = {}
    conf_by_class = {}
    iou_by_class = {}

    wb = Workbook()
    ws = wb.active
    ws.append(['start', 'end'])
    ws.append([metrics['start'], metrics['end']])
    
    labels = json.load(open("labels.json", "r"))
    analyzed_metrics = analysis(metrics)
    
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
        gt_bboxs = []
        pred_labels = []
        pred_bboxs = []
        pred_ious = []

        ws_label = wb.create_sheet(label)
        for image_name, result in tqdm(analyzed_metrics.items(), desc="image analysis"):
            if isinstance(result, int) or isinstance(result, str):
                continue

            for class_name, stats in result.items():
                if class_name == label: 
                    counts_per_class[label] += len(stats['label'])
                    for i in range(len(stats['label'])):
                        
                        correct = stats["correct"][i]
                        gt_label = stats["gt_label"][i]
                        gt_bbox = str(stats["gt_bbox"][i])
                        
                        pred_label = stats["label"][i]
                        pred_bbox = str(stats["bbox"][i])
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
                        gt_bboxs.append(gt_bbox)
                        pred_labels.append(pred_label)
                        pred_bboxs.append(pred_bbox)
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


        ws_label.append(["image_name", "correct", "gt_label", "gt_bbox", "label","bbox", "iou", "cumul_TP", "cumul_FP", "cumul_Prec", "cumul_Rec", "cumul_area"])

        for image_name, correct, gt_label, gt_bbox, pred_label, pred_bbox, pred_iou, cumul_TP, cumul_FP, cumul_Prec, cumul_Rec, cumul_area in zip(image_names, corrects, gt_labels, gt_bboxs, pred_labels, pred_bboxs, pred_ious, cumul_TPs, cumul_FPs, cum_Prec, cum_Rec, cum_area):
            ws_label.append([image_name, correct, gt_label, gt_bbox, pred_label, pred_bbox, pred_iou, cumul_TP, cumul_FP, cumul_Prec, cumul_Rec, cumul_area])
        
        
        class_mAP, class_mIoU = cum_area[-1], sum(pred_ious)/len(pred_ious)
        ws_label.append(["mAP", class_mAP, "mIoU", class_mIoU])
        mAPs[label] =  class_mAP
        mIoUs[label] =  class_mIoU

    mAP = sum(list(mAPs.values())) / len(labels)
    mIoU = sum(list(mIoUs.values())) / len(labels)

        



    #for image_name, result in tqdm(analysis(metrics).items(), desc="image analysis"):
        #print("result", result)
    #    if isinstance(result, int) or isinstance(result, str):
    #        continue

    #    for class_name, stats in result.items():
    #        if class_name not in is_correct_by_class:
    #            is_correct_by_class[class_name] = []
            
    #        if class_name not in conf_by_class:
    #            conf_by_class[class_name] = []
            
    #        if class_name not in iou_by_class:
    #            iou_by_class[class_name] = []

    #        for i in range(len(stats['label'])):
    #            try:
    #                is_correct_by_class[class_name].append(stats['correct'][i])
    #                conf_by_class[class_name].append(stats['conf'][i])
    #                iou_by_class[class_name].append(stats["iou"][i])

    #                ws.append([image_name, stats["correct"][i], str(stats["gt_label"][i]), str(stats['gt_bbox'][i]), str(stats['label'][i]), str(stats['bbox'][i]), str(stats['conf'][i]), stats['iou'][i]])

    #            except:
    #                continue
    
    
    #mAP = 0
    #mIoU = 0
    #count = 0
    #for class_name in is_correct_by_class:
    #    is_cor = is_correct_by_class[class_name]
    #    conf = conf_by_class[class_name]
    #    ious = iou_by_class[class_name]

    #    class_average_iou = np.mean(ious)
    #    class_average_ap = average_precision_score(is_cor, conf)
    ws.append(["class", "mIoU", "AP"])
    for label in mAPs:
        ws.append([label, mIoUs[label], mAPs[label]])
        print(f"Class: {label}, Average IoU: {mIoUs[label]}, Average Precision: {mAPs[label]}")
    ws.append(["final_mIoU", "final_mAP"])
    ws.append([mIoU, mAP])
    print(f"Final mAP :{mAP}, Final mIoU : {mIoU}")

    wb.save("test.xlsx")

write_to_excel(metrics)
