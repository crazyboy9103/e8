import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from openpyxl import Workbook

f = open("detailed_metrics.json", "r")
metrics = json.load(f)
f.close()
from openpyxl import Workbook
is_correct_by_class = {}
conf_by_class = {}
iou_by_class = {}
analysis_result = []

analysis_result.append(["image_name", "correct", "gt_label", "label", "conf", "iou", "", "class_name", "Average IoU", "Average Precision", "mAP", "mIoU"])
for image_name, result in tqdm(list(metrics.items())[1:-1], desc=f"image analysis"):
    if isinstance(result, int) or isinstance(result, str):
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
                # ws.append([image_name, str(stats["correct"][i]), str(stats["gt_label"][i]), str(stats['label'][i]), str(stats['conf'][i]), str(stats['iou'][i])])
                analysis_result.append([image_name, str(stats["correct"][i]), str(stats["gt_label"][i]), str(stats['label'][i]), str(stats['conf'][i]), str(stats['iou'][i])])
            except Exception as ex:
                continue
mAP = 0
mIoU = 0
count = 0
cnt = 1
for class_name in is_correct_by_class:
   is_cor = is_correct_by_class[class_name]
   conf = conf_by_class[class_name]
   ious = iou_by_class[class_name]
   class_average_iou = np.mean(ious)
   class_average_ap = average_precision_score(is_cor, conf)
   # ws.append([class_name, class_average_iou, class_average_ap])
   analysis_result[cnt].extend(["", class_average_iou, class_average_ap])
   print(f"Class: {class_name}, Average IoU: {class_average_iou}, Average Precision: {class_average_ap}")
   cnt += 1
   mIoU += class_average_iou
   mAP += class_average_ap
   count += 1
analysis_result[1].extend(["", mAP/count, mIoU/count])
# ws.append(["Final mAP", mAP/count, "Final maskIou", mIoU/count])
print(f"Final mAP :{mAP/count}, Final mIoU : {mIoU/count}")

wb = Workbook()
ws = wb.active
for row in analysis_result:
    ws.append(row)
wb.save("mrcnn_test.xlsx")
