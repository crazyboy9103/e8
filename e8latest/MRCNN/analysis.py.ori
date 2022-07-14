import numpy as np 
import json
from tqdm import tqdm


print("Reading detailed_metrics.json...")
f = open("detailed_metrics.json", "r")
metrics = json.load(f)
f.close()
print("Finished reading")

from sklearn.metrics import average_precision_score
def write_to_excel(metrics):
    from openpyxl import Workbook
    is_correct_by_class = {}
    conf_by_class = {}
    iou_by_class = {}

    wb = Workbook()
    ws = wb.active
    ws.append(["image_name", "correct", "gt_label", "label", "conf", "iou"])
    for image_name, result in tqdm(metrics.items(), desc="image analysis"):
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

                    ws.append([image_name, str(stats["correct"][i]), str(stats["gt_label"][i]), str(stats['label'][i]), str(stats['conf'][i]), str(stats['iou'][i])])

                except:
                    continue
    
    
    #mAP = 0
    #mIoU = 0
    #count = 0

    #wb.create_sheet(index=1, title="mAPandmIOU")
    #ws = wb[wb.sheetnames[1]]

    #for class_name in is_correct_by_class:
    #    is_cor = is_correct_by_class[class_name]
    #    conf = conf_by_class[class_name]
    #    ious = iou_by_class[class_name]

    #    class_average_iou = np.mean(ious)
    #    class_average_ap = average_precision_score(is_cor, conf)

    #    ws.append([class_name, class_average_iou, class_average_ap])
    #    print(f"Class: {class_name}, Average IoU: {class_average_iou}, Average Precision: {class_average_ap}")
    #    mIoU += class_average_iou
    #    mAP += class_average_ap
    #    count += 1
    #ws.append(["Final mAP", mAP/count, "Final maskIou", mIoU/count])
    #print(f"Final mAP :{mAP/count}, Final mIoU : {mIoU/count}")
    wb.save("test_mrcnn.xlsx")

write_to_excel(metrics)
