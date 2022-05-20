import json
from openpyxl import Workbook

class_names = ["crack", "finish", "ground", "living", "peel", "rebar", "window"]
model_names = ["eff_net_" + _cls for _cls in class_names]
json_names = [model_name + "_eval.json" for model_name in model_names]


wb = Workbook(write_only=True)
ws = wb.create_sheet("EFNET")

for model_name, json_name in zip(model_names, json_names):
    ws.append([model_name])
    with open(json_name, "r") as f:
        model_result = json.load(f)
    
    # Stats
    for label in ["best", "normal", "faulty"]:
        header = []
        values = []
        for c in ["tn", "fp", "fn", "tp", "prec", "recall", "f1", "acc"]:
            key = label + "_" + c
            value = model_result[key]
            
            header.append(key)
            values.append(value)
        
        ws.append(header)
        ws.append(values)
    
    ws.append([""])
    # Average Stats
    header = ["avg_f1", "avg_acc", "eval_start", "eval_end"]
    stats = [model_result[column] for column in header]
    ws.append(header)
    ws.append(stats)

    # Individual results
    header = ["image", "label", "pred", "correct", "time", "cum_total", "cum_correct"]
    ws.append(header)

    keys = ["image_names", "gt_labels", "pred_labels", "corrects", "times", "totals", "cum_corrects"]
    values = [model_result[key] for key in keys]

    for line in zip(*values):
        line = list(map(str, line))
        ws.append(line)
    

wb.save("eff_net_test.xlsx")


 
    