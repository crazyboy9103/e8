import os 
import json
from tqdm import tqdm
dic = json.load(open("dic.json","r"))
label_map = json.load(open("labels.json", "r"))


def test(data_dir):
    therm_counts = {label:0 for label in label_map}
    for root, dirs, files in tqdm(os.walk(data_dir), desc="Searching for files"):
        for file in files:
            if "_T_" in file:
                _, _, cat, cls, _, _  = file.split("_")
                label = f"{dic[cat]}_{dic[cls]}"
                therm_counts[label] += 1
    return therm_counts

if __name__ =="__main__":
    print(test("/data/dataset/recsys/e8/data_1230"))
        
    