import json
import os

metrics =[f for f in os.listdir() if "metrics" in f and f.endswith(".json")]
metrics = sorted(metrics, key=lambda f: int(f.split("metrics_")[-1].strip(".json")))
metric = metrics[-1]

metric = json.load(open(metric ,"r"))

for k, v in metric.items():
    temp=v.values()
    print (k , sum(temp)/len(temp))

