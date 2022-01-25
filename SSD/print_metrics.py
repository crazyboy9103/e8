import json
import os
import argparse

def print_metrics(metric):
    metric = json.load(open(metric ,"r"))
    for k, v in metric.items():
        temp=v.values()
        print (k , sum(temp)/len(temp))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="all metrics for each label")
    parser.add_argument("--metric", type=str, help="metric file name")
    args =  parser.parse_args()
    assert args.metric, "metric is not given, automatically finding metric"
    metrics =[f for f in os.listdir() if "metrics" in f and f.endswith(".json")]
    metrics = sorted(metrics, key=lambda f: int(f.split("metrics_")[-1].strip(".json")))
    metric = metrics[-1]
    args.metric = metric

    print_metrics(args.metric)