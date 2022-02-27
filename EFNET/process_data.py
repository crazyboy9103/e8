import torch 
from tqdm import tqdm
import os 
import json 
import numpy as np
import cv2
label_map = {"1": "best", "2":"normal", "3":"faulty"}
dataset_path = "./dataset/"
categories = {"crack":"C", "finish": "T" , "ground":"F", "living":"L", "peel":"P", "rebar":"X", "window":"W"}

W, H = 1080, 1440
W_new, H_new = 224, 224

def process_dataset(data_dir, category):
    paths_filename = {}

    for root, dirs, files in tqdm(os.walk(data_dir), desc="Searching for files"):
        for file in files:
            #print(file)
            temp = file.split("_")
            cat, _, img_type = temp[2], temp[3], temp[4]
            if cat == categories[category] and img_type == "R":
                paths_filename[file] = os.path.join(root, file)
    
    labels_names = [f for f in paths_filename.keys() if f.endswith(".json")]
    images_names = [f for f in paths_filename.keys() if f.endswith(".jpg")]
    
    labels_ids = [label.split("/")[-1].strip(".json") for label in labels_names]
    images_ids = [image.split("/")[-1].strip(".jpg") for image in images_names]
    labels_ids_idxs = {label_id: i for i, label_id in enumerate(labels_ids)}
    images_ids_idxs = {image_id: i for i, image_id in enumerate(images_ids)}
    
    labels_ids_set = set(labels_ids)
    images_ids_set = set(images_ids)
    

    pairs = sorted(list(labels_ids_set & images_ids_set)) # matching pairs         
    
    del labels_ids_set
    del images_ids_set
    labels_idxs = [labels_ids_idxs[pair] for pair in tqdm(pairs, desc="finding pairs for labels")]
    images_idxs = [images_ids_idxs[pair] for pair in tqdm(pairs, desc="finding pairs for images")]
    del labels_ids
    del images_ids
    del labels_ids_idxs
    del images_ids_idxs
    
    labels_names = [labels_names[idx] for idx in labels_idxs]
    images_names = [images_names[idx] for idx in images_idxs]
    del labels_idxs
    del images_idxs
    
    # to check the pairs 
    for label_path, image_path in tqdm(zip(labels_names, images_names), desc="checking valid pairs"):
        assert (label_path.split("/")[-1].strip(".json") == image_path.split("/")[-1].strip(".jpg"))
    
    print("Done validating dataset")
    
    print("Collecting processed files...")
    labels_paths = [paths_filename[name] for name in tqdm(labels_names)]
    images_paths = [paths_filename[name] for name in tqdm(images_names)]
    print("Done Collection")
    
    for img, label in tqdm(zip(images_paths, labels_paths), desc="Data processing"):
        info = {}
        info['boxes'], info['labels'] = [], []
        
        f = open(os.path.join(data_dir, label), "r")
        json_data = json.load(f)['Learning_Data_Info']
        f.close()

        annotations = json_data['Annotations']
        json_id = json_data['Json_Data_ID']
        _, _, _, cls, _, _  = json_id.split("_")

        # cat must be "W"
        # cls must be "1, 2, 3" : "best, normal, faulty"
        if img_type == "R":
            continue
    
        image = cv2.imread(img, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        filename = img.split("/")[-1].split(".jpg")[0]
        for i, ant in enumerate(annotations):
            ant_type = ant['Type']
            label = label_map[cls]
            if ant_type == "bbox":
                xmin, ymin, xmax, ymax = ant[ant_type]
             
                if xmin == xmax or ymin == ymax:
                    continue
                
                bbox_image = image[ymin:ymax, xmin:xmax]
                bbox_image = cv2.resize(bbox_image, (W_new, H_new), interpolation=cv2.INTER_AREA)
                cv2.imwrite(dataset_path + category +"/"+ label +"/"+ filename + "_" + str(i) + ".jpg", bbox_image)


for hr_category, alphabet in categories.items():
    process_dataset("/dataset/data_1230", hr_category)