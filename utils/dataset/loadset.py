import os
import json
import re
import cv2 as cv

DEFAULT_SHAPE = [600,1088]

def load_videos(dataset):
    ret = []
    if dataset["dataset_type"] == "labelme":
        load = load_labelme_data
    if dataset["dataset_type"] == "coco":
        load = load_coco_data
    if dataset["dataset_type"] == "yawei":
        load = load_yawei_data
    if dataset["dataset_type"] == "yolo":
        load = load_yolo_data
    root = dataset["path"]
    basenames = os.listdir(root)
    for basename in basenames:
        subdir_path = os.path.join(root,basename)
        if os.path.isdir(subdir_path):
            ret.append(load(subdir_path))
    return ret

def load_pics_from_videos(dataset):
    ret = []
    if dataset["dataset_type"] == "labelme":
        load = load_labelme_data
    if dataset["dataset_type"] == "coco":
        load = load_coco_data
    if dataset["dataset_type"] == "yawei":
        load = load_yawei_data
    if dataset["dataset_type"] == "yolo":
        load = load_yolo_data
    root = dataset["path"]
    basenames = os.listdir(root)
    for basename in basenames:
        subdir_path = os.path.join(root,basename)
        if os.path.isdir(subdir_path):
            ret += load(subdir_path)
    return ret

def load_pics(dataset):
    ret = []
    if dataset["dataset_type"] == "labelme":
        load = load_labelme_data
    if dataset["dataset_type"] == "coco":
        load = load_coco_data
    if dataset["dataset_type"] == "yawei":
        load = load_yawei_data
    if dataset["dataset_type"] == "yolo":
        load = load_yolo_data
    root = dataset["path"]
    ret += load(root)
    return ret


def load_labelme_data(root):
    ret = []
    basenames = os.listdir(root)
    for basename in basenames:
        file_path = os.path.join(root,basename)
        if os.path.isfile(file_path):
            if file_path.endswith("json"):
                label = None
                with open(file_path,"r") as label_file:
                    label = json.load(label_file)
                new_labels = {"image_path":"","items":[]}
                if "shapes" in label:
                    for item in label["shapes"]:
                        new_label = {}
                        if item["label"] == "external corner joint" or item["label"] == 4 or item["label"] == "4":
                            new_label['seam_type'] = 4
                        elif item["label"] == "lap joint" or item["label"] == 3 or item["label"] == "3":
                            new_label['seam_type'] = 3
                        elif item["label"] == "internal corner joint" or item["label"] == 2 or item["label"] == "2":
                            new_label['seam_type'] = 2               
                        elif item["label"] == "bult joint" or item["label"] == 1 or item["label"] == "1":
                            new_label['seam_type'] = 1
                        elif item["label"] == "welded" or item["label"] == 0 or item["label"] == "0":
                            new_label['seam_type'] = 0
                        else:
                            raise("type error:{}".format(item["label"]))
                            # new_label['seam_type'] = -1
                        new_label['p'] = []
                        for point in item['points']:
                            new_label['p'].append(point)
                        # assert new_label['p'][0][2] < new_label['p'][1][2]
                        # new_label['px'] = item['points'][0][0]
                        # new_label['py'] = item['points'][0][1]
                        new_labels["items"].append(new_label)
                    new_labels["image_path"] = os.path.join(root,label["imagePath"])
                    if 'external' in label.keys():
                        new_labels['external'] = {"image_path":os.path.join(root,label["external"]["imagePath"])}
                    new_labels["image_height"] = label["imageHeight"]
                    new_labels["image_width"] = label["imageWidth"]
                    ret.append(new_labels)
    return ret

def load_labelme_data_mult(root):
    ret = []
    basenames = os.listdir(root)
    for basename in basenames:
        file_path = os.path.join(root,basename)
        if os.path.isfile(file_path):
            if file_path.endswith("json"):
                label = None
                with open(file_path,"r") as label_file:
                    label = json.load(label_file)
                new_labels = {"image_path":"","items":[]}
                if "shapes" in label:
                    for item in label["shapes"]:
                        new_label = {}
                        if item["label"] == "external corner joint" or item["label"] == 3:
                            new_label['seam_type'] = 3
                        elif item["label"] == "lap joint" or item["label"] == 2:
                            new_label['seam_type'] = 2
                        elif item["label"] == "internal corner joint" or item["label"] == 1:
                            new_label['seam_type'] = 1                
                        elif item["label"] == "bult joint" or item["label"] == 0:
                            new_label['seam_type'] = 0
                        else:
                            new_label['seam_type'] = -1
                        new_label['points'] = [point for point in item['points']]
                        new_label['group_id'] = item['group_id']
                        # new_label['px'] = item['points'][0][0]
                        # new_label['py'] = item['points'][0][1]
                        new_labels["items"].append(new_label)
                    new_labels["image_path"] = os.path.join(root,label["imagePath"])
                    new_labels["image_height"] = label["imageHeight"]
                    new_labels["image_width"] = label["imageWidth"]
                    ret.append(new_labels)
    return ret    

def load_yawei_data(root):
    ret = []
    basenames = os.listdir(root)
    for basename in basenames:
        file_path = os.path.join(root,basename)
        if os.path.isfile(file_path):
            if file_path.endswith("json"):
                label = None
                with open(file_path,"r") as label_file:
                    label = json.load(label_file)
                new_labels = {"image_path":"","items":[]}
                new_label = {}
                if label["seamtype"] == "external corner joint" or label["seamtype"] == 3:
                    new_label['seam_type'] = 3
                elif label["seamtype"] == "lap joint" or label["seamtype"] == 2:
                    new_label['seam_type'] = 2
                elif label["seamtype"] == "internal corner joint" or label["seamtype"] == 1:
                    new_label['seam_type'] = 1                
                elif label["seamtype"] == "bult joint" or label["seamtype"] == 0:
                    new_label['seam_type'] = 0
                else:
                    new_label['seam_type'] = -1
                new_label['px'] = label['px']
                new_label['py'] = label['py']
                new_labels["items"].append(new_label)
                new_labels["image_path"] = os.path.join(root,label["image_name"])
                new_labels["image_height"] = 600
                new_labels["image_width"] = 1088
                ret.append(new_labels)
    return ret

def load_coco_data(root):
    pass

def load_yolo_data(root):
    ret = []
    ends = [".jpg",".bmp"]
    basenames = os.listdir(root)
    for basename in basenames:
        file_path = os.path.join(root,basename)
        if os.path.isfile(file_path):
            if file_path.endswith("txt"):
                image_path = ''
                for end in ends:
                    image_path = os.path.join(root, os.path.splitext(basename)[0]+end)
                    if(os.path.exists(image_path)):
                        break
                    else:
                        image_path = ''
                if image_path == '':
                    continue
                    # raise Exception("label:{},".format(basename))
                new_labels = {"image_path":"","items":[]}
                new_labels["image_path"] = image_path
                height,width,channel = cv.imread(image_path).shape
                new_labels["image_height"] = height
                new_labels["image_width"] = width
                with open(file_path,'r') as yolo_label_file:
                    lines = yolo_label_file.readlines()
                    for line in lines:
                        pattern = "(\d+)\s+([\d.]+)\s+([\d.]+)"
                        match = re.match(pattern,line)
                        new_label = {}
                        new_label['seam_type'] = int(match[1])
                        
                        new_label['px'] = float(match[2])*width
                        new_label['py'] = float(match[3])*height
                        new_labels["items"].append(new_label)

                ret.append(new_labels)
    return ret
    
