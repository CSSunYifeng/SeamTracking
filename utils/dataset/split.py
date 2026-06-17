
import os
import random
import json
import shutil
import time
import sys
ROOT = os.getenv('SEAMTRACKING_ROOT')
sys.path.append(ROOT)
# if __name__=="__main__":
#     from loadset import load_pics_from_videos,load_pics
# else:
from utils.dataset.loadset import load_pics_from_videos,load_pics


# TRAIN_RATIO = 0.01
# VAL_RATIO = 0.01

def delet_files(root):
    dirs = os.listdir(root)
    for dir in dirs:
        item_path = os.path.join(root,dir)
        if os.path.isfile(item_path):
            os.remove(item_path)
        else:
            delet_files(item_path)

def sort_p(opt,poses):
    poses = sorted(poses,key=lambda x:x[0])
    return poses

def split_pics(opt, data_loss = False):
    idx = 0
    dataset_path = os.path.join(ROOT,"data",opt["exp_name"])
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
        data_loss = True

    train_path = ""
    val_path = ""
    test_path = ""

    train_path = os.path.join(dataset_path,"train")
    val_path = os.path.join(dataset_path,"val")
    test_path = os.path.join(dataset_path,"test")

    if not os.path.exists(train_path):
        os.mkdir(os.path.join(train_path))
        data_loss = True
    if not os.path.exists(val_path):
        os.mkdir(val_path)
        data_loss = True
    if not os.path.exists(test_path):
        os.mkdir(test_path)
        data_loss = True
    if data_loss == False or opt["datasets"] == None:
        return
    if data_loss == True and opt["datasets"] == None:
        raise Exception("loss local dataset(loss $ROOT/data/$EXP/train or $ROOT/data/$EXP/val or $ROOT/data/$EXP/test ), please set opt['dataset'] to load data")



    if os.path.exists(train_path):
        delet_files(train_path)
    else:
        os.mkdir(train_path)
    if os.path.exists(val_path):
        delet_files(val_path)
    else:
        os.mkdir(val_path)
    if os.path.exists(test_path):
        delet_files(test_path)
    else:
        os.mkdir(test_path)

    datas = []
    for dataset_info in opt["datasets"]:
        if dataset_info['type'] == "videos":
            datas += load_pics_from_videos(dataset_info)
        elif dataset_info['type'] == "pics":
            datas += load_pics(dataset_info)
    random.shuffle(datas) # 洗牌
    train_len = int(len(datas)*opt['train_ratio'])
    val_len = int(len(datas)*opt['val_ratio'])
    test_len = int(len(datas)*(1-opt['train_ratio']-opt['val_ratio']))
    print("train num:{}, val num:{}, test num:{}".format(train_len,val_len,test_len))
    path = train_path
    for i in range(train_len):
        # datas[i]
        image_name = str(idx) + os.path.splitext(datas[i]['image_path'])[1] # os.path.split(datas[i]['image_path'])[1]
        img_path = os.path.join(path,image_name)
        shutil.copy(datas[i]['image_path'],img_path)
        filename = os.path.splitext(img_path)[0]
        json_dict = generate_labme_label(datas[i])
        json_dict["imagePath"] = image_name
        with open(filename+".json",'w') as json_file:
            json.dump(json_dict,json_file)
        idx += 1
    path = val_path
    head = train_len
    tail = train_len+val_len
    for i in range(head,tail):
        # datas[i]
        image_name = str(idx) + os.path.splitext(datas[i]['image_path'])[1]
        img_path = os.path.join(path,image_name)
        shutil.copy(datas[i]['image_path'],img_path)
        filename =  os.path.splitext(img_path)[0]
        json_dict = generate_labme_label(datas[i])
        json_dict["imagePath"] = image_name
        with open(filename+".json",'w') as json_file:
            json.dump(json_dict,json_file)
        idx += 1
    path = test_path
    head = train_len+val_len
    tail = train_len+val_len+test_len
    for i in range(head,tail):
        # datas[i]
        image_name = str(idx) + os.path.splitext(datas[i]['image_path'])[1]
        img_path = os.path.join(path,image_name)
        shutil.copy(datas[i]['image_path'],img_path)
        filename =  os.path.splitext(img_path)[0]
        json_dict = generate_labme_label(datas[i])
        json_dict["imagePath"] = image_name
        with open(filename+".json",'w') as json_file:
            json.dump(json_dict,json_file)
        idx += 1

def generate_labme_label(data):
    shapes = []
    for item in data['items']:
        shapes.append({
            "label":item['seam_type'],
            "points": [point for point in item['p']],# [[item['px'],item['py']]],
            "group_id": None,
            "shape_type":"point",
            "flags":{}
        })
    ret = {"version":"4.5.13",
           "flags":{},
           "shapes":shapes,
           "imagePath":os.path.split(data['image_path'])[1],
           "imageHeight":data['image_height'],
           "imageWidth":data['image_width']}
    return ret

def generate_coco_label(dataset):
    ret = {
            "licenses":[{
                "id":1,
                "url":"https://creativecommons.org/licenses/by/4.0/",
                "name":"CC BY 4.0"
                
            }],
            "categories":[
            {
                "id":0,
                "name":"bult joint",
                "supercategory": "None"
            },
            {
                "id":1,
                "name":"internal corner joint",
                "supercategory": "None"
            },
            {
                "id":2,
                "name":"lap joint",
                "supercategory": "None"
            },
            {
                "id":3,
                "name":"external corner joint",
                "supercategory": "None"
            }
            ],
            "images":[
                
            ],
            "annotations":[
                
            ]
        }
    images_list = ret["images"]
    annotations_list = ret["annotations"]
    image_idx = 0
    annotation_idx = 0
    for data in dataset:
        data_json = {
            "id":image_idx,
            "license":1,
            "file_name":os.path.split(data['image_path'])[1],
            "height":data['image_height'],
            "width":data['image_width'],
            "date_captured":str(time.time())
        }
        image_idx += 1
        images_list.append(data_json)
        for item in data['items']:
            annotation_json = {
                "id":annotation_idx,
                "image_id":image_idx,
                "category_id":item['seam_type'],
                'bbox':[
                    item['px']-25,
                    item['py']-25,
                    50,
                    50
                ],
                'area':2500,
                'segmentation':[],
            }
            annotations_list.append(annotation_json)
        annotation_idx+=1
    return ret
        
def trans_labelme_2_coco_label(dataset_path):
    dataset=[]
    for root,dir,files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.json') and file[0]!='_':
                json_path = os.path.join(root,file)
                with open(json_path,'r') as json_reader:
                    value = json.load(json_reader)
                dataset.append(value)
    ret = {
            "licenses":[{
                "id":1,
                "url":"https://creativecommons.org/licenses/by/4.0/",
                "name":"CC BY 4.0"
                
            }],
            "categories":[
            {
                "id":0,
                "name":"bult joint",
                "supercategory": "None"
            },
            {
                "id":1,
                "name":"internal corner joint",
                "supercategory": "None"
            },
            {
                "id":2,
                "name":"lap joint",
                "supercategory": "None"
            },
            {
                "id":3,
                "name":"external corner joint",
                "supercategory": "None"
            }
            ],
            "images":[
                
            ],
            "annotations":[
                
            ]
        }
    images_list = ret["images"]
    annotations_list = ret["annotations"]
    image_idx = 0
    annotation_idx = 0
    for data in dataset:
        data_json = {
            "id":image_idx,
            "license":1,
            "file_name":os.path.split(data['imagePath'])[1],
            "height":data['imageHeight'],
            "width":data['imageWidth'],
            "date_captured":str(time.time())
        }
        images_list.append(data_json)
        for item in data['shapes']:
            annotation_json = {
                "id":annotation_idx,
                "image_id":image_idx,
                "category_id":item['label'],
                'bbox':[
                    item['points'][0][0]-10,
                    item['points'][0][1]-10,
                    20,
                    20
                ],
                'area':2500,
                'segmentation':[],
            }
            annotations_list.append(annotation_json)
        annotation_idx+=1
        image_idx += 1
    # return ret
    with open(os.path.join(src,'_annotations.coco.json'),'w') as json_writer:
        json.dump(ret,json_writer)

# def trans_labelme_2_yolo_label(dataset_path,root):
#     subdirs = ['test','train','val']
#     for dir in subdirs:
#         dst_path = os.path.join(root,dir)
#         if not os.path.exists(dst_path):
#             os.makedirs(dst_path)
#         for root,dirs,files in os.walk(dataset_path):
#             for file in files:
#                 if file.endwith('.json'):
#                     file_path = os.path.join(dataset_path,file)

#         shutil.copy

def print_cls_src(opt,cls):
    datas = []
    for dataset_info in opt["datasets"]:
        if dataset_info['type'] == "videos":
            datas += load_pics_from_videos(dataset_info)
        elif dataset_info['type'] == "pics":
            datas += load_pics(dataset_info)
    result = {cls:[]}
    for item in datas:
        if item['items'][0]['seam_type'] == cls:
            result[cls].append(item["image_path"])
    with open("cls_src_{}.json".format(cls),'w') as json_file:
        json.dump(result,json_file)

def split_pics_without_json_by_cat(opt,cats,cunt):
    dataset_path = os.path.join(ROOT,"data",opt["exp_name"])
    data_loss = False
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
        data_loss = True
    ret = {}
    for cls in cats:
        ret[cls] = []
        _path = os.path.join(dataset_path,cls)

        if not os.path.exists(_path):
            os.mkdir(os.path.join(_path))
            data_loss = True
        if data_loss == False or opt["datasets"] == None:
            return
        if data_loss == True and opt["datasets"] == None:
            raise Exception("loss local dataset(loss $ROOT/data/$EXP/train or $ROOT/data/$EXP/val or $ROOT/data/$EXP/test ), please set opt['dataset'] to load data")
    datas = []
    for dataset_info in opt["datasets"]:
        if dataset_info['type'] == "videos":
            datas += load_pics_from_videos(dataset_info)
        elif dataset_info['type'] == "pics":
            datas += load_pics(dataset_info)

    random.shuffle(datas) 
    for data in datas:
        for ci,cls in enumerate(cats):
            if (str(data['items'][0]['seam_type']) == cls or str(data['items'][0]['seam_type']) == str(ci)) \
                and len(ret[cls])<cunt/len(cats):
                ret[cls].append(data)
                break
        break_flag = True
        for data_cat in ret:
            print("{}:{}".format(data_cat,len(ret[data_cat])))
            if len(ret[data_cat]) < cunt/len(cats):
                break_flag = False
        if break_flag:
            break
    img_idx = 0
    for data_cat in ret:
        for value in ret[data_cat]:
            file_name = str(img_idx) + os.path.splitext(os.path.split(value['image_path'])[1])[1]
            shutil.copy(value['image_path'],os.path.join(dataset_path,data_cat,file_name))
            img_idx += 1

