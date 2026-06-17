from torch.utils.data import dataset
if __name__=="__main__":
    from loadset import load_videos,load_pics_from_videos,load_pics
    from random_aug import argHuoHua
    from split import generate_coco_label
    from add_interference import inferencer
else:
    from utils.dataset.loadset import load_videos,load_pics_from_videos,load_pics
    from utils.dataset.random_aug import argHuoHua
    from utils.dataset.add_interference import inferencer
import cv2 as cv

import numpy as np
from numpy import ndarray
from typing import List,Tuple,Optional
import os
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from torch.utils.data.dataloader import DataLoader
import copy
import json
from scipy.io import savemat,loadmat
import torch.nn as nn
import torch

ROOT = os.getenv('SEAMTRACKING_ROOT')

def data_aug_half(im:ndarray,ps,cp,flip=0.5,move_ratio=1)->Tuple[ndarray,float,float]:
    h,w,c = im.shape
    x = cp[0]
    x_im = int(cp[0]*w)
    left_indcs = []
    right_indcs = []

    # 
    left_xys = []
    right_xys = []
    max_p = int(np.max(np.array(ps)[:,1]*h))
    min_p = int(np.min(np.array(ps)[:,1]*h))
    for i,p in enumerate(ps):
        assert 0 <= p[0] <= 1 and 0 <= p[1] <= 1, " [0, 1]"
        if p[0]<= x and p[2]<x:
            left_xys.append(Keypoint(x=int(p[0] * w), y=int(p[1] * h)))
            left_xys.append(Keypoint(x=int(p[2] * w), y=int(p[3] * h)))
            left_indcs.append(i)
        elif p[0]>= x and p[2]>x:
            right_xys.append(Keypoint(x=int(p[0] * w), y=int(p[1] * h)))
            right_xys.append(Keypoint(x=int(p[2] * w), y=int(p[3] * h)))
            right_indcs.append(i)
        else:
            raise("left right augment error")

    left_kps = KeypointsOnImage(left_xys, shape=im.shape)
    right_kps = KeypointsOnImage(right_xys, shape=im.shape)
    left = im[:,:x_im]
    right = im[:,x_im:]

    left_offset = np.random.randint(-int(h/5),int(h/5))

    if left_offset < 0:
        left_offset = np.max([left_offset,-min_p])
    elif left_offset > 0:
        left_offset = np.min([left_offset,h-max_p])
    
    left_aug =  iaa.TranslateY(px=left_offset)
    lefted= left_aug.augment_image(left)
    left_kps_aug = left_aug.augment_keypoints(left_kps)

    right_offset = np.random.randint(-int(h/5),int(h/5))

    if right_offset < 0:
        right_offset = np.max([right_offset,-min_p])
    elif right_offset > 0:
        right_offset = np.min([right_offset,h-max_p])

    right_aug =  iaa.TranslateY(px=right_offset)
    righted = right_aug.augment_image(right) # right_kps
    right_kps_aug = right_aug.augment_keypoints(right_kps) 

    # left_ret_set = [[left_kps_aug.keypoints[i].x/w,
    #                  left_aug.keypoints[i].y/h,
    #                  left_kps_aug.keypoints[i+1].x/w,
    #                  left_kps_aug.keypoints[i+1].y/h] for i in range(0,2*left_l,2)]
    # right_ret_set = [[right_kps_aug.keypoints[i].x/w,
    #                   right_aug.keypoints[i].y/h,
    #                   right_kps_aug.keypoints[i+1].x/w,
    #                   right_kps_aug.keypoints[i+1].y/h] for i in range(0,2*right_l,2)]
    for i,index in enumerate(left_indcs):
        ps[index] = [left_kps_aug.keypoints[i*2].x/w,
                     left_kps_aug.keypoints[i*2].y/h,
                     left_kps_aug.keypoints[i*2+1].x/w,
                     left_kps_aug.keypoints[i*2+1].y/h]
    for i,index in enumerate(right_indcs):
        ps[index] = [right_kps_aug.keypoints[i*2].x/w,
                     right_kps_aug.keypoints[i*2].y/h,
                     right_kps_aug.keypoints[i*2+1].x/w,
                     right_kps_aug.keypoints[i*2+1].y/h]
    image_aug = np.hstack((lefted, righted))
    return image_aug,ps



def data_augmentation(im:ndarray,ps,flip=0.5,move_ratio=1)->Tuple[ndarray,float,float]:
    assert len(im.shape)==3
    assert im.shape[2]==1 or im.shape[2]==3
    # hh = argHuoHua(im.shape)
    # im = hh.argImg(im)
    l = len(ps)
    # assert im.dtype==np.uint8
    height,width,ch = im.shape
    xys = []
    for p in ps:
    #     assert p[0]>=0 and p[0]<=1 and p[1]>=0 and p[1]<=1
    # x = int(x*width)
    # y = int(y*height)
        xys.append(Keypoint(x = int(p[0]*width),y = int(p[1]*height)))
        xys.append(Keypoint(x = int(p[2]*width),y = int(p[3]*height)))
    kps = KeypointsOnImage(xys, shape=im.shape)
    ps = np.transpose(ps,(1,0)) 
    x = min(np.array(ps[0])*width)
    y = min(np.array(ps[1])*height)
    dx = int(min(x,width-x)*0.5)*move_ratio
    dy = int(min(y,height-y)*0.5)*move_ratio
    seq = iaa.Sequential([
        iaa.Affine(scale={"x": (0.1, 1.5), "y": (0.1, 1.5)}, rotate=(-30,30)),
        iaa.Fliplr(flip), # 
        iaa.TranslateX(px=(-dx,dx)),
        iaa.TranslateY(px=(-dy,dy)),
        #iaa.GaussianBlur(sigma=(0, 3.0)) #，
    ])

    # _flip = iaa.Sequential([iaa.Fliplr(0.5)])
    # image_aug, kps_aug = _flip(image=im, keypoints=kps)

    # if kps_aug.keypoints != kps.keypoints:
    #     kps_aug = KeypointsOnImage(kps_aug.keypoints[::-1],shape=im.shape)

    image_aug, kps_aug = seq(image=im, keypoints=kps)
    ret_set = [[kps_aug.keypoints[i].x/width,kps_aug.keypoints[i].y/height,kps_aug.keypoints[i+1].x/width,kps_aug.keypoints[i+1].y/height] for i in range(0,2*l,2)]

    return image_aug,ret_set

def set_balance(dataset,cls_num):
    cat_cunts = np.array([0]*cls_num)
    cats = {}
    for i in range(cls_num):
        cats[i] = []
    for idx,data in enumerate(dataset):
        cat = data['items'][0]['seam_type']
        cat_cunts[cat] += 1
        cats[cat].append(idx)
    max_cat = max(cat_cunts)
    cat_req = -(cat_cunts - max_cat)
    times = cat_req/max_cat
    ans = []
    for i in range(cls_num):
        tt = int(np.floor(times[i]))
        dt = times[i]-tt
        g = cats[i][:int(dt*len(cats[i]))]
        cats[i] *= tt
        cats[i] += g
        ans += cats[i]
    ans = np.array(ans)
    np.random.shuffle(ans)
    for a in ans:
        ext = copy.deepcopy(dataset[a])
        ext['aug'] = True
        dataset.append(ext)
    return dataset

    


def set_augmentation(dataset,expand_ratio):
    l = len(dataset)
    times = int(expand_ratio)
    res = expand_ratio - times
    for i in range(l):
        is_aug = np.random.rand()<res
        dataset[i]['aug'] = False
        if is_aug:
            _data = copy.deepcopy(dataset[i])
            _data['aug'] = True
            dataset.append(_data)
    for j in range(times):
        for  i in range(l):
            _data = copy.deepcopy(dataset[i])
            _data['aug'] = True
            dataset.append(_data)
    return dataset


class seamPicDataLoader(dataset.Dataset):
    def __init__(self,opt,task,rank=None,debug=False):
        
        # self.rank = None
        # # print((rank is None))
        # # print(type(opt['device']) == str)
        # assert not((rank is None) ^ (type(opt['device']) == str))
        
        # if not rank is None:
        #     self.rank = rank
        self.debug=debug

        self.exp_name = opt["exp_name"]
        self.img_channels = opt["input_channels"]
        self.input_shape = (opt["input_height"],opt["input_width"])
        self.train_aug_expend_ratio = opt["train_aug_expend_ratio"]
        self.task = task
        self.small = False
        dataset = load_pics({"dataset_type":"labelme","path":os.path.join(ROOT,"data",self.exp_name,self.task)})
        dataset = set_balance(dataset,5) if len(dataset)>0 else dataset
        if self.task == "train":
            dataset = set_augmentation(dataset,self.train_aug_expend_ratio)
        else:
            dataset = set_augmentation(dataset,0)
        self.data = dataset# load_pics({"dataset_type":"labelme","path":os.path.join(ROOT,"data",self.exp_name,self.task)})
        self.inferencer = inferencer()
        # for dataset in opt["dataset"]:
        #     if dataset["dataset_type"] == "labelme":
        #         self.data += load_pics_from_videos(opt,self.dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        items = self.data[index]
        #if is video data, return raw data directly
        img_id = int(os.path.splitext(os.path.split(items['image_path'])[1])[0])
        im = cv.imread(items.get('image_path'))
        # if items['aug'] == True:

        
        # rgbim = im.copy() # im.clone()
        height,width,_ = im.shape   
        items_gt=[]
        if self.small:
            im = cv.resize(im,(width//2,height//2))
            im = im[:,:,np.newaxis]
        if self.img_channels == 1:
            # im = im[:,:,0:1]
            im = cv.cvtColor(im,cv.COLOR_RGB2GRAY)
            im = cv.resize(im,(self.input_shape[1],self.input_shape[0])) # im[:,:,0:1]
            im = np.expand_dims(im,axis=2)
        elif self.img_channels == 3:
            im = cv.resize(im,(self.input_shape[1],self.input_shape[0]))
            # pass
        else:
            raise Exception("Img channels size Error:{},it should be 1 or 3".format(self.img_channels))
        ps = []
        for item in items['items']:
            for p in item['p']:
                ps.append([p[0]/width,p[1]/height,p[2]/width,p[3]/height])

        # 
        if items['aug'] == True: # or self.debug == True:
            # im,ps = data_aug_half(im,ps,cp=[(ps[0][0]+ps[1][0])/2,(ps[0][1]+ps[1][1])/2])
            if self.inferencer is not None:
                if np.random.random() > 0.8:
                    im = self.inferencer.random_infer(im)
            im,ps = data_augmentation(im,ps)
            

        im = im.transpose(2,0,1).astype(np.float32)/255.
        ps = np.array(ps)
        for item_id,item in enumerate(items['items']):
            cp =None
            poses = np.zeros((2,2)) # [np.array([0,0])]*2 # np.zeros((3,2))#
            oris = np.zeros((2,2)) # [np.array([0,0])]*2
            # cp = [0,0]
            for i,point in enumerate(ps):
                point = np.array(point)
                # pos = np.array(point).astype(np.float32)
                # ori = np.array(point[2:]).astype(np.float32)
                if (point[:2]<0).any() or (point[:2]>1).any():
                    item_id = -1
                    break
            if item_id >= 0:
                ps = np.array(ps)[np.argsort(ps[:,2],axis=0)]
                poses = ps[:,:2] # [array[:2] for array in ps]
                oriL =  np.linalg.norm(np.array(ps[:,2:])-np.array(ps[:,:2]),axis=1)
                oris = np.divide(ps[:,2:]-ps[:,:2],oriL.reshape(-1,1)) # [(poses[2:][0]-pos[0])/oriL,(poses[2:][1]-pos[1])/oriL]
            item_ret = {'cp':np.array([0,0]) if item_id < 0 else np.array([(poses[0][0]+poses[1][0])/2,(poses[0][1]+poses[1][1])/2]),
                        'label':item['seam_type'],
                        'id':item_id,
                        'poses':[value for value in poses], # item_poses[:][:2],
                        'ori': [value for value in oris]} # item_poses[:][2:]}
            items_gt.append(item_ret)
        if self.debug:
            print(index)
            if index == 1169:
                aaa = 1
            im_rgb = np.floor(im*255)
            im_rgb = cv.cvtColor(im_rgb.transpose(1,2,0),cv.COLOR_GRAY2RGB)
            for _poses in items_gt:
                poss = _poses['poses']
                for i,p in enumerate(poss):
                    k = np.array([_poses['ori'][i][0]*0.1+p[0],_poses['ori'][i][1]*0.1+p[1]])
                    k = np.array([int(k[0]*self.input_shape[1]),int(k[1]*self.input_shape[0])])
                    k = k.astype(np.int64)
                    p = np.array([int(p[0]*self.input_shape[1]),int(p[1]*self.input_shape[0])])

                    cv.circle(im_rgb,p,5,(0,0,255),-1)
                    cv.line(im_rgb,p,k,(0,0,255),1)
                    cv.putText(im_rgb,str(i),[p[0],p[1]+10],cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            cv.imwrite(os.path.join(ROOT,'test','dataloader_test',"{}.bmp".format(index)),im_rgb)
        return (im,items_gt,img_id) #,poses,seamtypes,group_ids)
    

    def count_items(self):
        ans = [0,0,0,0]
        ans1 = [0,0,0,0]
        ans2 = [0,0,0,0]
        for item in self.data:
            ans[item['items'][0]['seam_type']]+=1
            if item['image_height'] == 600:
                ans1[item['items'][0]['seam_type']] += 1
            else:
                ans2[item['items'][0]['seam_type']] += 1
        return ans,ans1,ans2
        



    def state_dict(self):
        return {
            'data':self.data,
            'task':self.task,
            'small':self.small,
            'img_channels':self.img_channels,
            'exp_name':self.exp_name,
        }
    
    def load_state_dict(self,info):
        self.data = info['data']
        self.task = info['task']
        self.small = info['small']
        self.img_channels = info['img_channels']
        self.exp_name = info['exp_name']
    
def _heat_nms(heat,threshold=0.9,kernel=3):

    pad = (kernel-1)//2
    hmax = nn.functional.max_pool2d(heat,(kernel,kernel),stride=1,padding=pad)
    peak = (hmax==heat)
    keep = ((hmax==heat) & (heat > threshold)).float()
    indics = torch.nonzero(keep)
    result = {}
    result['heat'] = heat.cpu().detach().numpy()
    result['hmax'] = hmax.cpu().detach().numpy()
    result['peak'] = peak.cpu().detach().numpy()
    result['indics'] = indics.float().cpu().detach().numpy()
    result['final'] = (heat*keep).float().cpu().detach().numpy()
    savemat(os.path.join(ROOT,'results','experiment_5','heat_test_input.mat'),result)
    exit()

if __name__=="__main__":

    # test_dis
    opt = {"datasets":[],
           "device":"cuda:0",
           "exp_name":"exp_ori_1117",
           "train_ratio":0.7,
           "val_ratio":0.2,
            "input_width":320,
           "input_height":200,
           "input_channels":1,
           "train_batch_size":60,
           "train_aug_expend_ratio":2.5}
    # split.split_pics(opt)
    train_set = seamPicDataLoader(opt,"train",debug=True)
    i=0
    for s in train_set:
        i+=1

