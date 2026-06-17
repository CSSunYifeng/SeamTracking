from torch.utils.data import dataset
if __name__=="__main__":
    from loadset import load_videos,load_pics_from_videos,load_pics
    from split import generate_coco_label
else:
    from utils.dataset.loadset import load_videos,load_pics_from_videos,load_pics
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


def data_augmentation(im:ndarray,x:float,y:float)->Tuple[ndarray,float,float]:
    assert len(im.shape)==3
    assert im.shape[2]==1 or im.shape[2]==3
    assert im.dtype==np.uint8
    assert x>=0 and x<=1 and y>=0 and y<=1
    height,width,ch = im.shape
    x = int(x*width)
    y = int(y*height)
    kps = KeypointsOnImage([Keypoint(x=x, y=y)], shape=im.shape)
    dx = int(min(x,width-x)*0.5)
    dy = int(min(y,height-y)*0.5)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.TranslateX(px=(-dx,dx)),
        iaa.TranslateY(px=(-dy,dy)),
        #iaa.GaussianBlur(sigma=(0, 3.0)) #，
    ])
    image_aug, kps_aug = seq(image=im, keypoints=kps)
    return image_aug,kps_aug.keypoints[0].x/width,kps_aug.keypoints[0].y/height

class seamPicDataLoader(dataset.Dataset):
    def __init__(self,opt,task,rank=None):
        
        # self.rank = None
        # # print((rank is None))
        # # print(type(opt['device']) == str)
        # assert not((rank is None) ^ (type(opt['device']) == str))
        
        # if not rank is None:
        #     self.rank = rank

        self.exp_name = opt["exp_name"]
        self.img_channels = opt["input_channels"]
        self.input_shape = (opt["input_height"],opt["input_width"])
        self.task = task
        self.small = False
        self.data = load_pics({"dataset_type":"labelme","path":os.path.join(ROOT,"data",self.exp_name,self.task)})
        # for dataset in opt["dataset"]:
        #     if dataset["dataset_type"] == "labelme":
        #         self.data += load_pics_from_videos(opt,self.dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        #if is video data, return raw data directly
        if isinstance(item,list):
            return item
        im = cv.imread(item.get('image_path'))
        # rgbim = im.copy() # im.clone()
        height,width,_ = im.shape   
        x = item['items'][0]['p'][0][0]/width
        y = item['items'][0]['p'][0][1]/height

        # 
        if self.task == "train":
            im,x,y = data_augmentation(im,x,y)

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

        seamtype = item['items'][0]['seam_type']


        # rgbim = cv.cvtColor(im,cv.COLOR_GRAY2RGB) # DEBUG

        if self.small:
            im = cv.resize(im,(width//2,height//2))
            im = im[:,:,np.newaxis]
        im = im.transpose(2,0,1).astype(np.float32)/255.
        pos = np.array([x,y]).astype(np.float32)
        return (im,pos,seamtype)
    
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
