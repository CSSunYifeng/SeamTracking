import os
ROOT = os.getenv('SEAMTRACKING_ROOT')
import sys
sys.path.append(ROOT)
from typing import Callable,Tuple,List
import torch
from torch import Tensor
from torchvision import transforms
import numpy as np
from torch.utils.data import dataloader
from torch.utils.data.dataset import Dataset
import cv2 as cv
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt

import glob
import shutil
from tqdm import tqdm
import imageio
import time
import models.backbones.omniSeam.omnipose
import models.tasks.ori as ompi_mod
from scipy.io import savemat
import json
from factory import generateModel

def _heat_nms(heat,threshold=0.5,kernel=3):

    pad = (kernel-1)//2
    hmax = nn.functional.max_pool2d(heat,(kernel,kernel),stride=1,padding=pad)
    peak = (hmax==heat).to(torch.float)
    map = (heat > threshold).to(torch.float)
    keep = ((hmax==heat) & (heat > threshold)).float()
    indics = torch.nonzero(keep)
    return heat*keep,indics



class OutputModel(nn.Module):
    def __init__(self,model:nn.Module,nms_thresh):        
        super().__init__()
        self.cls_num = 4
        self.model = model
        self.src_shape = [0,0]
        self.nms_thresh = nms_thresh
        self.gray2rgb = transforms.Grayscale(num_output_channels=3)

    def preprocess(self,im):
        im = im.unsqueeze(1)
        im = im.tile((1,3,1,1))
        im = im.to(torch.float32)
        im = im/255.
        return im

	
    def forward(self,im):
        x = self.preprocess(im)
        x = self.model(x)
        x = self.postprocess(x)
        return x
	
    def heatmap_nms(self,heat,threshold=0.9,kernel=3):
        pad = (kernel-1)//2
        hmax = nn.functional.max_pool2d(heat,(kernel,kernel),stride=1,padding=pad)
        # peak = (hmax==heat).to(torch.float)
        # map = (heat > threshold).to(torch.float)
        keep = ((hmax==heat) & (heat > threshold)).float()
        # indics = torch.nonzero(keep,as_tuple=True)
        return keep

    def postprocess(self,x):
        hm,off,dir = x
        # savemat('heat_test.mat', {'heat': hm.detach().numpy(),'heat0': hm[0,:,:,:].detach().numpy()})
        hm = hm[:,:self.cls_num,:,:]
        
        hm = torch.sigmoid(hm)
        # _,_,height,width = hm.shape
        keep = self.heatmap_nms(hm,threshold=self.nms_thresh)
        results = torch.stack([hm[0,:,:,:],off[0,:,:,:],dir[0,:,:,:],keep[0,:,:,:]])
        
        return results

model_dir = os.path.join(ROOT, '/log/2025-01-17 14 56 26.954722 centernet__version_ywbr_omnipose_mod_half_ts/')
model_path = os.path.join(model_dir,'best.pth')
json_path = os.path.join(model_dir,'')
with open(os.path.join(json_path,'opt.json'),'r') as json_writer:
    json_value = json.load(json_writer)

net = generateModel(json_value)# ompi_mod.HRNet('2stage_double')
input_layer_num = 1


net.to('cpu')
net.load_state_dict(torch.load(model_path,map_location=torch.device('cpu'))['model_state_dict'])

from torch.nn.utils import parameters_to_vector
 

net.eval()

image_path = os.path.join(ROOT, '/data/cylinder_test/test/test.bmp')
height,width = 0,0
if input_layer_num == 1:
    im = cv.imread(image_path,cv.IMREAD_GRAYSCALE)
    im = cv.resize(im,(512,512))
    height,width = im.shape
    input_data = torch.from_numpy(im).unsqueeze(0)
    # im = np.expand_dims(im,2)
elif input_layer_num == 3:
    im = cv.imread(image_path)
    im = cv.resize(im,(512,512))
    height,width,_ = im.shape
    input_data = torch.from_numpy(im.transpose(2,0,1)).unsqueeze(0)
elif input_layer_num != 3:
        raise Exception("  ")
        height,width,_ = im.shape



torch.save(net,'./output/model.pt')

nms_thresh = 0.5
print("nms_threshold:{}".format(nms_thresh))
openvino_net = OutputModel(net,nms_thresh = nms_thresh)
openvino_net.eval()
output = openvino_net(input_data)
path = './output/model_ts.onnx'
torch.onnx.export(openvino_net,input_data,path,input_names = ['input'],output_names = ['output'])
print(path)
print("finished")

pass
