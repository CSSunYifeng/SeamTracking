import os
ROOT = os.getenv('SEAMTRACKING_ROOT')
import shutil
import torch
import cv2 as cv
import torchvision
import sys
import onnxruntime as ort
import factory
sys.path.append(ROOT)
# from cunt_dis import plt_dis
print(ROOT)
from utils.calss_score import micro_f1score,plot_confusion_matrix
# import model
from tqdm import tqdm

import copy
import torch.nn as nn
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils.dataset.loadset import load_labelme_data
import time
from scipy.io import savemat
from prettytable import *
import numpy as np
import torch.nn.functional as F
import loss as mls
from torch.utils.data.dataloader import DataLoader
import models.tasks.ori
from thop import profile
from torchinfo import summary
cls_num = 5
DEVICE = 'cuda:0'
# SAMPLE = ["479","440","402","220"]
# SAMPLE = ["1","2"]
# SAMPLE = ["G_760"]
SAMPLE = [] # "0"]
LOGDIRS = ['hrnets2_waspv2',
           'hrnets2',
           'resnet18',
           'resnet50',
           ]


def _heat_nms(heat,threshold=0.5,kernel=5,single=False):

    pad = (kernel-1)//2
    hmax = nn.functional.max_pool2d(heat,(kernel,kernel),stride=1,padding=pad)
    peak = (hmax==heat).to(torch.float)
    map = (heat > threshold).to(torch.float)
    keep = ((hmax==heat) & (heat > threshold)).float()
    indics = torch.nonzero(keep)
    indics_len = indics.shape[0]
    # result = {}
    # result['heat'] = heat.cpu().numpy()
    # result['hmax'] = hmax.cpu().numpy()
    # result['peak'] = peak.cpu().numpy()
    # result['map'] = map.cpu().numpy()
    # result['indics'] = indics.float().cpu().numpy()
    # result['final'] = (heat*keep).float().cpu().numpy()
    # savemat(os.path.join(ROOT,'results','experiment_6','heat_test_666.mat'),result)
    # exit()

    if single == True and indics_len > 0:
        nonzero_values = keep[indics[:,0],indics[:,1],indics[:,2],indics[:,3]]
        max_idx = torch.argmax(nonzero_values)
        indics = indics[max_idx].unsqueeze(0)
    return heat*keep,indics


def inference_data(im:torch.tensor,net:nn.Module,single=False):
    t = 0
    with torch.no_grad():
        a = time.time()
        output = net(im)
        b = time.time()
        t = b-a
    hm,off,dir = output

    hm = hm[:,:cls_num,:,:]
    hm = torch.sigmoid(hm)
    # _,_,height,width = hm.shape
    # savemat(os.path.join(ROOT,'results','experiment_6','off_test_2635.mat'),{'off':off.cpu().numpy()})
    _,indcs = _heat_nms(hm,single=single)
    results = []
    _,_,hm_height,hm_width = hm.shape
    off = off[:,:4,:,:].view(1,4,-1)
    dir = dir[:,:4,:,:].view(1,4,-1)
    for index in indcs :
        x = index[3]
        y = index[2]
        cat_idx = index[1]
        cof = hm[0,cat_idx,y,x]
        indices = y*hm_width+x
        r_x_l = (x/hm_width+off[0,0,indices]).to(torch.float)
        r_y_l = (y/hm_height+off[0,1,indices]).to(torch.float)
        r_x_r = (x/hm_width+off[0,2,indices]).to(torch.float)
        r_y_r = (y/hm_height+off[0,3,indices]).to(torch.float)

        # dir_vec1 = (dir[0,0,indices],dir[0,1,indices])
        # dir_vec2 = (dir[0,2,indices],dir[0,3,indices])
        # theta1 = torch.arctan2(dir[0,0,indices],dir[0,1,indices])
        # theta2 = torch.arctan2(dir[0,2,indices],dir[0,3,indices])
        ori11 = dir[0,0,indices]
        ori12 = dir[0,1,indices]
        ori21 = dir[0,2,indices]
        ori22 = dir[0,3,indices]

        cat_idx = cat_idx.to(torch.int64)
        cof = torch.floor(cof*100).to(torch.int64)
        result = (torch.stack([cat_idx,r_x_l,r_y_l,r_x_r,r_y_r,ori11,ori12,ori21,ori22,cof]).reshape(-1)).detach().cpu().numpy()
        results.append(result)
    return results,t

def inference_json(img_path,json_value,single=False):
    results = []
    time_list = []
    for value in json_value:
        if os.path.split(value['path'])[-1] == os.path.split(img_path)[-1]:
            if single and len(value['items']) > 0:
                max_idx = -1
                max_conf = 0
                for i,item in enumerate(value['items']):
                    if len(item['point']) < 1 or item['conf'] < 0.5:
                        continue
                    if item['conf'] > max_conf:
                        max_idx = i
                        max_conf = item['conf']
                result = [value['items'][max_idx]['cls']]
                result += value['items'][max_idx]['point'][0]
                result += value['items'][max_idx]['point'][2]
                result += value['items'][max_idx]['point'][1]
                result += value['items'][max_idx]['point'][3]
                # if max_conf > 0.5:
                results.append(result)

            else:
                for item in value['items']:
                    if len(item['point']) < 1 or item['conf'] < 0.5:
                        continue
                    result = [item['cls']]
                    # for p in item['points']:
                    result += item['point'][0]
                    result += item['point'][2]
                    result += item['point'][1]
                    result += item['point'][3]
                    results.append(result)
            return results,value['time']
    return results,2





def inference_onnx(im,ort_model,output_name,input_name):
    # im = cv.cvtColor(im.astype(np.uint8),cv.COLOR_GRAY2BGR)
    im = cv.resize(im,(512,512))
    im = im.transpose(2,0,1)
    im = np.expand_dims(im,(0))
    # im = im.astype(np.float32)
    # im = np.expand_dims(im,(3))
    # im = im.transpose(3,2,0,1)/255
    # im = im.astype(np.float32)

    ort_outputs = ort_model.run(output_names=[output_name.name],input_feed={input_name.name: im})      
    ort_outputs = np.array(ort_outputs[0])

    hm = ort_outputs[0,:4,:,:]
    off = ort_outputs[1,:4,:,:]
    dir = ort_outputs[2,:4,:,:]
    keep = ort_outputs[3,:4,:,:]
    # hm = torch.sigmoid(hm)
    indcs = np.transpose(np.nonzero(keep))
    _,hm_height,hm_width = hm.shape
    off = off.reshape(4,-1)
    dir = dir.reshape(4,-1)
    results=[]
    for index in indcs :
        x = index[2]
        y = index[1]
        cat_idx = index[0]
        cof = hm[cat_idx,y,x]
        indices = y*hm_width+x
        r_x_l = x/hm_width+off[0,indices]
        r_y_l = y/hm_height+off[1,indices]
        r_x_r = x/hm_width+off[2,indices]
        r_y_r = y/hm_height+off[3,indices]

        # dir_vec1 = (dir[0,0,indices],dir[0,1,indices])
        # dir_vec2 = (dir[0,2,indices],dir[0,3,indices])
        # theta1 = torch.arctan2(dir[0,0,indices],dir[0,1,indices])
        # theta2 = torch.arctan2(dir[0,2,indices],dir[0,3,indices])

        ori11 = dir[0,indices]
        ori12 = dir[1,indices]
        ori21 = dir[2,indices]
        ori22 = dir[3,indices]

        # cat_idx = cat_idx.to(np.int64)
        cof = np.floor(cof*100)
        result = (np.stack([cat_idx,r_x_l,r_y_l,r_x_r,r_y_r,ori11,ori12,ori21,ori22,cof]).reshape(-1))
        results.append(result)   
    return results,0
        


def labelmeDataLoader(data_set_path):
    return load_labelme_data(data_set_path)

def freeDataLoader(data_set_path):
    ret = []
    for root,dirs,files in os.walk(data_set_path):
        for file in files:
            if file.endswith('bmp') or file.endswith('jpg') or file.endswith('png'):
                value = {'image_path':os.path.join(root,file)}
                ret.append(value)
    return ret

model_recommend = {}
TABLE = []

def get_dataset(exp,with_lable = True):
    if with_lable:
        return labelmeDataLoader(os.path.join(ROOT,"data",exp,"test"))
    else:
        return freeDataLoader(os.path.join(ROOT,"data",exp,"test"))

def calc_flops(device,net,input_shape): # ,input_shape=(1,1,600,1088)):
    # input_shape = (1,3,512,512)
    # net.select_to_device(current,[0],[0],'cpu')
    d = torch.rand(*input_shape).to(device)
    flops, params = profile(net, (d,))
    print("model flops:{} GFLOPs".format(flops/1e9))
    return flops/1000000000


def test_public(model_path,model_name,net,img_channel,data_slice=slice(None,None),save_figs=False,input_shape=(1088,600),exp:str='exp_3',inference=inference_data,with_label=True,GFlOP=np.inf,half=False,enable_debbug_F1=False):
    device = DEVICE

    # calc_flops(device,net,(1,3,512,512))
    # 'omnipose_mod_2stage'
    flops = GFlOP
    if isinstance(net, nn.Module):
        net.to(device)
        # if model_name == 'omnipose_mod_1stage':
        #     flops = calc_flops(device,net,[1,3,512,512])
        #     summary(net,(3,512,512),device=device)
        # else:

            

        # hook_net = model_factory(type).to(device)
        # hook_net.eval()
        # for layer in hook_net.children():
        #     layer.register_forward_hook(hook_fn)
        # hook_input_tensor = torch.randn(1,3,512,512).to(device)
        # hook_net(hook_input_tensor)
        

        net_load = torch.load(model_path,map_location=torch.device(device))
        net_params = net_load['model_state_dict']
        print(net_load['train_set_state_dict']['data'][0]['image_path'])
        net.load_state_dict(net_params)
        if half:
            net = net.half()
        net.eval()


    datas = get_dataset(exp,with_lable=with_label)
    final_results = {'model_name':model_name,'results':[]}
    dises = []
    dises_cls={0:[],1:[],2:[],3:[]}
    datas = datas[data_slice]
    t = 0
    cunt = 0
    result_plot_path = os.path.join(ROOT,'results',exp,'plot',model_name)
    gt_labels = []
    labels = []
    topk_1_cunt = 0
    aa_1_cunt = 0
    topk_2_cunt = 0
    aa_2_cunt = 0

    # dis_sum = 0
    dis_list = []
    angle_list = []
    valid_points_num = 0
    label_pred = []
    label_gt = []

    top_error_imgs = []

    loc_map = []
    angular_map = []
    valid_mask = []
    loc_map_keys = []
    loc_map_src_keys = []
    for d_idx,data in enumerate(tqdm(datas)):
        
        # if not os.path.split(data['image_path'])[1] == "2820.png":
        #     continue
        # assert os.path.split(data['image_path'])[1] != "1842.bmp"
        # assert data['items'][0]['seam_type'] != 0
        c = os.path.exists(data['image_path'])
        rgbim = cv.imread(data['image_path'])
        if with_label:
            gt_labels.append(data['items'][0]['seam_type'])
        src_scale = rgbim.shape
        src_scale = (src_scale[1],src_scale[0])

        # flip = int(src_scale[0]/2)
        # rgbim[:,:flip] = rgbim[:,flip:][:,::-1]

        rgbim = cv.resize(rgbim,input_shape)
        im = None
        if img_channel == 1:
            im = cv.cvtColor(rgbim,cv.COLOR_RGB2GRAY)
            im = np.expand_dims(im,axis=2)
        else:
            im = rgbim
        # savemat("./heat_test_2635_src.mat",{'im':im})
        # im[:,:int(input_shape[1]/2)] = im[:,int(input_shape[1]/2):][:,::-1]
        
        im = im.transpose(2,0,1).astype(np.float32)# im.reshape((1,img_channel,IMG_SHAPE[1],IMG_SHAPE[0]))
        im = im/255
        
        im = torch.from_numpy(im).unsqueeze(0)
        im = im.to(device)
        if half:
            im = im.half()

        results,dt = inference(im,net) if isinstance(net, nn.Module) else inference(data['image_path'],net)
        # if np.any(np.isnan(results)): # or not np.array(results[1:]).all():
        #     results = []

        # if len(results)< 1 or results[0][0] == 0:
        #     continue
        # print(results)
        t += dt
        # print(dt)
        result_path =  os.path.join(ROOT,result_plot_path,'imgs')
        if save_figs: #  and cunt % 100 == 0:
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            img_name = os.path.split(data['image_path'])[1]
            ac = []
            for subList in results:
                ac.append([d_value if not np.isnan(d_value) else 1 for d_value in subList])
            results = ac
            if(os.path.splitext(img_name)[0] in SAMPLE):
                experiment_img_result(ac,rgbim,result_path,img_name,
                                    data,src_scale,input_shape,0.25,with_label)
            generate_img_result(ac,rgbim,result_path,img_name,
                                data,src_scale,input_shape,with_label)

            # print("save img")


        if with_label:
            # gt_label = [value['label'] for value in data['items']]
            gt_dict = {}
            for i in range(0,len(data['items'])):
                gt_dict[i] = []
            left_map = np.zeros([len(data['items']),len(results)])+np.nan
            right_map = np.zeros([len(data['items']),len(results)])+np.nan
            angle_map_left = np.zeros([len(data['items']),len(results)])+np.nan
            angle_map_right = np.zeros([len(data['items']),len(results)])+np.nan
            # results2 = [results[0]] if len(results) > 0 else []
            # assert len(results) <= 1
            for result_idx,pred_item in enumerate(results):
                cunt += 1
                total_dis = 0
                p_list = np.array([[pred_item[1],pred_item[2]],[pred_item[3],pred_item[4]]])
                scale_m = np.array([[input_shape[0],0],[0,input_shape[1]]])
                p_list = p_list@scale_m

                v_list = np.array([np.array([pred_item[5],pred_item[6]])/np.linalg.norm(np.array([pred_item[5],pred_item[6]])),
                                   np.array([pred_item[7],pred_item[8]])/np.linalg.norm(np.array([pred_item[7],pred_item[8]]))])
                # a_list = np.array([np.arctan2(pred_item[6],pred_item[5]),np.arctan2(pred_item[8],pred_item[7])])
                # print(data['image_path'])
                assert len(data['items'])>0
                for i,item in enumerate(data['items']):
                    for j,p in enumerate(item['p']):
                        pred_p = p_list[j]
                        p = [p[0]/data['image_width']*input_shape[0],
                             p[1]/data['image_height']*input_shape[1],
                             p[2]/data['image_width']*input_shape[0],
                             p[3]/data['image_height']*input_shape[1]]
                        _dis = np.linalg.norm(np.array(p[:2])-np.array(pred_p))
                        loc_map.append([p[:2],pred_p,[_dis,_dis]])
                        # keykey = data['external']['image_path'] if 'external' in data.keys() else data['image_path']
                        if 'external' in data.keys():
                                loc_map_src_keys.append(data['external']['image_path'])
                        if j == 0:
                            left_map[i][result_idx] = _dis
                        elif j == 1:
                            right_map[i][result_idx] = _dis
                        loc_map_keys.append(data['image_path'])
                        valid_mask.append(True)

                    left_pred_v = v_list[0]
                    right_pred_v = v_list[1]
                    left_gt_vector = [data['items'][i]['p'][0][2]-data['items'][i]['p'][0][0],data['items'][i]['p'][0][3]-data['items'][i]['p'][0][1]]
                    right_gt_vector = [data['items'][i]['p'][1][2]-data['items'][i]['p'][1][0],data['items'][i]['p'][1][3]-data['items'][i]['p'][1][1]]

                    left_gt_v = np.array(left_gt_vector)
                    left_gt_v = left_gt_v/np.linalg.norm(left_gt_v)
                    # left_gt_a = np.arctan2(left_gt_v[1],left_gt_v[0])
                    right_gt_v = np.array(right_gt_vector)
                    right_gt_v = right_gt_v/np.linalg.norm(right_gt_v)                
                    # right_gt_a = np.arctan2(right_gt_v[1],right_gt_v[0])
                    # if (data['image_path'] == os.path.join(ROOT,"data/exp_ori_total3/test/1422.bmp")):
                    #     a = 1
                    diff_left_angle = np.dot(left_pred_v,left_gt_v)
                    diff_left_angle = np.arccos(1 if diff_left_angle > 1 else diff_left_angle)
                    diff_right_angle = np.dot(right_pred_v,right_gt_v)
                    diff_right_angle = np.arccos(1 if diff_right_angle > 1 else diff_right_angle)

                    angle_map_left[i][result_idx] = diff_left_angle/np.pi*180
                    angle_map_right[i][result_idx] = diff_right_angle/np.pi*180

                    angular_map.append([angle_map_left[i][result_idx],angle_map_right[i][result_idx]])
                    
                    # valid_mask.append(True)

            if(len(results)==0):
                # for i in range(2):
                for i,item in enumerate(data['items']):
                    for j,p in enumerate(item['p']):
                        # pred_p = p_list[j]
                        p = [p[0]/data['image_width']*input_shape[0],
                             p[1]/data['image_height']*input_shape[1],
                             p[2]/data['image_width']*input_shape[0],
                             p[3]/data['image_height']*input_shape[1]]
                        loc_map.append([p[:2],[-1,-1],[-1,-1]])
                        # keykey = data['external']['image_path'] if 'external' in data.keys() else data['image_path']
                        if 'external' in data.keys():
                            loc_map_src_keys.append(data['external']['image_path'])
                        loc_map_keys.append(data['image_path'])
                        valid_mask.append(False)
                        # assert data['image_path'] != "197.bmp"
                    angular_map.append([-1,-1])

            # assert len(angular_map) == (d_idx+1)
            # print(len(angular_map))
            for k in range(len(data['items'])):
                if data['items'][k]['seam_type'] == 0:
                    continue
                valid_points_num += 2
                if (np.all(left_map[k]==-1)):
                    label_pred.append(-1)
                    if (enable_debbug_F1):
                        print("FN img:{}".format(data['image_path']))
                else:
                    min_value_idx = np.argmin(left_map[k])
                    left_map[k][left_map[k] != left_map[k][min_value_idx]]  = np.nan
                    right_map[k][right_map[k] != right_map[k][min_value_idx]]  = np.nan
                    angle_map_left[k][angle_map_left[k] != angle_map_left[k][min_value_idx]]  = np.nan
                    angle_map_right[k][angle_map_right[k] != angle_map_right[k][min_value_idx]]  = np.nan
                    label_pred.append(int(results[min_value_idx][0]))
                    if(data['items'][k]['seam_type'] == results[min_value_idx][0]):
                        dis_list.append(left_map[k][min_value_idx])
                        dis_list.append(right_map[k][min_value_idx])
                        angle_list.append(np.abs(angle_map_left[k][min_value_idx]))
                        angle_list.append(np.abs(angle_map_right[k][min_value_idx]))
                    else:
                        if (enable_debbug_F1):
                            print("FN img:{}".format(data['image_path']))
                label_gt.append(data['items'][k]['seam_type'])
            # print((d_idx+1) *2 - valid_points_num)
            for kk in range(len(results)):
                if (np.all(left_map[:,kk]==-1)):
                    label_gt.append(-1)
                    if (enable_debbug_F1):
                        print("FP img:{}".format(data['image_path']))
                else:
                    min_value_idx = np.argmin(left_map[:,min(k,len(results)-1)])
                    label_gt.append(data['items'][min_value_idx]['seam_type'])
                    if (enable_debbug_F1):
                        if(data['items'][k]['seam_type'] != results[min_value_idx][0]):
                            print("FP img:{}".format(data['image_path']))
                label_pred.append(int(results[kk][0]))


                # angle_list.append(diff_left_angle/np.pi*180)
                # angle_list.append(diff_right_angle/np.pi*180)

                # dis_list.append(min_dis)
                # top_error_imgs.append(data['image_path'])
                # point_sum += 1
                
    # sorted_idcs = np.argsort(dis_list)[::-1]
    # print("max_error:{}".format([os.path.split(top_error_imgs[i])[-1] for i in sorted_idcs[0:10]]))

    assert len(label_gt) == len(label_pred)
    cm_path = os.path.join(result_path,'cm')
    if not os.path.exists(cm_path):
        os.makedirs(cm_path)
    plot_confusion_matrix(label_pred,label_gt,6,[0,1,2,3,4,5,-1],os.path.join(cm_path,'confusion_matrix.png'))


    if with_label:
        dis = np.average(dis_list)
        std = np.std(dis_list)
        var = np.var(dis_list)
        pmax = 0 if len(dis_list) == 0 else np.max(dis_list)
        pmin = 0 if len(dis_list) == 0 else np.min(dis_list)
        for _d in dis_list:
            if _d < 1:
                topk_1_cunt += 1
            if _d < 2:
                topk_2_cunt += 1
        
        for _a in angle_list:
            if _a < 1:
                aa_1_cunt += 1
            if _a < 2:
                aa_2_cunt += 1
        angle = np.average(angle_list)
        angle_std = np.std(angle_list)
        angle_var = np.var(angle_list)
        angle_max = 0 if len(angle_list) == 0 else np.max(angle_list)
        angle_min = 0 if len(angle_list) == 0 else np.min(angle_list)
        f1_score = micro_f1score(label_pred,label_gt,6)
        assert d_idx + 1 - len(datas) == 0
        points_num = len(datas)*2 # cunt # len(datas)*2
        final_results['results']=[model_name,
                                  t/len(datas) if len(datas)>0 else 0 ,
                                  flops,
                                  f1_score,
                                #   topk_1_cunt/valid_points_num if valid_points_num > 0 else 0,
                                #   topk_2_cunt/valid_points_num if valid_points_num > 0 else 0,
                                  topk_1_cunt/points_num if points_num > 0 else 0,
                                  topk_2_cunt/points_num if points_num > 0 else 0,
                                  dis,
                                  std,
                                  var,
                                  pmax,
                                  pmin,
                                #   aa_1_cunt/valid_points_num if valid_points_num > 0 else 0,
                                #   aa_2_cunt/valid_points_num if valid_points_num> 0 else 0,
                                  aa_1_cunt/points_num if points_num > 0 else 0,
                                  aa_2_cunt/points_num if points_num > 0 else 0,
                                  angle,
                                  angle_std,
                                  angle_var,
                                  angle_max,
                                  angle_min]
        final_results['loc_map'] = np.array(loc_map)
        final_results['loc_map_keys'] = loc_map_keys
        final_results['angular_map'] = angular_map
        final_results['valid_mask'] = valid_mask
        final_results['loc_map_src_keys'] = loc_map_src_keys
        return final_results

def experiment_img_result(results,rgbim,result_path,result_name,labels,src_scale,input_shape,scale2,with_label:bool):
    dscale = scale2 # 0.25
    
    rect = [int(input_shape[0]*dscale),int(input_shape[1]*dscale)]
    if len(results) <= 0:
        return
    
    image_clone = copy.deepcopy(rgbim)
    for idx in range(len(results)):
        result = results[idx]

        ss = 0.2
        for jdx,label in enumerate(labels['items']):
            # label = labels['items'][idx]
            px_l = int(label['p'][0][0]/src_scale[0]*input_shape[0])
            py_l = int(label['p'][0][1]/src_scale[1]*input_shape[1])
            px_r = int(label['p'][1][0]/src_scale[0]*input_shape[0])
            py_r = int(label['p'][1][1]/src_scale[1]*input_shape[1])
            scale_m = np.array([[input_shape[0]/src_scale[0],0],[0,input_shape[1]/src_scale[1]]])
            dir_l = (np.array([label['p'][0][2:]])@scale_m)[0] \
                    -(np.array([label['p'][0][:2]])@scale_m)[0]
            dir_l = dir_l/np.linalg.norm(dir_l)*ss*input_shape[0]
            dir_r = (np.array([label['p'][1][2:]])@scale_m)[0] \
                    -(np.array([label['p'][1][:2]])@scale_m)[0]
            dir_r = dir_r/np.linalg.norm(dir_r)*ss*input_shape[0]
            
            ppx_l = int(result[1]*input_shape[0])
            ppy_l = int(result[2]*input_shape[1])
            ppx_r = int(result[3]*input_shape[0])
            ppy_r = int(result[4]*input_shape[1])
            cp = ((ppx_l+ppx_r)/2,(ppy_l+ppy_r)/2)

            new_rect = [int(cp[0]-rect[0]/2),int(cp[0]+rect[0]/2),int(cp[1]-rect[1]/2),int(cp[1]+rect[1]/2)]

            if new_rect[0] < 0:
                new_rect[0] -= new_rect[0]
                new_rect[1] -= new_rect[0]
            if new_rect[2] < 0:
                new_rect[2] -= new_rect[1]
                new_rect[3] -= new_rect[1]
            if new_rect[1] > input_shape[0]:
                new_rect[0] -= new_rect[1]-input_shape[0]
                new_rect[1] -= new_rect[1]-input_shape[0]
            if new_rect[3] > input_shape[1]:
                new_rect[2] -= new_rect[3]-input_shape[1]
                new_rect[3] -= new_rect[3]-input_shape[1]


            new_image = copy.deepcopy(image_clone)[new_rect[2]:new_rect[3],new_rect[0]:new_rect[1]]

            # cv.rectangle(rgbim,(new_rect[0],new_rect[2]),(new_rect[1],new_rect[3]),(0,0,255),1)
            # cv.circle(rgbim,[int(cp[0]),int(cp[1])],10,(0,0,255),1)

            plist = (np.array([[px_l,py_l],[px_r,py_r]])-np.array([[new_rect[0],new_rect[2]],[new_rect[0],new_rect[2]]]))*1/dscale
            pplist = (np.array([[ppx_l,ppy_l],[ppx_r,ppy_r]])-np.array([[new_rect[0],new_rect[2]],[new_rect[0],new_rect[2]]]))*1/dscale

            px_l = int(plist[0,0])
            py_l = int(plist[0,1])
            px_r = int(plist[1,0])
            py_r = int(plist[1,1])

            ppx_l = int(pplist[0,0])
            ppy_l = int(pplist[0,1])
            ppx_r = int(pplist[1,0])
            ppy_r = int(pplist[1,1])
            

            
            new_img = cv.resize(new_image,tuple(input_shape))

            

            cv.circle(new_img,(px_l,py_l),5,(255,0,0),-1)
            # cv.putText(rgbim,str(0),(px_l,py_l-20),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
            cv.arrowedLine(new_img,(px_l,py_l),(int(px_l+dir_l[0]),int(py_l+dir_l[1])),(255,0,0),2)
            cv.circle(new_img,(px_r,py_r),5,(255,0,0),-1)
            # cv.putText(rgbim,str(1),(px_r,py_r-20),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
            cv.arrowedLine(new_img,(px_r,py_r),(int(px_r+dir_r[0]),int(py_r+dir_r[1])),(255,0,0),2)
            cv.line(new_img,(px_r,py_r),(px_l,py_l),(255,0,0),1)

            cv.circle(new_img,(ppx_l,ppy_l),5,(0,0,255),-1)
            # cv.putText(rgbim,str(0),(ppx_l,ppy_l),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            cv.arrowedLine(new_img,(ppx_l,ppy_l),(int(ppx_l+ss*result[5]*input_shape[0]),int(ppy_l+ss*result[6]*input_shape[1])),(0,0,255),2)
            cv.circle(new_img,(ppx_r,ppy_r),5,(0,0,255),-1)
            # cv.putText(rgbim,str(1),(ppx_r,ppy_r),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            cv.arrowedLine(new_img,(ppx_r,ppy_r),(int(ppx_r+ss*result[7]*input_shape[0]),int(ppy_r+ss*result[8]*input_shape[1])),(0,0,255),2)
            cv.line(new_img,(ppx_r,ppy_r),(ppx_l,ppy_l),(0,0,255),1)
            cv.putText(new_img,str('left'),(int(ppx_l+ss*result[5]*input_shape[0]),int(ppy_l+ss*result[6]*input_shape[1])),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            cv.putText(new_img,str('right'),(int(ppx_r+ss*result[7]*input_shape[0]),int(ppy_r+ss*result[8]*input_shape[1])),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            # cv.putText(rgbim,str(get_cat_str(int(result[0]))),(int((ppx_r+ppx_l)/2),int((ppy_r+ppy_l)/2)+20),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
            # cv.putText(rgbim,str(get_cat_str(int(result[0]))),(0,0+20),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
            cv.rectangle(rgbim,(new_rect[0],new_rect[2]),(new_rect[1],new_rect[3]),(0,0,255),2)
            # dst = cv.addWeighted(rgbim,0.7,hm,0.3,0)
            if not os.path.exists(os.path.join(result_path,'experiment')):
                os.makedirs(os.path.join(result_path,'experiment'))
            cv.imwrite(os.path.join(result_path,'experiment',os.path.splitext(result_name)[0]+f"_{idx}_{jdx}.png"),new_img)

def generate_img_result(results,rgbim,result_path,result_name,labels,src_scale,input_shape,with_label:bool):

    # cv.resize(hm,rgbim.shape[:2],hm)
    # result = [value for value in result]

    def get_cat_str(type_idx:int):
        ret = ""
        if type_idx == 0:
            ret = "welded"
        elif type_idx == 1:
            ret = "butt"
        elif type_idx == 2:
            ret = "internal corner"
        elif type_idx == 3:
            ret = "lap"
        elif type_idx == 4:
            ret = "external corner"
        return ret


    if with_label:
        for label in labels['items']:

            px_l = int(label['p'][0][0]/src_scale[0]*input_shape[0])
            py_l = int(label['p'][0][1]/src_scale[1]*input_shape[1])
            px_r = int(label['p'][1][0]/src_scale[0]*input_shape[0])
            py_r = int(label['p'][1][1]/src_scale[1]*input_shape[1])
            scale_m = np.array([[input_shape[0]/src_scale[0],0],[0,input_shape[1]/src_scale[1]]])
            dir_l = (np.array([label['p'][0][2:]])@scale_m)[0] \
                   -(np.array([label['p'][0][:2]])@scale_m)[0]
            dir_r = (np.array([label['p'][1][2:]])@scale_m)[0] \
                   -(np.array([label['p'][1][:2]])@scale_m)[0]
            dir_l = dir_l/np.linalg.norm(dir_l)*300
            dir_r = dir_r/np.linalg.norm(dir_r)*300
            cv.circle(rgbim,(px_l,py_l),5,(255,0,0),-1)
            # cv.putText(rgbim,str(0),(px_l,py_l-20),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
            cv.arrowedLine(rgbim,(px_l,py_l),(int(px_l+dir_l[0]),int(py_l+dir_l[1])),(255,0,0),2)
            cv.circle(rgbim,(px_r,py_r),5,(255,0,0),-1)
            # cv.putText(rgbim,str(1),(px_r,py_r-20),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
            cv.arrowedLine(rgbim,(px_r,py_r),(int(px_r+dir_r[0]),int(py_r+dir_r[1])),(255,0,0),2)
            cv.line(rgbim,(px_r,py_r),(px_l,py_l),(255,0,0),1)
            
    for result in results:
        ppx_l = int(result[1]*input_shape[0])
        ppy_l = int(result[2]*input_shape[1])
        ppx_r = int(result[3]*input_shape[0])
        ppy_r = int(result[4]*input_shape[1])
        cv.circle(rgbim,(ppx_l,ppy_l),5,(0,0,255),-1)
        # cv.putText(rgbim,str(0),(ppx_l,ppy_l),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
        cv.arrowedLine(rgbim,(ppx_l,ppy_l),(int(ppx_l+0.1*result[5]*input_shape[0]),int(ppy_l+0.1*result[6]*input_shape[1])),(0,0,255),2)
        cv.circle(rgbim,(ppx_r,ppy_r),5,(0,0,255),-1)
        # cv.putText(rgbim,str(1),(ppx_r,ppy_r),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
        cv.arrowedLine(rgbim,(ppx_r,ppy_r),(int(ppx_r+0.1*result[7]*input_shape[0]),int(ppy_r+0.1*result[8]*input_shape[1])),(0,0,255),2)
        cv.line(rgbim,(ppx_r,ppy_r),(ppx_l,ppy_l),(0,0,255),1)
        cv.putText(rgbim,str('left'),(int(ppx_l+0.1*result[5]*input_shape[0]),int(ppy_l+0.1*result[6]*input_shape[1])),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
        cv.putText(rgbim,str('right'),(int(ppx_r+0.1*result[7]*input_shape[0]),int(ppy_r+0.1*result[8]*input_shape[1])),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
        # cv.putText(rgbim,str(get_cat_str(int(result[0]))),(int((ppx_r+ppx_l)/2),int((ppy_r+ppy_l)/2)+20),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
        # cv.putText(rgbim,str(get_cat_str(int(result[0]))),(0,0+20),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
        
        
    dst = rgbim
    # dst = cv.addWeighted(rgbim,0.7,hm,0.3,0)
    cv.imwrite(os.path.join(result_path,result_name),dst)

def cn_resnet_MB_innerdecoder_infer(model_path,model_name,layer_num,input_shape,channels,exp='exp_dis_5',with_label=False,save_figs=False):
    device = DEVICE

    # summary(test_model,(1,3,512,512),device=device,depth=6)
    if channels == 3:
        models = models.tasks.ori.MyModelResNet_head_MB_mult_innerdecoder(layer_num)
    results = test_public(model_path,model_name,models,channels,data_slice=slice(None,None),save_figs=save_figs,input_shape=input_shape,exp=exp,inference=inference_data,with_label=with_label)
    flops = calc_flops(device,models,[1,3,512,512])
    if results is not None:
        results['results'][2] = flops 
    return results

def hrnets2_waspv2_mod_infer(model_path,model_name,input_shape,exp='exp_dis_5',type='common',with_label=False,half=False,save_figs=False):
    # calc_flops('cpu',models.tasks.ori.Omnipose_mod_mult(type),[1,3,512,512])
    if type != None:
        types = type.split('_')
        is_no_gauss = "no" in types and "gauss" in types # type[-9:] == "_no_gauss"
    model_factory = models.tasks.ori.Omnipose_mod_mult # models
    device = DEVICE
    test_model = model_factory(type,with_gauss_filter= not is_no_gauss).to(device)
    flops = calc_flops(device,test_model,[1,3,512,512])
    summary(test_model,(1,3,512,512),device=device,depth=6)
    net = model_factory(type,with_gauss_filter= not is_no_gauss)
    return test_public(model_path,model_name,net,3,data_slice=slice(None,None),save_figs=save_figs,input_shape=input_shape,exp=exp,inference=inference_data,with_label=with_label,GFlOP=flops,half=half)

def hrnets2_mod_infer(model_path,model_name,input_shape,exp='exp_dis_5',type='common',with_label=False,save_figs=False):
    # calc_flops('cpu',models.tasks.ori.Omnipose_mod_mult(type),[1,3,512,512])
    if type != None:
        types = type.split('_')
        is_no_gauss = "no" in types and "gauss" in types # type[-9:] == "_no_gauss"
    model_factory = models.tasks.ori.HRNet # models
    device = DEVICE
    test_model = model_factory(type,with_gauss_filter= not is_no_gauss).to(device)
    flops = calc_flops(device,test_model,[1,3,512,512])
    summary(test_model,(1,3,512,512),device=device,depth=6)
    net = model_factory(type,with_gauss_filter= not is_no_gauss)
    return test_public(model_path,model_name,net,3,data_slice=slice(None,None),save_figs=save_figs,input_shape=input_shape,exp=exp,inference=inference_data,with_label=with_label,GFlOP=flops)

def mobilenet_infer(model_path,model_name,input_shape,exp='exp_dis_5',with_label=False,half=False,depth=5,save_figs=False):
    model_factory = models.tasks.ori.mobilenetv2_decoder # models
    device = DEVICE
    test_model = model_factory().to(device)
    flops = calc_flops(device,test_model,[1,3,512,512])
    summary(test_model,(1,3,512,512),device=device,depth=6)
    net = model_factory()
    return test_public(model_path,model_name,net,3,data_slice=slice(None,None),save_figs=save_figs,input_shape=input_shape,exp=exp,inference=inference_data,with_label=with_label,GFlOP=flops,half=half)

def shufflenet_infer(model_path,model_name,input_shape,exp='exp_dis_5',with_label=False,half=False,depth=5,save_figs=False):
    model_factory = models.tasks.ori.shufflenet_decoder # models
    device = DEVICE
    test_model = model_factory().to(device)
    flops = calc_flops(device,test_model,[1,3,512,512])
    summary(test_model,(1,3,512,512),device=device,depth=6)
    net = model_factory()
    return test_public(model_path,model_name,net,3,data_slice=slice(None,None),save_figs=save_figs,input_shape=input_shape,exp=exp,inference=inference_data,with_label=with_label,GFlOP=flops,half=half)

def json_infer(file_path,model_name,input_shape,exp='exp_dis_5',with_label=False,save_figs=False):
    with open(file_path,'r') as json_reader:
        _value = json.load(json_reader)
    json_value = _value
    return test_public(None,model_name,json_value,3,data_slice=slice(None,None),save_figs=save_figs,input_shape=input_shape,exp=exp,inference=inference_json,with_label=with_label,GFlOP=0)

def test_model(model_path,exp,input_shape=(512,512),save_figs=True,with_label=True,half=False):
    with open(os.path.join(model_path,'opt.json'),'r') as json_reader:
        model_config = json.load(json_reader)
    model_name = model_config['model']
    # calc_flops('cpu',models.tasks.ori.Omnipose_mod_mult(type),[1,3,512,512])
    # if mtype != None:
    #     types = type.split('_')
    #     is_no_gauss = "no" in types and "gauss" in types # type[-9:] == "_no_gauss"
    # model_factory = factory.generateModel(model_name) # models.tasks.ori.Omnipose_mod_mult # models
    device = DEVICE
    test_model = factory.generateModel(model_config).to(device) #  model_factory(mtype,with_gauss_filter= not is_no_gauss).to(device)
    flops = calc_flops(device,test_model,[1,3,512,512])
    summary(test_model,(1,3,512,512),device=device,depth=6)
    net = factory.generateModel(model_config) # model_factory(mtype,with_gauss_filter= not is_no_gauss)
    return test_public(os.path.join(model_path,'best.pth'),model_name,net,3,data_slice=slice(None,None),save_figs=save_figs,input_shape=input_shape,exp=exp,inference=inference_data,with_label=with_label,GFlOP=flops,half=half)
    pass

def test():
    # local infer
    results = []
    exp = 'exp_ori' # 'mult_test' # 'exp_ori_total6' # 'exp_ori_inter3' #'exp_ori_total6' # 'exp_ori_wh_new_type1' # 'pybullet_cylinder'# 'pybullet_intersection' # 'pybullet_ts2' # 'new_camera' # 'exp_dis_5' # 'cylinder_test' '2024_09_02__09_00_15' 'exp_dis_5'
    with_label = True
    log_dirs = LOGDIRS
    for dir in log_dirs:
        ret = test_model(os.path.join('log',dir),exp,with_label = with_label)
        if ret != None:
            del ret['loc_map']
            results.append(ret) 
    diff_path = os.path.join(ROOT,f'results/{exp}/diff')
    if not os.path.exists(diff_path):
        os.makedirs(diff_path)
    with open(os.path.join(diff_path, exp+'.json'),'w') as json_writer:
        json.dump(results,json_writer)

    # json infer
    results_dict = {'yolov8':'experiments/yolo_results/results_yolov8.json',
                    'pose_anything':'experiments/pose_anything_results/pose_anything_results.json'}
    for ind_model_name in results_dict.keys():
        ret = json_infer(results_dict[ind_model_name],ind_model_name,(512,512),exp=exp,with_label=with_label,save_figs=True)
        if ret != None:
            del ret['loc_map']
            results.append(ret) 

    if len(results) > 0:
        analysis_results(results)

def test_trans_enhance():
    # local infer
    results = []
    exp = 'exp_ori_enhance' # 'mult_test' # 'exp_ori_total6' # 'exp_ori_inter3' #'exp_ori_total6' # 'exp_ori_wh_new_type1' # 'pybullet_cylinder'# 'pybullet_intersection' # 'pybullet_ts2' # 'new_camera' # 'exp_dis_5' # 'cylinder_test' '2024_09_02__09_00_15' 'exp_dis_5'
    with_label = True
    log_dirs = LOGDIRS
    for dir in log_dirs:
        ret = test_model(os.path.join('log',dir),exp,with_label = with_label)
        if ret != None:
            del ret['loc_map']
            results.append(ret) 
    diff_path = os.path.join(ROOT,f'results/{exp}/diff')
    if not os.path.exists(diff_path):
        os.makedirs(diff_path)
    with open(os.path.join(diff_path, exp+'.json'),'w') as json_writer:
        json.dump(results,json_writer)

    # json infer
    results_dict = {'yolov8':'experiments/yolo_results/results_yolov8_offset_2507.json',
                    'pose_anything':'experiments/pose_anything_results/pose_anything_offset_2507_results.json'}
    for ind_model_name in results_dict.keys():
        ret = json_infer(results_dict[ind_model_name],ind_model_name,(512,512),exp=exp,with_label=with_label,save_figs=True)
        if ret != None:
            del ret['loc_map']
            results.append(ret) 

    if len(results) > 0:
        analysis_results(results)

def test_public_data():
    # local infer
    results = []
    exp = 'individual_public' # 'mult_test' # 'exp_ori_total6' # 'exp_ori_inter3' #'exp_ori_total6' # 'exp_ori_wh_new_type1' # 'pybullet_cylinder'# 'pybullet_intersection' # 'pybullet_ts2' # 'new_camera' # 'exp_dis_5' # 'cylinder_test' '2024_09_02__09_00_15' 'exp_dis_5'
    with_label = True
    log_dirs = LOGDIRS
    for dir in log_dirs:
        ret = test_model(os.path.join('log',dir),exp,with_label = with_label)
        if ret != None:
            del ret['loc_map']
            results.append(ret) 
    diff_path = os.path.join(ROOT,f'results/{exp}/diff')
    if not os.path.exists(diff_path):
        os.makedirs(diff_path)
    with open(os.path.join(diff_path, exp+'.json'),'w') as json_writer:
        json.dump(results,json_writer)

    # json infer
    results_dict = {'yolov8':'experiments/yolo_results/results_yolov8_pub_new_labeled3.json',
                    'pose_anything':'experiments/pose_anything_results/pose_anything_pub_new_results.json'}
    for ind_model_name in results_dict.keys():
        ret = json_infer(results_dict[ind_model_name],ind_model_name,(512,512),exp=exp,with_label=with_label,save_figs=True)
        if ret != None:
            del ret['loc_map']
            results.append(ret) 

    if len(results) > 0:
        analysis_results(results)

def test_public_data_enhance():
    # local infer
    results = []
    exp = 'individual_public_enhance' # 'mult_test' # 'exp_ori_total6' # 'exp_ori_inter3' #'exp_ori_total6' # 'exp_ori_wh_new_type1' # 'pybullet_cylinder'# 'pybullet_intersection' # 'pybullet_ts2' # 'new_camera' # 'exp_dis_5' # 'cylinder_test' '2024_09_02__09_00_15' 'exp_dis_5'
    with_label = True
    log_dirs = LOGDIRS
    for dir in log_dirs:
        ret = test_model(os.path.join('log',dir),exp,with_label = with_label)
        if ret != None:
            del ret['loc_map']
            results.append(ret) 
    diff_path = os.path.join(ROOT,f'results/{exp}/diff')
    if not os.path.exists(diff_path):
        os.makedirs(diff_path)
    with open(os.path.join(diff_path, exp+'.json'),'w') as json_writer:
        json.dump(results,json_writer)

    # json infer
    results_dict = {'yolov8':'experiments/yolo_results/results_yolov8_pub_new_offset_2510.json',
                    'pose_anything':'experiments/pose_anything_results/pose_anything_pub_new_offset_2510_results.json'}
    for ind_model_name in results_dict.keys():
        ret = json_infer(results_dict[ind_model_name],ind_model_name,(512,512),exp=exp,with_label=with_label,save_figs=True)
        if ret != None:
            del ret['loc_map']
            results.append(ret) 

    if len(results) > 0:
        analysis_results(results)

def test_fbz():
    # local infer
    results = []
    exp = 'pub_fbz' # 'mult_test' # 'exp_ori_total6' # 'exp_ori_inter3' #'exp_ori_total6' # 'exp_ori_wh_new_type1' # 'pybullet_cylinder'# 'pybullet_intersection' # 'pybullet_ts2' # 'new_camera' # 'exp_dis_5' # 'cylinder_test' '2024_09_02__09_00_15' 'exp_dis_5'
    with_label = True
    log_dirs = LOGDIRS
    for dir in log_dirs:
        ret = test_model(os.path.join('log',dir),exp,with_label = with_label)
        if ret != None:
            del ret['loc_map']
            results.append(ret) 
    diff_path = os.path.join(ROOT,f'results/{exp}/diff')
    if not os.path.exists(diff_path):
        os.makedirs(diff_path)
    with open(os.path.join(diff_path, exp+'.json'),'w') as json_writer:
        json.dump(results,json_writer)

    # json infer
    results_dict = {'yolov8':'experiments/yolo_results/results_yolov8_fbz.json',
                    'pose_anything':'experiments/pose_anything_results/pose_anything_fbz_results.json'}
    for ind_model_name in results_dict.keys():
        ret = json_infer(results_dict[ind_model_name],ind_model_name,(512,512),exp=exp,with_label=with_label,save_figs=True)
        if ret != None:
            del ret['loc_map']
            results.append(ret) 

    if len(results) > 0:
        analysis_results(results)

def test_zj2():
    # local infer
    results = []
    exp = 'zj2' # 'mult_test' # 'exp_ori_total6' # 'exp_ori_inter3' #'exp_ori_total6' # 'exp_ori_wh_new_type1' # 'pybullet_cylinder'# 'pybullet_intersection' # 'pybullet_ts2' # 'new_camera' # 'exp_dis_5' # 'cylinder_test' '2024_09_02__09_00_15' 'exp_dis_5'
    with_label = True
    log_dirs = LOGDIRS
    for dir in log_dirs:
        ret = test_model(os.path.join('log',dir),exp,with_label = with_label)
        if ret != None:
            del ret['loc_map']
            results.append(ret) 
    diff_path = os.path.join(ROOT,f'results/{exp}/diff')
    if not os.path.exists(diff_path):
        os.makedirs(diff_path)
    with open(os.path.join(diff_path, exp+'.json'),'w') as json_writer:
        json.dump(results,json_writer)

    # json infer
    results_dict = {'yolov8':'experiments/yolo_results/results_yolov8_zj2.json',
                    'pose_anything':'experiments/pose_anything_results/pose_anything_zj2_results.json'}
    for ind_model_name in results_dict.keys():
        ret = json_infer(results_dict[ind_model_name],ind_model_name,(512,512),exp=exp,with_label=with_label,save_figs=True)
        if ret != None:
            del ret['loc_map']
            results.append(ret) 

    if len(results) > 0:
        analysis_results(results)

def test_mult():
    # local infer
    results = []
    exp = 'mult_test' # 'mult_test' # 'exp_ori_total6' # 'exp_ori_inter3' #'exp_ori_total6' # 'exp_ori_wh_new_type1' # 'pybullet_cylinder'# 'pybullet_intersection' # 'pybullet_ts2' # 'new_camera' # 'exp_dis_5' # 'cylinder_test' '2024_09_02__09_00_15' 'exp_dis_5'
    with_label = True
    log_dirs = LOGDIRS
    for dir in log_dirs:
        ret = test_model(os.path.join('log',dir),exp,with_label = with_label)
        if ret != None:
            del ret['loc_map']
            results.append(ret) 
    diff_path = os.path.join(ROOT,f'results/{exp}/diff')
    if not os.path.exists(diff_path):
        os.makedirs(diff_path)
    with open(os.path.join(diff_path, exp+'.json'),'w') as json_writer:
        json.dump(results,json_writer)

    # json infer
    results_dict = {'yolov8':'experiments/yolo_results/results_mult.json',
                    'pose_anything':'experiments/pose_anything_results/pose_anything_mult_results.json'}
    for ind_model_name in results_dict.keys():
        ret = json_infer(results_dict[ind_model_name],ind_model_name,(512,512),exp=exp,with_label=with_label,save_figs=True)
        if ret != None:
            del ret['loc_map']
            results.append(ret) 

    if len(results) > 0:
        analysis_results(results)


def analysis_results(ret_results, save_path='latex_table.txt'):
    TABLE = [
        "model", "time", "Gflops", "f1score", "topk_1", "topk_2",
        "mean", "std", "var", "max", "min", "aa_1", "aa_2",
        "angle_mean", "angle_std", "angle_var", "angle_max", "angle_min"
    ]

    tb = PrettyTable(TABLE)

    # 
    rows = []
    for row in ret_results:
        tb.add_row(row['results'])
        rows.append(row['results'])

    #  best 
    best = ['best']
    model_name_list = [value['results'][0] for value in ret_results]
    for i in range(1, len(TABLE)):
        values = np.array([value['results'][i] for value in ret_results])
        # （ f1score、topk、aa），（ mae、std、angle_mean）
        # ：f1score, topk_1, topk_2, aa_1, aa_2
        if TABLE[i] in ['f1score', 'topk_1', 'topk_2', 'aa_1', 'aa_2']:
            idx = np.argmax(values)
        else:
            idx = np.argmin(values)
        best.append(model_name_list[idx])
    tb.add_row(best)

    #  PrettyTable
    print(tb)

# LaTeX 
    latex_lines = []
    header = r"""\begin{table*}[h]
        \\renewcommand\cellgape{\\Gape[5pt]}
        \\caption{Results of Comparison}\\label{table:offset_cmp}
        \\centering
        \\setlength{\\tabcolsep}{5pt}
        \\begin{tabularx}{\textwidth}{@{\\extracolsep{\\fill}}l|@{}l|ccccccccccccccc}
        \\hline
        \\multicolumn{2}{l|}{Model}& F1 & MAE & SD & Max & MAE & SD & Max & $\\text{PCK@1}$ & $\\text{PCK@2}$ & $\\text{PCA@1}$ & $\\text{PCA@2}$ & Time & GFLOPs\\
        \\multicolumn{2}{l|}{ } &  & (pixel) & (pixel) & (pixel)  & ($^\\circ$) & ($^\\circ$) & ($^\\circ$) & (\%) &  (\%) &  (\%) &  (\%) & (ms) &\\
        \\hline"""
    latex_lines.append(header)

    #  FIELD_NAMES 
    idx_map = {
        'F1': 3,
        'MAE_px': 6,
        'SD_px': 7,
        'Max_px': 9,
        'MAE_deg': 13,
        'SD_deg': 14,
        'Max_deg': 16,
        'PCK@1': 4,
        'PCK@2': 5,
        'PCA@1': 11,
        'PCA@2': 12,
        'Time': 1,
        'GFLOPs': 2
    }

    # （ best）
    for row in ret_results:
        res = row['results']
        tab_sub_head = '\\multirow{4}{*}{\\rotatebox{90}{Proposed}}' if 'Proposed' in res[0] else '\\multicolumn{{2}}{{l|}}{{{}}}'.format(res[0])
        line = f"{tab_sub_head} & "
        line += f"${res[idx_map['F1']]:.3f}$ & ${res[idx_map['MAE_px']]:.2f}$ & ${res[idx_map['SD_px']]:.2f}$ & ${res[idx_map['Max_px']]:.2f}$ & "
        line += f"${res[idx_map['MAE_deg']]:.2f}$ & ${res[idx_map['SD_deg']]:.2f}$ & ${res[idx_map['Max_deg']]:.2f}$ & "
        line += f"${res[idx_map['PCK@1']]*100:.2f}$ & ${res[idx_map['PCK@2']]*100:.2f}$ & ${res[idx_map['PCA@1']]*100:.2f}$ & ${res[idx_map['PCA@2']]*100:.2f}$ & "
        line += f"${res[idx_map['Time']]*1000:.2f}$ & {res[idx_map['GFLOPs']]:.1f} \\\\"
        latex_lines.append(line)

    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabularx}")
    latex_lines.append(r"\end{table*}")

    # 
    with open(save_path, 'w') as f:
        for line in latex_lines:
            f.write(line + '\n')


    print(f"LaTeX table saved to {save_path}")


def calc_flops(device,net,input_shape): # ,input_shape=(1,1,600,1088)):
    # input_shape = (1,3,512,512)
    # net.select_to_device(current,[0],[0],'cpu')
    d = torch.rand(*input_shape).to(device)
    flops, params = profile(net, (d,))
    print("model flops:{} GFLOPs".format(flops/1e9))
    return flops/1000000000

if __name__=="__main__":


    test()
    # test_trans_enhance()
    # test_public_data()
    # test_public_data_enhance()
    test_fbz()
    test_zj2()
    test_mult()



