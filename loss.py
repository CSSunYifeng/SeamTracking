import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Dict
import numpy as np
from models.utils import *
from scipy.io import savemat,loadmat
import os 
ROOT = os.getenv('SEAMTRACKING_ROOT')
from scipy.spatial import KDTree

'''

'''
def get_pos_from_hm(hm:torch.Tensor):
    batch,cat,height,width = hm.size()
    hm = hm.view(batch,-1)
    _,indices = torch.topk(hm,1,dim=-1)#indices:N*1
    labels = torch.div(indices,(height*width),rounding_mode='trunc')
    indices = indices%(height*width)#indices:N*1
    pos_pred_x = indices%width#pos_pred_x:N*1
    pos_pred_y = torch.div(indices,width,rounding_mode='trunc')#pos_pred_y:N*1
    return labels,pos_pred_x,pos_pred_y

def get_threshold_from_hm_gt(hm_gt:torch.Tensor,item_num):
    scores,_ = torch.topk(hm_gt.view(size=(hm_gt.shape[0],-1)),k=10*item_num,dim=-1)
    return scores.min()

def _neg_loss_mult(pred, gt, item_num,opt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    # if opt['iter'] == 1:
    # _heat_nms(gt)

    thresh_hold = get_threshold_from_hm_gt(gt, 1) # ：
    pos_inds = gt.ge(thresh_hold).float() # ：pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(thresh_hold).float() # ：neg_inds = gt.lt(1).float()
    # pos_inds = gt.eq(1).float() # ：pos_inds = gt.eq(1).float()
    # neg_inds = gt.lt(1).float() # ：neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)# torch.pow(1 - gt, 4)

    loss = 0
    pred = torch.clamp(pred, 0.0001, 0.9999)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds # pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds # neg_inds

    num_pos  = pos_inds.float().sum() # 
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    #print('num_pos=',num_pos)
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    # _loss = loss.detach().clone().cpu().numpy() # tensor.detach().numpy()
    # has_nan = np.isnan(np.array(_loss)).any()
    # if has_nan==True:
    #     torch.save(pred, 'debug_pred.pt')
    #     torch.save(neg_weights, 'debug_neg_weights.pt')
    #     torch.save(neg_inds, 'debug_neg_inds.pt')
    #     torch.save(pos_inds, 'debug_pos_inds.pt')
    #     torch.save(pos_loss, 'debug_pos_loss.pt')
    #     torch.save(neg_loss, 'debug_neg_loss.pt')
    # assert not has_nan
    return loss

class HeatmapLoss_mult(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.opt = opt
        if "heatmap_sigma" in self.opt:
            self.std = self.opt["heatmap_sigma"] # 0.05
        else:
            self.std = 0.05
        print("heatmap_sigma:{}".format(self.std))
    def forward(self,hm,pos_gt): #
        '''
        hm:(N,C,W,H)
        pos_gt:(N,2)
        labels:(N)
        '''

        hm_gt = torch.zeros_like(hm)#hm_gt:(N,C,W,H)
        batch_size,class_cnt,rows,cols = hm.shape
        item_num = 0
        for i in range(batch_size):
            r = torch.linspace(0,1,rows)
            c = torch.linspace(0,1,cols)
            R,C = torch.meshgrid(r,c,indexing='ij')
            R = R.to(hm.device)
            C = C.to(hm.device)
            for item in pos_gt:
                label = item['label']
                item_num += len(item['poses'])
                p = item['cp']
                # for p in item['poses']:
                ret = torch.exp(-((R-p[i][1])**2+(C-p[i][0])**2)/(2*self.std**2))
                hm_gt[i][label[i]] = torch.max(hm_gt[i][label[i]] ,ret) if label[i]>=0 else 0
        return _neg_loss_mult(hm,hm_gt,item_num=3,opt=self.opt)

class OffsetLoss_mult(nn.Module):
    def __init__(self,smooth=False):
        super().__init__()
        self.smooth = smooth

    # def forward_mult(self,hm,off,batch):
    #     batch_size,cat,height,width = hm.size()
    #     # labels,pos_pred_x,pos_pred_y = get_pos_from_hm(hm)
    #     ret = 0
    #     N = 0 # len(item['poses'])
    #     for item in batch:
    #         N += len(item['poses'])
    #         for p in item['poses']:
    #             D = torch.transpose(p,1,0)
    #             pos_pred_x = D[0].unsqueeze(-1)
    #             pos_pred_y = D[1].unsqueeze(-1)
    #             # for i in range(batch_size):
    #             #     for item in batch:
    #             #         pos_pred_x = np.append(pos_pred_x,p[i][0])
    #             #         pos_pred_y = np.append(pos_pred_y,p[i][1])
    #             # pos_pred_x,pos_pred_y,width,height = batch
    #             indices = pos_pred_y*width+pos_pred_x
    #             pos_pred_x = pos_pred_x/width
    #             pos_pred_y = pos_pred_y/height
    #             pos_pred = torch.cat([pos_pred_x,pos_pred_y],dim=1)#pos_pred:N*2 
    #             indices = indices.unsqueeze(dim=-1).expand(batch_size,2,1)#indices:N*2*1
    #             indices = torch.floor(indices).to(torch.int64)
    #             off = off.view(batch_size,2,-1)#off:N*2*(width*height)  
    #             off_val = torch.gather(off,-1,indices)#off_val:N*2*1
    #             off_val = off_val.squeeze(dim=-1)#off_val:N*2
    #             ret += F.l1_loss(off_val+pos_pred,p)
    #     assert N > 0
    #     return ret/N
    # get_pos_from_hm

    def forward_mult_max(self,hm,off,batch):
        batch_size,cat,height,width = hm.size()
        # labels,pos_pred_x,pos_pred_y = get_pos_from_hm(hm)
        ret = 0
        # N = 0 # len(item['poses'])
        _,pos_pred_x,pos_pred_y = get_pos_from_hm(hm)
        # p_best = torch.cat([x,y],dim=1)
        # N += len(item['poses'])
        # D = torch.transpose(p_best,1,0)
        # pos_pred_x = D[0].unsqueeze(-1)
        # pos_pred_y = D[1].unsqueeze(-1)
        indices = torch.floor(pos_pred_y)*width+torch.floor(pos_pred_x)
        # pos_pred_x = pos_pred_x/width
        # pos_pred_y = pos_pred_y/height

        # # external = torch.zeros_like(pos_pred_y)
        # pos_pred = torch.cat([pos_pred_x,pos_pred_y,pos_pred_x,pos_pred_y,pos_pred_x,pos_pred_y],dim=1)#pos_pred:N*2 
        # external_p = torch.zeros_like(item['poses'][0])
        p_mid = [value for value in batch[0]['poses']]
        p = torch.cat(p_mid,dim=1)
        cpx = torch.floor(pos_pred_x*width)/width
        cpy = torch.floor(pos_pred_y*height)/height
        cp = torch.tile(torch.cat([cpx,cpy],dim=1),(1,2)) # trans (batch_size,2) to (batch_size,6) by repeating 3 times in dim 1
        # dp = torch.tensor(np.array([[item['cp'][0]*width,item['cp'][1]*height]*3]))
        # cp[:,0:5:2] = torch.floor(cp[:,0:5:2])/width
        # cp[:,1:6:2] = torch.floor(cp[:,1:6:2])/height 

        # for p in item['poses']:
            # p.append(item['poses'][i])

            # for i in range(batch_size):
            #     for item in batch:
            #         pos_pred_x = np.append(pos_pred_x,p[i][0])
            #         pos_pred_y = np.append(pos_pred_y,p[i][1])
            # pos_pred_x,pos_pred_y,width,height = batch

        indices = indices.unsqueeze(dim=-1).expand(batch_size,4,1)#indices:N*2*1
        indices = torch.floor(indices).to(torch.int64)
        off = off[:,:4,:,:].view(batch_size,4,-1)#off:N*2*(width*height)  
        off_val = torch.gather(off,-1,indices)#off_val:N*2*1
        off_val = off_val.squeeze(dim=-1)#off_val:N*2
        # ss = p-(torch.floor(cp*16)/16)
        # _heat_nms(p)
        ret += F.l1_loss(off_val,p-cp)# off_val,pos_gt-torch.floor(pos_gt))
        # ret = ret/N if N>0 else 0
        return ret
    

    def forward_mult_2(self,hm,off,batch):
        batch_size,cat,height,width = hm.size()
        # labels,pos_pred_x,pos_pred_y = get_pos_from_hm(hm)
        ret = 0
        # N = 0 # len(item['poses'])
        for item in batch:
            # N += len(item['poses'])
            D = torch.transpose(item['cp'],1,0)
            pos_pred_x = D[0].unsqueeze(-1)
            pos_pred_y = D[1].unsqueeze(-1)
            indices = torch.floor(pos_pred_y*height)*width+torch.floor(pos_pred_x*width)
            # pos_pred_x = pos_pred_x/width
            # pos_pred_y = pos_pred_y/height

            # # external = torch.zeros_like(pos_pred_y)
            # pos_pred = torch.cat([pos_pred_x,pos_pred_y,pos_pred_x,pos_pred_y,pos_pred_x,pos_pred_y],dim=1)#pos_pred:N*2 
            # external_p = torch.zeros_like(item['poses'][0])
            p = torch.cat([value for value in item['poses']],dim=1)
            cpx = torch.floor(item['cp'][:,0]*width)/width
            cpy = torch.floor(item['cp'][:,1]*height)/height
            cp = torch.tile(torch.cat([cpx.unsqueeze(-1),cpy.unsqueeze(-1)],dim=1),(1,2)) # trans (batch_size,2) to (batch_size,6) by repeating 3 times in dim 1
            # dp = torch.tensor(np.array([[item['cp'][0]*width,item['cp'][1]*height]*3]))
            # cp[:,0:5:2] = torch.floor(cp[:,0:5:2])/width
            # cp[:,1:6:2] = torch.floor(cp[:,1:6:2])/height 

            # for p in item['poses']:
                # p.append(item['poses'][i])

                # for i in range(batch_size):
                #     for item in batch:
                #         pos_pred_x = np.append(pos_pred_x,p[i][0])
                #         pos_pred_y = np.append(pos_pred_y,p[i][1])
                # pos_pred_x,pos_pred_y,width,height = batch

            indices = indices.unsqueeze(dim=-1).expand(batch_size,4,1)#indices:N*2*1
            indices = torch.floor(indices).to(torch.int64)
            off = off[:,:4,:,:].view(batch_size,4,-1)#off:N*2*(width*height)  
            off_val = torch.gather(off,-1,indices)#off_val:N*2*1
            off_val = off_val.squeeze(dim=-1)#off_val:N*2
            # ss = p-(torch.floor(cp*16)/16)
            # _heat_nms(p)
            ret += F.l1_loss(off_val,p-cp)# off_val,pos_gt-torch.floor(pos_gt))
        # ret = ret/N if N>0 else 0
        return ret
    
    # def forward_mult_3(self,hm,off,batch):
    #     batch_size,cat,height,width = hm.size()
    #     # labels,pos_pred_x,pos_pred_y = get_pos_from_hm(hm)
    #     ret = 0
    #     N = 0 # len(item['poses'])
    #     hm_points = _heat_nms(hm)
    #     cp_list = [item['cp'] for item in batch]
    #     hmp_list = [[hm_p[1],hm_p[2]] for hm_p in hm_points]
    #     kd_tree = KDTree(hmp_list)
    #     _,indcs = kd_tree.query(cp_list)

    #     for item in batch:
    #         N += len(item['poses'])
    #         D = torch.transpose(item['cp'],1,0)
    #         pos_pred_x = D[0].unsqueeze(-1)
    #         pos_pred_y = D[1].unsqueeze(-1)
    #         indices = torch.floor(pos_pred_y*height)*width+torch.floor(pos_pred_x*width)
    #         # pos_pred_x = pos_pred_x/width
    #         # pos_pred_y = pos_pred_y/height

    #         # # external = torch.zeros_like(pos_pred_y)
    #         # pos_pred = torch.cat([pos_pred_x,pos_pred_y,pos_pred_x,pos_pred_y,pos_pred_x,pos_pred_y],dim=1)#pos_pred:N*2 
    #         # external_p = torch.zeros_like(item['poses'][0])
    #         p = torch.cat([value for value in item['poses']],dim=1)
    #         cpx = torch.floor(item['cp'][:,0]*width)/width
    #         cpy = torch.floor(item['cp'][:,1]*height)/height
    #         cp = torch.tile(torch.cat([cpx.unsqueeze(-1),cpy.unsqueeze(-1)],dim=1),(1,3)) # trans (batch_size,2) to (batch_size,6) by repeating 3 times in dim 1
    #         # dp = torch.tensor(np.array([[item['cp'][0]*width,item['cp'][1]*height]*3]))
    #         # cp[:,0:5:2] = torch.floor(cp[:,0:5:2])/width
    #         # cp[:,1:6:2] = torch.floor(cp[:,1:6:2])/height 

    #         # for p in item['poses']:
    #             # p.append(item['poses'][i])

    #             # for i in range(batch_size):
    #             #     for item in batch:
    #             #         pos_pred_x = np.append(pos_pred_x,p[i][0])
    #             #         pos_pred_y = np.append(pos_pred_y,p[i][1])
    #             # pos_pred_x,pos_pred_y,width,height = batch

    #         indices = indices.unsqueeze(dim=-1).expand(batch_size,6,1)#indices:N*2*1
    #         indices = torch.floor(indices).to(torch.int64)
    #         off = off[:,:6,:,:].view(batch_size,6,-1)#off:N*2*(width*height)  
    #         off_val = torch.gather(off,-1,indices)#off_val:N*2*1
    #         off_val = off_val.squeeze(dim=-1)#off_val:N*2
    #         # ss = p-(torch.floor(cp*16)/16)
    #         # _heat_nms(p)
    #         ret += F.l1_loss(off_val,p-cp)# off_val,pos_gt-torch.floor(pos_gt))
    #     assert N > 0
    #     return ret/N
    
    # def forward_mult_sim_single(self,hm,off,batch):
    #     batch_size,cat,height,width = hm.size()
    #     # labels,pos_pred_x,pos_pred_y = get_pos_from_hm(hm)
    #     item = batch[0]
    #     pos_gt=batch[0]['cp']
    #     labels,pos_pred_x,pos_pred_y = get_pos_from_hm(hm)
    #     indices = pos_pred_y*width+pos_pred_x
    #     pos_pred_x = pos_pred_x/width
    #     pos_pred_y = pos_pred_y/height
    #     pos_pred = torch.cat([pos_pred_x,pos_pred_y],dim=1)#pos_pred:N*2 
    #     indices = indices.unsqueeze(dim=-1).expand(batch_size,2,1)#indices:N*2*1
    #     off = off.view(batch_size,2,-1)#off:N*2*(width*height)  
    #     off_val = torch.gather(off,-1,indices)#off_val:N*2*1
    #     off_val = off_val.squeeze(dim=-1)#off_val:N*2
    #     return F.l1_loss(off_val,pos_gt-torch.floor(pos_gt))

    def forward(self,hm,off,pos_gt):
        return self.forward_mult_2(hm,off,pos_gt)

class OriLoss_mult(nn.Module):
    def __init__(self,smooth=False):
        super().__init__()
        self.smooth = smooth
    

    def forward_mult(self,hm,ori,batch):
        batch_size,cat,height,width = hm.size()
        # labels,pos_pred_x,pos_pred_y = get_pos_from_hm(hm)
        ret = 0
        # N = 0 # len(item['poses'])
        for item in batch:
            # N += len(item['poses'])
            D = torch.transpose(item['cp'],1,0)
            pos_pred_x = D[0].unsqueeze(-1)
            pos_pred_y = D[1].unsqueeze(-1)
            indices = torch.floor(pos_pred_y*height)*width+torch.floor(pos_pred_x*width)

            o = torch.cat([value for value in item['ori']],dim=1)

            indices = indices.unsqueeze(dim=-1).expand(batch_size,4,1)#indices:N*2*1
            indices = torch.floor(indices).to(torch.int64)
            ori = ori[:,:4,:,:].view(batch_size,4,-1)#off:N*2*(width*height)  
            val = torch.gather(ori,-1,indices)#off_val:N*2*1
            val = val.squeeze(dim=-1)#off_val:N*2
            # ss = p-(torch.floor(cp*16)/16)
            # _heat_nms(p)
            ret += F.l1_loss(val,o)# off_val,pos_gt-torch.floor(pos_gt))
        # ret = ret/N if N>0 else 0
        return ret

    def forward(self,hm,ori,pos_gt):
        return self.forward_mult(hm,ori,pos_gt)

class MyLoss_ori_mult(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.dis_loss_func = None
        if "loss_alpha" in opt and "loss_beta" in opt:
            self.alpha = opt["loss_alpha"]
            self.beta = opt["loss_beta"]
            self.theta = opt["loss_theta"]
        else:
            self.alpha = 1
            self.beta = 1
        self.hm_loss_func = HeatmapLoss_mult(opt)
        self.off_loss_func = OffsetLoss_mult(opt['smooth_loss'])
        self.ori_loss_func = OriLoss_mult(opt['smooth_loss'])
    def forward_common(self,hm,off,ori,pos_gt)->Dict[str,torch.tensor]:
        hm_loss = self.hm_loss_func(hm,pos_gt)
        off_loss = self.off_loss_func(hm,off,pos_gt)
        ori_loss = self.ori_loss_func(hm,ori,pos_gt)
        # print("hm_loss:{}, off_loss:{}".format(hm_loss,off_loss))
        return {
            'total_loss':self.alpha*hm_loss+self.beta*off_loss+self.theta*ori_loss,
            'hm_loss':hm_loss,
            'off_loss':off_loss,
            'ori_loss':ori_loss
        }
    def forward(self,hm,off,ori,pos_gt)->Dict[str,torch.tensor]:
        return self.forward_common(hm,off,ori,pos_gt)


def calc_metrics(hm,off,pos_gt,labels_gt,opt)->Dict[str,float]:
    batch, cat, height, width = hm.size()
    labels,x,y = get_pos_from_hm(hm)
    labels = labels.squeeze(dim=-1)
    
    cls_acc = torch.sum(labels==labels_gt)/batch*100

    inds = y*width+x
    out = torch.gather(off.view(batch,2,-1),-1,inds.expand(batch,2).unsqueeze(dim=2))
    out = out.squeeze(dim=-1)#N*2
    off_x = out[:,0:1]
    off_y = out[:,1:]

    x = x/width + off_x
    y = y/height + off_y

    l1_loss_x = F.l1_loss(x,pos_gt[:,0:1])#(N,1)
    l1_loss_y = F.l1_loss(y,pos_gt[:,1:])#(N,1)

    mae_x_um = l1_loss_x*opt["input_width"]*(10000/opt["input_width"])#1088)
    mae_y_um = l1_loss_y*opt["input_height"]*(10000/opt["input_height"]) #1088)

    mae_distance = (mae_x_um**2+mae_y_um**2)**(1/2)

    valid_dis_ratio = ((mae_distance<=100).sum()/mae_distance.numel()).item()*100



    return {
        'cls_acc':cls_acc.item(),
        'labels_pred_list':labels.cpu().detach().numpy(),
        'labels_gt_list':labels_gt.cpu().detach().numpy(),
        'l1_loss_x':l1_loss_x.item(),
        'l1_loss_y':l1_loss_y.item(),
        'l1_loss_x_list':(x-pos_gt[:,0:1]).abs().cpu().detach().numpy(),
        'l1_loss_y_list':(y-pos_gt[:,1:]).abs().cpu().detach().numpy(),
        'valid_dis_raio':valid_dis_ratio,
    }

def calc_metrics_mult(hm,off,pos_gt,labels_gt,opt)->Dict[str,float]:
    batch, cat, height, width = hm.size()
    labels,x,y = get_pos_from_hm(hm)
    labels = labels.squeeze(dim=-1)
    
    cls_acc = torch.sum(labels==labels_gt)/batch*100

    inds = y*width+x
    out = torch.gather(off.view(batch,2,-1),-1,inds.expand(batch,2).unsqueeze(dim=2))
    out = out.squeeze(dim=-1)#N*2
    off_x = out[:,0:1]
    off_y = out[:,1:]

    x = x/width + off_x
    y = y/height + off_y

    l1_loss_x = F.l1_loss(x,pos_gt[:,0:1])#(N,1)
    l1_loss_y = F.l1_loss(y,pos_gt[:,1:])#(N,1)

    mae_x_um = l1_loss_x*opt["input_width"]*(10000/opt["input_width"])#1088)
    mae_y_um = l1_loss_y*opt["input_height"]*(10000/opt["input_height"]) #1088)

    mae_distance = (mae_x_um**2+mae_y_um**2)**(1/2)

    valid_dis_ratio = ((mae_distance<=100).sum()/mae_distance.numel()).item()*100



    return {
        'cls_acc':cls_acc.item(),
        'labels_pred_list':labels.cpu().detach().numpy(),
        'labels_gt_list':labels_gt.cpu().detach().numpy(),
        'l1_loss_x':l1_loss_x.item(),
        'l1_loss_y':l1_loss_y.item(),
        'l1_loss_x_list':(x-pos_gt[:,0:1]).abs().cpu().detach().numpy(),
        'l1_loss_y_list':(y-pos_gt[:,1:]).abs().cpu().detach().numpy(),
        'valid_dis_raio':valid_dis_ratio,
    }

def calc_metrics_DDP(hm,off,pos_gt,labels_gt,opt)->Dict[str,float]:
    batch, cat, height, width = hm.size()
    labels,x,y = get_pos_from_hm(hm)
    labels = labels.squeeze(dim=-1)
    
    cls_acc = torch.sum(labels==labels_gt)/batch*100

    inds = y*width+x
    out = torch.gather(off.view(batch,2,-1),-1,inds.expand(batch,2).unsqueeze(dim=2))
    out = out.squeeze(dim=-1)#N*2
    off_x = out[:,0:1]
    off_y = out[:,1:]

    x = x/width + off_x
    y = y/height + off_y

    l1_loss_x = F.l1_loss(x,pos_gt[:,0:1])#(N,1)
    l1_loss_y = F.l1_loss(y,pos_gt[:,1:])#(N,1)

    mae_x_um = l1_loss_x*opt["input_width"]*(10000/opt["input_width"])#1088)
    mae_y_um = l1_loss_y*opt["input_height"]*(10000/opt["input_height"]) #1088)

    mae_distance = (mae_x_um**2+mae_y_um**2)**(1/2)

    valid_dis_ratio = ((mae_distance<=100).sum()/mae_distance.numel()) # .item()*100



    return {
        'cls_acc':cls_acc,#.item(),
        'labels_pred_list':labels,# .cpu().detach().numpy(),
        'labels_gt_list':labels_gt,#.cpu().detach().numpy(),
        'l1_loss_x':l1_loss_x,#.item(),
        'l1_loss_y':l1_loss_y,#.item(),
        # 'l1_loss_x_list':(x-pos_gt[:,0:1]).abs(),#.cpu().detach().numpy(),
        # 'l1_loss_y_list':(y-pos_gt[:,1:]).abs(),#.cpu().detach().numpy(),
        'valid_dis_raio':valid_dis_ratio,
    }
