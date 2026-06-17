from loss import *
from typing import Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SubsetRandomSampler
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
import utils.device as memory_device
import random 
import torch
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from torch.cuda.amp import autocast,GradScaler
from utils.dataset import dataloader_ori,split
import json
from thop import profile
from tqdm import tqdm
from factory import generateModel
import task_list as tl

TRAIN_RATIO = 0.7 
VAL_RATIO = 0.1
SCALE_FACTOR = 128.0 #  
ROOT = os.getenv('SEAMTRACKING_ROOT')

def get_confusion_matrix_figure(pred,labels,label_choose,cls_names,figsize):
    num_class = len(cls_names)
    cmtx = confusion_matrix(labels,pred,labels=label_choose)
    figure = plt.figure(figsize=figsize)
    plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(num_class)
    plt.xticks(tick_marks, cls_names, rotation=45)
    plt.yticks(tick_marks, cls_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] > threshold else "black"
        plt.text(j,i,format(cmtx[i, j], ".2f") if cmtx[i, j] != 0 else ".",
            horizontalalignment="center",
            color=color,
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure

def load_config():
    c = open('config.yaml','r')
    return yaml.load(c)


def train_model(epoch_idx:int,
                device:str,
                train_dl:DataLoader,
                net:nn.Module,
                loss_fn:Callable,
                optimizer:optim.Optimizer,
                writer:SummaryWriter=None,
                scaler:GradScaler=None):
    net.train()
    batches_cnt:int = len(train_dl)

    losses = {}
    # z = time.time()

    for items in tqdm(train_dl):
        im,item_gt,_ = items
        # _heat_nms(im)
        im = im.to(device)
        for item in item_gt:
            item['cp'] = item['cp'].to(device)
            # print(device)
            item['label'] = item['label'].to(device)
            # print(item['label'].device)
            item['id'] = item['id'].to(device)
            item['poses'] = [value.to(device) for value in item['poses']]
            item['ori'] = [value.to(device) for value in item['ori']]
        # item_gt = [value.to(device) for values in item_gt for value in values.values()]
        # labels_gt = labels_gt.to(device)
        # print(item['label'].device)
        loss_dict = None
        optimizer.zero_grad()
        # print("time 1:{}".format(b-a))
        # c = time.time()
        if scaler != None:
            with autocast():
                hm,off,ori = net(im) # forward
                if opt['half'] == True:
                    hm = hm.sigmoid().to(torch.float32) # ，，float32，float16，focal losslog
                # pred = torch.sigmoid(hm).type(torch.float32)# ，，float32，float16，focal losslog
                # hm = hm.sigmoid().type(torch.float32)
                loss_dict = loss_fn(hm,off,ori,item_gt)#off,pos_gt,labels_gt)
            scaler.scale(loss_dict['total_loss']).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            hm,off,ori = net(im) # forward
            # _heat_nms(off)
            hm = hm.sigmoid()
            loss_dict = loss_fn(hm,off,ori,item_gt)#off,pos_gt,labels_gt)            
            loss_dict['total_loss'].backward()
            optimizer.step()

        for k,v in loss_dict.items():
            losses[k] = losses.get(k,0)+v.item() 
        # d = time.time()
        # print("time2:{}".format(d-c))
        # e = time.time()
        if scaler == None:
            net.zero_grad()
        # print("time3:{}".format(f-e))
    # y = time.time()
    # print("time part1:{}".format(y-z))
    # s = time.time()


    for k in losses.keys():
        losses[k] /= batches_cnt
        if writer is not None:
            writer.add_scalar(k+'/train',losses[k],epoch_idx)
    # q = time.time()
    # print("time part2:{}".format(q-s))

def val_model(epoch_idx:int,device:str,val_dl:DataLoader,net:nn.Module,loss_fn:Callable,writer:SummaryWriter=None,opt=None):
    net.eval()

    losses = {}
    matrics = {}
    
    batch_cnt = len(val_dl)

    with torch.no_grad():
        for items in tqdm(val_dl):
            # im,pos_gt,labels_gt = item
            im,item_gt,_ = items
            im = im.to(device)
            for item in item_gt:
                item['cp'] = item['cp'].to(device)
                item['label'] = item['label'].to(device)
                item['id'] = item['id'].to(device)
                item['poses'] = [value.to(device) for value in item['poses']]
                item['ori'] = [value.to(device) for value in item['ori']]
            # labels_gt = labels_gt.to(device)

            hm,off,ori = net(im)
            hm = hm.sigmoid()
            # matrics_dict = calc_metrics_mult(hm,off,item['poses'],item['label'],opt)
            losses_dict = loss_fn(hm,off,ori,item_gt)#off,pos_gt,labels_gt)

            # for k,v in matrics_dict.items():
            #     if not k.endswith('list'):
            #         matrics[k] = matrics.get(k,0)+v
            #     else:
            #         matrics[k] = matrics.get(k,[])+[v]
            for k,v in losses_dict.items():
                losses[k] = losses.get(k,0)+v.item()
    for k in losses.keys():
        losses[k] /= batch_cnt
        if writer is not None:
            writer.add_scalar(k+'/val',losses[k],epoch_idx)
    for k in matrics.keys():
        if not k.endswith('list'):
            matrics[k] /= batch_cnt
            if writer is not None:
                writer.add_scalar(k+'/val',matrics[k],epoch_idx)
        else:
            matrics[k] = np.concatenate(matrics[k])
            if k.startswith('l1_loss'):
                writer.add_histogram(k,matrics[k],epoch_idx)

    avg_total_loss = losses['total_loss']
    # avg_valid_dis_ratio = matrics['valid_dis_raio']


    return avg_total_loss,1 # avg_valid_dis_ratio

def save_checkpoints(epoch_idx:int,
                    model:nn.Module,
                    optimizer:torch.optim.Optimizer,
                    val_loss:float,
                    valid_dis_ratio:float,
                    check_new_best:bool,
                    train_set,
                    val_set,
                    test_set,
                    ):
    info = {
        'epoch_idx':epoch_idx,
        'model_class_name':model.__class__.__name__,
        'model_state_dict':model.state_dict(),
        'optimizer_class_name':optimizer.__class__.__name__,
        'optimizer_state_dict':optimizer.state_dict(),
        'val_loss':val_loss,
        'valid_dis_ratio':valid_dis_ratio,
        'train_set_state_dict':train_set.state_dict(),
        'val_set_state_dict':val_set.state_dict(),
        'test_set_state_dict':test_set.state_dict(),
    }
    if check_new_best:
        if os.path.exists('best.pth'):
            old_info = torch.load('best.pth')
            old_value = (old_info['valid_dis_ratio'],-old_info['val_loss'])
            print('old_value:',old_value)
            print('cur_value:',(valid_dis_ratio,-val_loss))
            if old_value>(valid_dis_ratio,-val_loss):
                return 
            else:
                os.remove('best.pth')
                print('save model with valid_dis_ratio={} and val_loss={}'.format(valid_dis_ratio,val_loss))
                torch.save(info,'best.pth')
        else:
            print('save model with valid_dis_ratio={} and val_loss={}'.format(valid_dis_ratio,val_loss))
            torch.save(info,'best.pth')
    else:
        torch.save(info,'checkpoint.pth')

def train(opt):
    print(opt)
    task_version = opt["task_version"]# 'test_yubodataset_1'
    train_task = '_version_'+str(task_version)
    comment = 'centernet_train'+train_task #  ./log 
    print("task_version:{0}".format(train_task))
    random.seed(123456)
    torch.manual_seed(123456)
    np.random.seed(123456)

    use_gpu = True
    if opt["device"] == "auto":
        if use_gpu:
            device = memory_device.get_available_gpu_blocked(1024*7)
            if device is None:
                print('no available gpu')
                exit()
        else:
            device = 'cpu'
    else:
        device = opt["device"]

    if opt['copy'] != None:
        # pretrain_net = generateModel({'model':opt['copy']['src_model']}).to(device)
        pretrain_net_state_dict = torch.load(opt['copy']['src_model_state_dict_path'],map_location=torch.device(device))
        # pretrain_net.load_state_dict(pretrain_net_state_dict['model_state_dict'])

    net = generateModel(opt)
    optimizer = optim.AdamW(net.parameters(),lr=(1e-3)*opt['train_batch_size']/16)
    # optimizer = optim.SGD(net.parameters(), lr=(1e-3), momentum=0.9, weight_decay=0.0001)
    scaler=None

    # ，，nvidia-sim

    net.to(device) # 


    writer = None

    if opt['check_point'] == None:
        working_dir = './log/'+str(datetime.now()).replace(':',' ')+' centernet_'+train_task
        if opt['debug'] == False:
            os.mkdir(working_dir)
            with open(os.path.join(working_dir,'opt.json'),'w') as opt_json_file:
                json.dump(opt,opt_json_file)
            os.chdir(working_dir)
        head = 1
        epoches = opt["epoches"]# 500
        tail = epoches+1
        writer = SummaryWriter() # tensorboard
        writer.add_text('comment',comment)

    else:
        assert opt['debug'] == False
        checkpoint_epoch = opt["check_point"]["start_epoch"]
        working_dir = opt["check_point"]["log_dir"]
        os.chdir(working_dir)
        checkpoint = torch.load(working_dir + '/checkpoint.pth',map_location=torch.device(device))
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        net.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        head = checkpoint['epoch_idx']+1
        tail = opt["check_point"]["max_epoch"]+1

        def get_all_files(directory):
            file_list = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_list.append(file_path)
            return file_list
        if opt['debug'] == False:
            logdir = working_dir+'/runs'
            file_list = get_all_files(logdir)
            assert len(file_list[0])>0
            tsboard_log = os.path.dirname(file_list[0])
            writer = SummaryWriter(tsboard_log,purge_step=checkpoint_epoch) # tensorboard

    if opt["half"]:
        scaler = GradScaler()
        ac = autocast
    

    split.split_pics(opt)
    train_set = dataloader_ori.seamPicDataLoader(opt,"train")
    val_set = dataloader_ori.seamPicDataLoader(opt,"val")
    test_set = dataloader_ori.seamPicDataLoader(opt,"test")

    train_dl = None
    if ("train_sampler" in opt) and opt["train_sampler"]>0:
        train_data_cunt = len(train_set)
        pseudo_dataset = list(range(train_data_cunt))
        train_sempler_len = int(train_data_cunt/5)
        train_sampler = SubsetRandomSampler(pseudo_dataset[:train_sempler_len])
        train_dl = DataLoader(train_set,batch_size=opt["train_batch_size"],sampler=train_sampler)
    else:
        train_dl = DataLoader(train_set,batch_size=opt["train_batch_size"],shuffle=True)
    val_dl = DataLoader(val_set,batch_size=opt["val_batch_size"])
    print('train_dl:',len(train_dl))
    print('val_dl:',len(val_dl))

    for data in val_dl:
        input_shape = data[0].shape
        flop_net = generateModel(opt)
        flop_net.to(device)
        calc_flops(device,flop_net,[1]+list(input_shape)[1:])
        break
    
    early_stop_cnt = 0
    max_not_descent_cnt = 19

    loss_fn = MyLoss_ori_mult(opt)
    old_val_loss = -1
    old_valid_dis_ratio = -1

    if opt['copy']!=None:
        copy_params_from_pretrain_model(pretrain_net_state_dict['model_state_dict'],net)


    for epoch_idx in range(head,tail):
        print('start epoch {}'.format(epoch_idx))
        train_loss = train_model(epoch_idx=epoch_idx,device=device,train_dl=train_dl,net=net,
                                 loss_fn=loss_fn,optimizer=optimizer,writer=writer,scaler=scaler)
        val_loss,valid_dis_ratio = val_model(epoch_idx=epoch_idx,device=device,val_dl=val_dl,net=net,loss_fn=loss_fn,writer=writer,opt=opt)
        if opt['debug'] == False:
            if epoch_idx%20==0:
                save_checkpoints(epoch_idx,net,optimizer,val_loss,valid_dis_ratio,False,train_set,val_set,test_set)
            save_checkpoints(epoch_idx,net,optimizer,val_loss,valid_dis_ratio,True,train_set,val_set,test_set)
        if old_val_loss<0:
            old_val_loss = val_loss
        elif old_val_loss <= val_loss:
            early_stop_cnt += 1
        else:
            early_stop_cnt = 0
        old_val_loss = val_loss
        if early_stop_cnt>max_not_descent_cnt:
            print('not descent from {} epoch,stop'.format(early_stop_cnt))
            break

def calc_flops(device,net,input_shape): # ,input_shape=(1,1,600,1088)):
    # input_shape = (1,3,512,512)
    # net.select_to_device(current,[0],[0],'cpu')
    d = torch.rand(*input_shape).to(device)
    flops, params = profile(net, (d,))
    print("model flops:{} GFLOPs".format(flops/1e9))
    # return flops/1000000000

def copy_params_from_pretrain_model(src_net,dst_net):
    src_net_state_dict = src_net
    dst_net_state_dict = dst_net.state_dict()
    for name, param in src_net_state_dict.items():
        if name in dst_net_state_dict and dst_net_state_dict[name].size() == param.size():
            dst_net_state_dict[name].copy_(param)
            print(f"Copied layer {name} from src dst")

if __name__=='__main__':
    opt = tl.train_resnet18_opt
    opt["task_version"] += ""
    train(opt) 
