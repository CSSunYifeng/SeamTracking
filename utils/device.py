import os
from matplotlib.style import available
import xmltodict
from typing import *
import time
import torch
from torch.utils.data import dataset

#（mb）
def get_available_gpus(memory_needed:int):
    r = os.popen('nvidia-smi -q -x').read()
    r = xmltodict.parse(r)
    gpus = r['nvidia_smi_log']['gpu']
    gpus = [gpu['fb_memory_usage']['free'] for gpu in gpus]
    gpus = [int(gpu.split(' ')[0]) for gpu in gpus]
    available_gpus = []
    for idx,gpu in enumerate(gpus):
        if gpu>=memory_needed:
            available_gpus.append(('cuda:'+str(idx),gpu))
    available_gpus.sort(key=lambda x:x[1],reverse=True)
    available_gpus = [x[0] for x in available_gpus]
    return available_gpus

def get_available_gpu(memory_needed:int):
    available_gpus = get_available_gpus(memory_needed)
    if len(available_gpus)==0:
        return None
    return available_gpus[0]

def get_available_gpu_blocked(memory_needed:int,max_try_cnt:Optional[int]=10):
    cnt = 0
    while max_try_cnt is None or cnt<max_try_cnt:
        available_gpus = get_available_gpus(memory_needed)
        if len(available_gpus)>0:
            return available_gpus[0]
        cnt += 1
        time.sleep(20)
    return None

def print_mem_used(device:str):
    device = torch.device(device)
    mem = torch.cuda.memory_allocated(device)
    mem /= 1024*1024
    maxmem = torch.cuda.max_memory_allocated(device)
    maxmem /= 1024*1024
    print('mem used:{}M \t maxmem used:{}M'.format(mem,maxmem))

if __name__=='__main__':
    print(get_available_gpus(2048))