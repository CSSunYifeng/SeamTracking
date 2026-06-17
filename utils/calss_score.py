import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def TP_FP_FN(labels:list,gt_labels:list,cls_num):
    if not len(labels) == len(gt_labels):
        raise "labels length not equal gt_label length"
    confusion_matrix = np.zeros((cls_num,cls_num))
    for idx in range(len(gt_labels)):
        confusion_matrix[gt_labels[idx]+1,labels[idx]+1]+=1
    ans = np.zeros((cls_num,3)) # TP,FP,FN
    for row in range(cls_num):
        for col in range(cls_num):
            if row>col:
                ans[row,2] += confusion_matrix[row,col]
            elif row == col:
                ans[row,0] += confusion_matrix[row,col]
            else:
                ans[row,1] += confusion_matrix[row,col]
    return ans

def micro_f1score(labels:list,gt_labels:list,cls_num):
    ans = TP_FP_FN(labels,gt_labels,cls_num)
    TP_sum = np.sum(ans[:,0])
    FP_sum = np.sum(ans[:,1])
    FN_sum = np.sum(ans[:,2])
    P = TP_sum/(TP_sum+FP_sum)
    R = TP_sum/(TP_sum+FN_sum)
    F1_score = 2*P*R/(P+R)
    return F1_score

def macro_f1score(labels:list,gt_labels:list,cls_num):
    ans = TP_FP_FN(labels,gt_labels,cls_num)
    P_list = np.array([ans[i,0]/(ans[i,0]+ans[i,1]) for i in range(cls_num)])
    R_list = np.array([ans[i,0]/(ans[i,0]+ans[i,2]) for i in range(cls_num)])
    P = np.mean(P_list)
    R = np.mean(R_list)
    F1_score = 2*P*R/(P+R)
    return F1_score
    

def plot_confusion_matrix(labels: list, gt_labels: list, cls_num: int, class_names=None, save_path='confusion_matrix.png'):
    if not len(labels) == len(gt_labels):
        raise ValueError("labels length not equal gt_label length")

    # 
    confusion_matrix = np.zeros((cls_num, cls_num), dtype=int)
    for idx in range(len(gt_labels)):
        confusion_matrix[gt_labels[idx], labels[idx]] += 1

    if class_names is None:
        class_names = [f'Class {i}' for i in range(cls_num)]

    # 
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    # 
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    
    

    