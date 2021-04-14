import json
from PIL import Image
import numpy as np
import cv2 
import os 

import glob 
from minio import Minio
from minio.error import S3Error
import time
from random import randrange
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import segmentation_models_pytorch as smp
import albumentations as albu
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import sys
from time import sleep
import openpyxl

#### 함수 내 변수는 함수가 끝이나면 사라진다. 
# 해당 변수를 살리고 싶다면 return 을 해야 한다. 
# 하지만 출력 시 반드시 변수를 지정해줘야 알맞게 쓸 수 있다. 

# def function () :

#     a = 3 
#     b = 5
#     c = a + b 
#     d = a * b 

#     return c, d

# c, d = function ()

# print (c, d)

#### 일반적으론 입력 받는 변수가 없더라도 괄호를 써야 한다. 

# def Hello ():
#     print ("Hello")

# Hello ()

#### 함수는 함수 밖의 변수도 참조한다. (하지만 함수는 함수 밖의 연산에 영향을 줄 수 없다!) (다만 return 값으로 그 값을 알 수 있을 뿐)

# a = 5 

# def function () :

#     a = 10 

# Create Folder

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

# helper function for data visualization

def visualize(**images):

    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def print_progress(iteration, total, prefix='Progress:', suffix='Complete', decimals=1, bar_length=100):
    
    str_format = "{0:." + str(decimals) + "f}"
    current_progress = iteration / float(total)
    percents = str_format.format(100 * current_progress)
    filled_length = int(round(bar_length * current_progress))
    bar = "#" * filled_length + '-' * (bar_length - filled_length)

    # 캐리지 리턴(\r) 문자를 이용해서 출력후 커서를 라인의 처음으로 옮김 
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    # 현재 위치가 전체 위치에 도달하면 개행문자 추가 
    if iteration == total:
        sys.stdout.write('\n')

    # 버퍼의 문자를 출력 
    sys.stdout.flush()

def my_IOU_score (gt_mask, pr_mask) : 
    
    # b is width / a is height 

    (a,b)=gt_mask.shape

    # d is width / c is height 

    (c,d)=pr_mask.shape

    if (a,b) == (c,d) :

        g_count=0
        for i in range(a):
            for j in range(b):
                if gt_mask[i][j] == 0:
                    g_count += 1

        p_count=0
        for i in range(c):
            for j in range(d):
                if pr_mask[i][j] == 0:
                    p_count += 1

        # Overlapping Area

        o_count=0
        for i in range(a):
            for j in range(b):
                if (gt_mask[i][j] == 0) and (pr_mask[i][j] == 0):
                    o_count += 1

    else : 
        print ("error")

    return g_count, p_count, o_count

def CreateExel (Basic, name):

    write_wb = openpyxl.Workbook()
    write_ws = write_wb.create_sheet(name)
    write_ws.append(['Architecture', 'Encoder', 'Latency', 'Throughput', 'CPU', 'GPU', "Mean", "Variance", "Max of Accuracy", "Min of Accuracy"])
    name = name + ".xlsx"
    address = os.path.join(Basic,name)
    write_wb.save(address)

def print_inf (Architecture, ENCODER, Latency,Throughput, CPU, GPU, mean, var, max_v, min_v) :

    print ("The Architecture is ", Architecture) 
    print ("The Encoder is ", ENCODER) 
    print ("The Latency is ", Latency, "s" )
    print ("The Throughput is ", Throughput, "s")
    print ("The used memory of CPU is ", CPU , "MB")
    print ("The used memory of GPU is ", GPU, "MB") 
    print ("The Mean of Accuracy is ",mean ,"%")
    print ("The Variance of Accuracy is ", var,"%")
    print ("The Max of Accuracy is ", max_v,"%")
    print ("The Min of Accuracy is ", min_v,"%")