from refer.utils import * 
from refer.datasets import * 
from refer.inference import *
from refer.model import build_model
from refer.monitor import Monitor
from refer.monitor import gpu_max

import segmentation_models_pytorch as smp
import openpyxl 
import time
import argparse
import os 
import psutil
from minio import Minio

# Calculate the initial

start = time.time()

first_mem = psutil.virtual_memory()
initial_cpu = first_mem.used / (2 ** 20)

record_file_first = "first.txt"
os.system("gpustat > " + record_file_first)
initial_gpu = gpu_max(record_file_first)

# Argument Parsing 

parser = argparse.ArgumentParser(description="Infer the result")
parser.add_argument('--base', '--b', help = 'the base address of the data')
parser.add_argument('--architecture', '--a', help = 'the Architecture of model')
parser.add_argument('--encoder', '--e', help = 'the Encoder of model')
args = parser.parse_args()

Basic = args.base
Basic = os.path.abspath(Basic)
archi = args.architecture
encoder = args.encoder

# Directory

x_train_dir = os.path.join(Basic, 'train')
x_train_dir = os.path.abspath(x_train_dir)
y_train_dir = os.path.join(Basic, 'M_train')
y_train_dir = os.path.abspath(y_train_dir)

x_valid_dir = os.path.join(Basic, 'valid')
x_valid_dir = os.path.abspath(x_valid_dir)
y_valid_dir = os.path.join(Basic, 'M_valid')
y_valid_dir = os.path.abspath(y_valid_dir)

x_test_dir = os.path.join(Basic, 'tests')
x_test_dir = os.path.abspath(x_test_dir)
y_test_dir = os.path.join(Basic, 'M_tests')
y_test_dir = os.path.abspath(y_test_dir)

model_dir = os.path.join(Basic, 'best_model.pth')
model_dir = os.path.abspath(model_dir)

# Basic Setting 

CLASSES = ['gum']
ACTIVATION = 'sigmoid' 
DEVICE = 'cuda'
ENCODER_WEIGHTS = 'imagenet'

loss = smp.utils.losses.DiceLoss()

metrics = [smp.utils.metrics.IoU(threshold=0.5),]

# Model Setting 

model, preprocessing_fn = build_model (Architecture = archi, encoder = encoder, weights = ENCODER_WEIGHTS, CLASSES = CLASSES, activation = ACTIVATION)

optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001),])

# Inference 

name = archi + encoder + "gpu.txt"

cpu, mean, var, max_v, min_v, first, done, m = Monitor (inference, model_dir, x_test_dir, y_test_dir, preprocessing_fn, CLASSES, DEVICE, name)

Latency = done - first 

Throughput = (done - start) / m 

CPU = int(max(cpu)) - int(initial_cpu)

GPU = int(gpu_max(name)) - int(initial_gpu)

print ()
print_inf (archi, encoder, Latency, Throughput, CPU, GPU, mean, var, max_v, min_v) 

# Store the data 

excel = os.path.join(Basic,"result.xlsx")
load_wb = openpyxl.load_workbook(excel)
load_ws = load_wb['result']
load_ws.append([archi, encoder, Latency, Throughput, CPU, GPU, mean, var, max_v, min_v])
load_wb.save(excel)



