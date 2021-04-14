from refer.utils import * 
from refer.datasets import * 
from refer.train import *
from refer.inference import *
from refer.monitor import *
from refer.InferenceNUpload import *

import segmentation_models_pytorch as smp
import openpyxl 
import time
import argparse
import os 
import psutil

# python3 part5_Final.py --b "./" --a "Unet" --e "se_resnet50" --be 'best_model.pth' --m "minio.aivisor.vsmart00.com" --i "haruband" --p "haru1004"

# Argument Parsing 

parser = argparse.ArgumentParser(description="Infer and Upload the data based on the best model")
parser.add_argument('--base', '--b', help = 'the base address of the data')
parser.add_argument('--architecture', '--a', help = 'the Architecture of model')
parser.add_argument('--encoder', '--e', help = 'the Encoder of model')
parser.add_argument('--best_model', '--be', help = 'the best model')
parser.add_argument('--address', '--m', help = 'the address of MinIO')
parser.add_argument('--id', '--i', help = 'ID')
parser.add_argument('--password', '--p', help = 'Password')
args = parser.parse_args()

Basic = args.base
Basic = os.path.abspath(Basic)
archi = args.architecture
encoder = args.encoder
best_model = args.best_model
AD=args.address
ID=args.id
PS=args.password

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

json_dir = os.path.join(Basic, 'result')
json_dir = os.path.abspath(json_dir)
createFolder(json_dir)

model_dir = os.path.join(Basic, best_model)
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

InferenceNUpload (model_dir, x_test_dir, y_test_dir, json_dir, preprocessing_fn, CLASSES, DEVICE, AD, ID, PS)





