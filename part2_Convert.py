from minio import Minio
import os 
import argparse
import numpy as np 
import cv2
import glob
import json

from refer.utils import CreateExel
from refer.utils import createFolder
from refer.utils import print_progress

# Argument  

parser = argparse.ArgumentParser(description="Convert the Data")
parser.add_argument('--base', '--b', help = 'the base address of the data')
args = parser.parse_args()

Basic = args.base
Basic = os.path.abspath(Basic)

def convert2line (Basic, folder): 

    # The Address

    Input_Address = os.path.join(Basic, folder)
    Input_Address = os.path.abspath(Input_Address)
    print('Input Address is ' + Input_Address)

    M_folder = "M_" + folder
    Mask_Address = Basic + M_folder
    Mask_Address = os.path.join(Basic, M_folder)
    Mask_Address = os.path.abspath(Mask_Address)
    print('Address of Masked file is ' + Mask_Address)
    createFolder(Mask_Address)

    # Arrange the file

    text_storage=[]
    image_storage=[]

    for name in sorted(glob.glob(Input_Address + '/*.json')):
        text_storage.append(name) 
 
    for name in sorted(glob.glob(Input_Address + '/*.BMP')):
        image_storage.append(name) 

    # Convert

    i=0
    print_progress(i,len(text_storage))

    j=0
    for j in range(len(text_storage)):


        # Step 1. Read the image   
        img = cv2.imread(image_storage[j]) 
        (a,b,c)=img.shape
        name1 = image_storage[j]
        sub1 = Input_Address
        sub2 ="/"
        name2 = name1.replace(sub1, '')
        fname = name2.replace(sub2, '')

        # Step 2. 

        text = open(text_storage[j])
        data = json.load(text)
        pts = data ["one"]

        size = np.array([b,a])
        pts = (pts*size).astype(int)

        # Step 3. Generate Mask 
        img=np.zeros((a,b))
        Mask = cv2.fillPoly(img, [pts], 1)
        Mask = np.array(Mask)
        os.chdir(Mask_Address) 
        cv2.imwrite(fname, Mask)     

        i += 1
        print_progress(i,len(text_storage))

    # Remove the json file 

    for name in sorted(glob.glob(Input_Address + '/*.json')):
        os.remove(name)

# Download & Convert

folder_name = ['tests', 'train', 'valid']

for folder in folder_name:      

    # Convert

    print ("Converting the file")
    convert2line (Basic = Basic , folder = folder) 