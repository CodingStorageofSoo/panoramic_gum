import numpy as np
import time
import torch
import json
from minio import Minio

import os 

from refer.utils import *
from refer.datasets import *
from torch.utils.data import DataLoader

def InferenceNUpload (model_dir, x_test_dir, y_test_dir, json_dir, preprocessing_fn, CLASSES, DEVICE, AD, ID, PS) : 

    # Connect the MinIO server

    client = Minio(AD, 
            access_key=ID,
            secret_key=PS,
            secure=True
        )

    # load best model 
    best_model = torch.load(model_dir)

    # create test dataset
    test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    test_dataloader = DataLoader(test_dataset)

    # test dataset without transformations for image visualization
    test_dataset_vis = Dataset(
        x_test_dir, y_test_dir, 
        classes=CLASSES,
    )

    # Name

    sub = x_test_dir
    name =[]
    n=0
    for n in range(len(test_dataset)):
        name1 = test_dataset.images_fps[n]
        name1 = name1.replace(sub, '')
        name.append(name1) 
        n += 1

    # Inference 

    print("Inferencing the Result")

    n = 0
    m = len(test_dataset)
    for n in range(len(test_dataset)):

        print_progress(n,m)

        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        # the coordinate 

        img = cv2.imread(test_dataset.masks_fps[0])

        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(imgray,0.5,255,0)

        contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        # Store the coordinate into json 

        filename = name[n] + '.json'
        file_path = json_dir + filename

        one = dict()
        one["one"] = contours[0].reshape(-1,2).tolist()

        with open(file_path, 'w') as make_file:
            json.dump(one, make_file)

        minIO_address = "pano_result_data_json" + filename
        client.fput_object("soo", minIO_address, file_path)
