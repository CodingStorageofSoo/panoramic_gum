import time
import torch
from refer.utils import *
from refer.datasets import *
from torch.utils.data import DataLoader
import numpy as np
import os 

def inference (model_dir, x_test_dir, y_test_dir, preprocessing_fn, CLASSES, DEVICE) : 

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

    prob=[]
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

        g_count, p_count, o_count = my_IOU_score (gt_mask, pr_mask)

        prob.append((o_count / (g_count+p_count-o_count))*100)

        if n == 0 :
            first = time.time()

    print_progress(m,m)

    prob=np.array(prob)
    max_v = np.max(prob)
    min_v = np.min(prob)
    mean = np.mean(prob)
    var = np.var(prob)

    done = time.time()

    return mean, var, max_v, min_v, first, done, m