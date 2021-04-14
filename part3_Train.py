import os 
import torch
import segmentation_models_pytorch as smp
import argparse

from refer.utils import * 
from refer.datasets import * 
from refer.model import build_model

# Argument Parsing 

parser = argparse.ArgumentParser(description="Train the Model")
parser.add_argument('--base', '--b', help = 'the base address of the data')
parser.add_argument('--architecture', '--a', help = 'the Architecture of model')
parser.add_argument('--encoder', '--e', help = 'the Encoder of model')
parser.add_argument('--batch_size', '--bat', type=int, help = 'the batch_size of training')
parser.add_argument('--epochs', '--ep', type=int, help = 'the epochs of training')

args = parser.parse_args()

Basic = args.base
Basic = os.path.abspath(Basic)
archi = args.architecture
encoder = args.encoder
batch_size = args.batch_size
epochs = args.epochs

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

# Train 

def train (x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, model_dir, model, preprocessing_fn, CLASSES, DEVICE, loss, metrics, optimizer, batch_size, epochs) : 

    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    # train model for 40 epochs

    max_score = 0

    for i in range(0, epochs):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, model_dir)
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

# Train 

print ("Training the Model")
train (x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, model_dir, model, preprocessing_fn, CLASSES, DEVICE, loss, metrics, optimizer, batch_size, epochs)
print ("Train is finished")