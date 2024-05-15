import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import wandb
import pprint
import shutil
import wandb

from data_loader.get_cifar100 import *
from loop_one_epoch.loop import *
from loop_one_epoch.utils import *
from models import *
from optimizer import *

data_name = 'cifar100'
batch_size = 32
model_name = 'resnet18'
opt_name = 'SAM'
lr = 0.1
momentum = 0.9
weight_decay = 0.001
rho = 0.1
sch_name = 'cosine'
T_max = 200
epochs = 200
log_dict = {}
test_dict = {}

# 1. BUILD THE DATASET
train_dataloader, val_dataloader, test_dataloader, classes = get_cifar100(batch_size=batch_size, num_workers=4)
try:
    num_classes = len(classes)
except:
    num_classes = classes
# 2. BUILD THE MODEL
model = get_model(model_name)
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
model.to(device)

# 3.a OPTIMIZING MODEL PARAMETERS
criterion = nn.CrossEntropyLoss().to(device)
optimizer = get_optimizer(model, opt_name, opt_hyperparameter={'lr':lr, 'momentum':momentum, 'weight_decay':weight_decay, 'rho':rho})
scheduler = get_scheduler(optimizer, sch_name, sch_hyperparameter={'T_max':T_max})

# 3.b TRAINING
if __name__ == '__main__':
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n----------------------------------------")
        train(train_dataloader, model, criterion, optimizer, device, log_dict)
        best_acc = val(val_dataloader, model, criterion, device, log_dict, test_dict, epoch, best_acc)
        scheduler.step()
        print(f"Best accuracy: {best_acc}")
        # wandb.log(log_dict)

    log_dict = {}
    test(test_dataloader, model, criterion, device, log_dict)
    # wandb.log(log_dict)