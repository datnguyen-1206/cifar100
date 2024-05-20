import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import wandb
import argparse
import yaml
import pprint
import shutil
import wandb
import datetime

from data_loader.get_cifar100 import *
from loop_one_epoch.loop import *
from loop_one_epoch.utils import *
from models import *
from optimizer import *

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 0. SETUP CONFIGURATION
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--experiment', default='example', type=str, help='path to YAML config file')
parser.add_argument('--rho', default=None, type=float, help='SAM rho')
parser.add_argument('--model_name', default=None, type=str, help='Model name')
parser.add_argument('--opt_name', default=None, type=str, help='Optimization name')
parser.add_argument('--project_name', default=None, type=str, help='Wandb Project name')
parser.add_argument('--framework_name', default=None, type=str, help='Logging Framework')
args = parser.parse_args()

yaml_filepath = os.path.join(".", "config", f"{args.experiment}.yaml")
with open(yaml_filepath, "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.Loader)
    cfg = override_cfg(cfg, args)
    pprint.pprint(cfg)
seed = cfg['trainer'].get('seed', 42)
initialize(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, start_epoch = 0, 0


print('==> Initialize Logging Framework..')
logging_name = get_logging_name(cfg)
logging_name += ('_' + current_time)

framework_name = cfg['logging']['framework_name']

if framework_name == 'wandb':
    wandb.init(
        project=cfg['logging']['project_name'], name=logging_name,
        config=cfg
    )
logging_dict = {}


# 1. BUILD THE DATASET
train_dataloader, val_dataloader, test_dataloader, classes = get_cifar100(**cfg['dataloader'])
try:
    num_classes = len(classes)
except:
    num_classes = classes
# 2. BUILD THE MODEL
model = get_model(**cfg['model'])
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
model.to(device)
epochs = cfg['trainer']['epochs']

# 3.a OPTIMIZING MODEL PARAMETERS
criterion = nn.CrossEntropyLoss().to(device)
optimizer = get_optimizer(model, **cfg['optimizer'])
scheduler = get_scheduler(optimizer, **cfg['scheduler'])

# 3.b TRAINING
if __name__ == '__main__':
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n----------------------------------------")
        train(train_dataloader, model, criterion, optimizer, device, logging_dict)
        best_acc = val(val_dataloader, model, criterion, device, logging_dict, logging_name, epoch, best_acc)
        scheduler.step()
        print(f"Best accuracy: {best_acc}")
        wandb.log(logging_dict)

    log_dict = {}
    test(test_dataloader, model, criterion, device, log_dict)
    wandb.log(logging_dict)