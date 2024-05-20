import torch 
import os
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from optimizer.sam import SAM
from .utils import *



def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

def train(
    train_dataloader,
    model,
    criterion,
    optimizer,
    device,
    logging_dict,
):
    model.train()
    loss = 0
    total = 0
    correct = 0
    print (train_dataloader)
    for batch_index, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        opt_name = type(optimizer).__name__
        if opt_name == 'SGD':
            with torch.no_grad():
                predictions = model(images)
                first_loss = criterion(predictions, labels)

                first_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        elif opt_name == 'SAM':
            # first forward-backward step
            enable_running_stats(model)  # <- this is the important line
            with torch.cuda.amp.autocast():
                predictions = model(images)
                first_loss = criterion(predictions, labels)
            first_loss.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)  # <- this is the important line
            criterion(model(images), labels).backward()
            optimizer.second_step(zero_grad=True)
        
        loss += first_loss.item()
        _, predicted = predictions.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loss_mean = loss/(batch_index+1)
        acc = 100.*correct/total
        progress_bar(batch_index, len(train_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (loss_mean, acc, correct, total))
        
        logging_dict['TRAIN/loss'] = loss_mean
        logging_dict['TRAIN/acc'] = acc


@torch.no_grad()
def val(
    val_dataloader,
    model,
    criterion,
    device,
    logging_dict,
    logging_name,
    epoch,
    best_acc,
):
    loss = 0
    total = 0
    correct = 0
    model.eval()
    for batch_idx, (images, labels) in enumerate(val_dataloader):
        images, labels = images.to(device), labels.to(device)

        predictions = model(images)
        first_loss = criterion(predictions, labels)

        loss += first_loss.item()
        _, predicted = predictions.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loss_mean = loss/(batch_idx+1)
        acc = 100.*correct/total
        progress_bar(batch_idx, len(val_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (loss_mean, acc, correct, total))
        
    if acc > best_acc:
        print('Saving best checkpoint ...')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'loss': loss,
            'epoch': epoch
        }
        save_path = os.path.join('checkpoint', logging_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(state, os.path.join(save_path, 'ckpt_best.pth'))
        best_acc = acc

        logging_dict['VAL/best_acc'] = best_acc
        logging_dict['VAL/loss'] = loss_mean
        logging_dict['VAL/acc'] = acc
    return best_acc

@torch.no_grad()
def test(
    test_dataloader,
    model,
    criterion,
    device,
    logging_dict,
    logging_name = 'default',
):
    print('==> Resuming from best checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    load_path = os.path.join('checkpoint', logging_name, 'ckpt_best.pth')
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['net'])

    model.eval()
    for batch_idx, (images, labels) in enumerate(test_dataloader):
        images, labels = images.to(device), labels.to(device)
        predictions = model(images)
        first_loss = criterion(predictions, labels)

        loss += first_loss.item()
        _, predicted = predictions.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loss_mean = loss/(batch_idx+1)
        acc = 100.*correct/total
        progress_bar(batch_idx, len(test_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (loss_mean, acc, correct, total))

        logging_dict['TEST/loss'] = loss_mean
        logging_dict['TEST/acc'] = acc
        # Add heatmap https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap