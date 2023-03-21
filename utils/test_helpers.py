import time
import torch
import torch.nn as nn
# from utils.misc import *
from utils.utils import AverageMeter


def test(teloader, model):
    model.eval()
    # batch_time = AverageMeter('Time', ':6.3f')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    batch_time = AverageMeter()
    top1 = AverageMeter()
    # progress = ProgressMeter(len(teloader), batch_time, top1, prefix='Test: ')
    one_hot = []
    losses = []
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    end = time.time()
    for i, (inputs, labels) in enumerate(teloader):
        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            losses.append(criterion(outputs, labels).cpu())
            one_hot.append(predicted.eq(labels).cpu())
        acc1 = one_hot[-1].sum().item() / len(labels)
        top1.update(acc1, len(labels))
        batch_time.update(time.time() - end)
        end = time.time()
    model.train()
    return top1.avg


def test_individual(inputs, labels, model, verbose=False):
    model.eval()
    # top1 = AverageMeter('Acc@1', ':6.2f')
    top1 = AverageMeter()
    one_hot = []
    losses = []
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    with torch.no_grad():
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        losses.append(criterion(outputs, labels).cpu())
        one_hot.append(predicted.eq(labels).cpu())

        acc1 = one_hot[-1].sum().item() / len(labels)
        top1.update(acc1, len(labels))
    if verbose:
        one_hot = torch.cat(one_hot).numpy()
        losses = torch.cat(losses).numpy()
        return 1-top1.avg, one_hot, losses
    else:
        return top1.avg