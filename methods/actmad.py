import argparse
from pathlib import Path
import torch.optim as optim
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from utils.data_loader import get_loader, set_severity, set_yolo_save_dir
import globals
import logging
from utils.testing_yolov3 import test
from utils.results_manager import ResultsManager

log = logging.getLogger('MAIN.ACTMAD')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out.clone())

    def clear(self):
        self.outputs = []


def get_out(output_holder):
    out = torch.vstack(output_holder.outputs)
    out = torch.mean(out, dim=0)
    return out


def get_clean_out(output_holder):
    out = torch.vstack(output_holder.outputs)

    out = torch.mean(out, dim=0)
    return out


def get_out_var(output_holder):
    out = torch.vstack(output_holder.outputs)
    out = torch.var(out, dim=0)
    return out


def get_clean_out_var(out_holder):

    out = torch.vstack(out_holder.outputs)
    out = torch.var(out, dim=0)
    return out


def take_mean(input_ten):
    input_ten = torch.mean(input_ten, dim=0)
    return input_ten


def actmad(opt, model, half_precision=False):
    results = ResultsManager()
    opt.verbose = True
    tmp_lr = opt.lr
    opt.lr = opt.actmad_lr
    for opt.task in globals.TASKS:
        if not set_severity(opt):
            continue

        model.load_state_dict(torch.load(opt.ckpt_path))
        log.info(f'TASK: {opt.task}')
        all_ap = []
        set_yolo_save_dir(opt, 'actmad', scenario='')
        device = next(model.parameters()).device

        # Half
        half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
        if half:
            model.half()

        # Loaders
        # train set loader for initial task
        tmp_task = opt.task
        opt.task = 'initial'
        dataloader_train = get_loader(opt, split='train', pad=0.5, rect=True)
        opt.task = tmp_task

        # validation set loader for current task
        dataloader_val = get_loader(opt, split='val', pad=0.5, rect=True)

        # test loader same as valiadtion loader but with image size 840
        tmp_opt_imgsz = opt.img_size
        opt.img_size = [840, 840]
        dataloader_test = get_loader(opt, split='val', pad=0.5, rect=True)
        opt.img_size = tmp_opt_imgsz

        l1_loss = nn.L1Loss(reduction='mean')

        for k, v in model.named_parameters():
            v.requires_grad = True

        chosen_bn_layers = []
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                chosen_bn_layers.append(m)
        chosen_bn_layers = chosen_bn_layers[26:]
        n_chosen_layers = len(chosen_bn_layers)
        save_outputs = [SaveOutput() for _ in range(n_chosen_layers)]
        clean_mean_act_list = [AverageMeter() for _ in range(n_chosen_layers)]
        clean_var_act_list = [AverageMeter()  for _ in range(n_chosen_layers)]
        clean_mean_list_final = []
        clean_var_list_final = []
        with torch.no_grad():
            for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader_train)):
                img = img.to(device, non_blocking=True)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                model.eval()
                hook_list = [chosen_bn_layers[i].register_forward_hook(save_outputs[i]) for i in range(n_chosen_layers)]
                _ = model(img)

                for i in range(n_chosen_layers):
                    clean_mean_act_list[i].update(get_clean_out(save_outputs[i]))  # compute mean from clean data
                    clean_var_act_list[i].update(get_clean_out_var(save_outputs[i]))  # compute variane from clean data

                    save_outputs[i].clear()
                    hook_list[i].remove()

            for i in range(n_chosen_layers):
                clean_mean_list_final.append(clean_mean_act_list[i].avg)  # [C, H, W]
                clean_var_list_final.append(clean_var_act_list[i].avg)  # [C, H, W]

        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

        log.info('Starting TEST TIME ADAPTATION WITH ActMAD...')
        # ap_epochs = list()

        for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader_val)):
            model.train()
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()

            optimizer.zero_grad()
            save_outputs_tta = [SaveOutput() for _ in range(n_chosen_layers)]

            hook_list_tta = [chosen_bn_layers[x].register_forward_hook(save_outputs_tta[x])
                             for x in range(n_chosen_layers)]
            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            img = img.cuda()
            _ = model(img)
            batch_mean_tta = [get_out(save_outputs_tta[x]) for x in range(n_chosen_layers)]
            batch_var_tta = [get_out_var(save_outputs_tta[x]) for x in range(n_chosen_layers)]

            loss_mean = torch.tensor(0, requires_grad=True, dtype=torch.float).float().cuda()
            loss_var = torch.tensor(0, requires_grad=True, dtype=torch.float).float().cuda()

            for i in range(n_chosen_layers):
                loss_mean += l1_loss(batch_mean_tta[i].cuda(), clean_mean_list_final[i].cuda())
                loss_var += l1_loss(batch_var_tta[i].cuda(), clean_var_list_final[i].cuda())

            loss = loss_mean + loss_var

            loss.backward()
            optimizer.step()
            for z in range(n_chosen_layers):
                save_outputs_tta[z].clear()
                hook_list_tta[z].remove()

            # test performance after every batch and save
            if opt.actmad_save == 'each_batch':
                ap50 = test(batch_size=opt.batch_size,
                            imgsz=opt.img_size,
                            conf_thres=opt.conf_thres,
                            iou_thres=opt.iou_thres,
                            augment=opt.augment,
                            verbose=False,
                            multi_label=True,
                            model=model,
                            dataloader=dataloader_test,
                            save_dir=opt.save_dir)[-1]
                all_ap.append(np.mean(ap50))
                Path(f'{opt.save_dir}/results_stf_ttt/{opt.task}/all/').mkdir(parents=True, exist_ok=True)
                np.save(f'{opt.save_dir}/results_stf_ttt/{opt.task}/all/{batch_i}.npy', ap50)
                if np.mean(ap50) >= max(all_ap):
                    Path(f'{opt.save_dir}/results_stf_ttt/{opt.task}/best/').mkdir(exist_ok=True, parents=True)
                    np.save(f'{opt.save_dir}/results_stf_ttt/{opt.task}/best/{opt.task}.npy', ap50)
                    state = {
                        'net': model.state_dict()
                    }
                    Path(f'{opt.save_dir}/results_stf_ttt/models/').mkdir(parents=True, exist_ok=True)
                    torch.save(state, f'{opt.save_dir}/results_stf_ttt/models/{opt.task}.pt')

        if opt.actmad_save == 'each_batch':
            # load best ckpt
            best_ckpt = torch.load(f'{opt.save_dir}/results_stf_ttt/models/{opt.task}.pt')
            model.load_state_dict(best_ckpt['net'])
            log.info(f'>>>>>> BEST checkpoint for {opt.task}:')
        else:
            # save last ckpt
            Path(f'{opt.save_dir}/results_stf_ttt/models/').mkdir(parents=True, exist_ok=True)
            state = {
                'net': model.state_dict()
            }
            torch.save(state, f'{opt.save_dir}/results_stf_ttt/models/{opt.task}.pt')

        map50 = test(batch_size=opt.batch_size,
                     imgsz=opt.img_size,
                     conf_thres=opt.conf_thres,
                     iou_thres=opt.iou_thres,
                     augment=opt.augment,
                     verbose=opt.verbose,
                     multi_label=True,
                     model=model,
                     dataloader=dataloader_test,
                     save_dir=opt.save_dir)[0] * 100

        log.info(f'mAP@50 on current task ({opt.task}): {map50:.1f}')
        severity_str = '' if opt.task == 'initial' else f'{opt.severity}'
        results.add_result('ActMAD', f'{opt.task} {severity_str}', map50, 'online')

    opt.lr = tmp_lr


