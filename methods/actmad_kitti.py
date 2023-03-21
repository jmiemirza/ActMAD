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
from utils.utils import AverageMeter, SaveOutput
log = logging.getLogger('METHOD.ACTMAD')


def actmad_kitti(args, model, half_precision=False):
    log.info(f'Evaluating ActMAD on {args.dataset.upper()}')
    results = ResultsManager()
    args.verbose = True

    for args.task in globals.TASKS:
        if not set_severity(args):
            continue

        model.load_state_dict(torch.load(args.ckpt_path))
        log.info(f'TASK: {args.task}')
        all_ap = []
        set_yolo_save_dir(args, 'actmad', scenario='')
        device = next(model.parameters()).device

        # Half
        half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
        if half:
            model.half()

        # Loaders
        # train set loader for initial task
        tmp_task = args.task
        args.task = 'initial'
        dataloader_train = get_loader(args, split='train', pad=0.5, rect=True)
        args.task = tmp_task

        # test set loader for current task
        dataloader_val = get_loader(args, split='test', pad=0.5, rect=True)

        # test loader loader but with image size 840
        tmp_opt_imgsz = args.img_size
        args.img_size = [840, 840]
        dataloader_test = get_loader(args, split='test', pad=0.5, rect=True)
        args.img_size = tmp_opt_imgsz

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
        clean_var_act_list = [AverageMeter() for _ in range(n_chosen_layers)]
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
                    clean_mean_act_list[i].update(save_outputs[i].get_out_mean())  # compute mean from clean data
                    clean_var_act_list[i].update(save_outputs[i].get_out_var())  # compute variane from clean data

                    save_outputs[i].clear()
                    hook_list[i].remove()

            for i in range(n_chosen_layers):
                clean_mean_list_final.append(clean_mean_act_list[i].avg)  # [C, H, W]
                clean_var_list_final.append(clean_var_act_list[i].avg)  # [C, H, W]

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

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
            batch_mean_tta = [save_outputs_tta[x].get_out_mean() for x in range(n_chosen_layers)]
            batch_var_tta = [save_outputs_tta[x].get_out_var() for x in range(n_chosen_layers)]

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
            if args.actmad_save == 'each_batch':
                ap50 = test(batch_size=args.batch_size,
                            imgsz=args.img_size,
                            conf_thres=args.conf_thres,
                            iou_thres=args.iou_thres,
                            augment=args.augment,
                            verbose=False,
                            multi_label=True,
                            model=model,
                            dataloader=dataloader_test,
                            save_dir=args.save_dir)[-1]
                all_ap.append(np.mean(ap50))
                Path(f'{args.save_dir}/results_stf_ttt/{args.task}/all/').mkdir(parents=True, exist_ok=True)
                np.save(f'{args.save_dir}/results_stf_ttt/{args.task}/all/{batch_i}.npy', ap50)
                if np.mean(ap50) >= max(all_ap):
                    Path(f'{args.save_dir}/results_stf_ttt/{args.task}/best/').mkdir(exist_ok=True, parents=True)
                    np.save(f'{args.save_dir}/results_stf_ttt/{args.task}/best/{args.task}.npy', ap50)
                    state = {
                        'net': model.state_dict()
                    }
                    Path(f'{args.save_dir}/results_stf_ttt/models/').mkdir(parents=True, exist_ok=True)
                    torch.save(state, f'{args.save_dir}/results_stf_ttt/models/{args.task}.pt')

        if args.actmad_save == 'each_batch':
            # load best ckpt
            best_ckpt = torch.load(f'{args.save_dir}/results_stf_ttt/models/{args.task}.pt')
            model.load_state_dict(best_ckpt['net'])
            log.info(f'>>>>>> BEST checkpoint for {args.task}:')
        else:
            # save last ckpt
            Path(f'{args.save_dir}/results_stf_ttt/models/').mkdir(parents=True, exist_ok=True)
            state = {
                'net': model.state_dict()
            }
            torch.save(state, f'{args.save_dir}/results_stf_ttt/models/{args.task}.pt')

        map50 = test(batch_size=args.batch_size,
                     imgsz=args.img_size,
                     conf_thres=args.conf_thres,
                     iou_thres=args.iou_thres,
                     augment=args.augment,
                     verbose=args.verbose,
                     multi_label=True,
                     model=model,
                     dataloader=dataloader_test,
                     save_dir=args.save_dir)[0] * 100

        log.info(f'mAP@50 on current task ({args.task}): {map50:.1f}')
        severity_str = '' if args.task == 'initial' else f'{args.severity}'
        results.add_result('ActMAD', f'{args.task} {severity_str}', map50, 'KITTI')

