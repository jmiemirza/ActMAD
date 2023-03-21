from __future__ import print_function
import torch.optim as optim
from utils.test_helpers import *
import torch.nn as nn
from utils.utils import SaveEmb
import globals, logging
from utils.data_loader import get_loader
from utils.results_manager import ResultsManager
from utils.data_loader import set_severity
log = logging.getLogger('METHOD.ACTMAD')


def actmad_cifar(args, net):
    log.info(f'Evaluating ActMAD on {args.dataset.upper()}')
    results = ResultsManager()
    l1_loss = nn.L1Loss(reduction='mean')
    all_res = []
    ckpt = torch.load(args.ckpt_path)
    net.load_state_dict(ckpt)
    chosen_layers = []
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            chosen_layers.append(m)

    n_chosen_layers = len(chosen_layers)
    hook_list = [SaveEmb() for _ in range(n_chosen_layers)]
    clean_mean = []
    clean_var = []

    args.task = 'initial'
    args.severity = 0
    tr_loader = get_loader(args, split='train')

    for idx, (inputs, labels) in enumerate(tr_loader):
        hooks = [chosen_layers[i].register_forward_hook(hook_list[i]) for i in range(n_chosen_layers)]
        inputs = inputs.cuda()
        with torch.no_grad():
            net.eval()
            _ = net(inputs)

            for yy in range(n_chosen_layers):
                hook_list[yy].statistics_update(), hook_list[yy].clear(), hooks[yy].remove()

    for i in range(n_chosen_layers):
        clean_mean.append(hook_list[i].pop_mean()), clean_var.append(hook_list[i].pop_var())

    set_severity(args)

    all_res = list()
    for args.task in globals.TASKS:
        para_to_opt = []
        err_corr = []
        net.load_state_dict(ckpt)

        te_loader_ = get_loader(args, split='test', use_tr_transform=True)
        te_loader = get_loader(args, split='test')

        err_cls = 100 - (test(te_loader, net) * 100)
        log.info(f'Task {args.task}:')
        log.info(f'Error before adaptation: {err_cls: .1f}')
        print('Iteration \t\t Loss \t\t Error(%)')
        parameters = list(para_to_opt)
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)

        for idx, (inputs, labels) in enumerate(te_loader_):
            net.train()
            for name, param in net.named_parameters():
                for m in net.modules():
                    if isinstance(m, nn.modules.batchnorm._BatchNorm):
                        m.eval()
            optimizer.zero_grad()
            save_outputs_tta = [SaveEmb() for _ in range(n_chosen_layers)]

            hooks_list_tta = [chosen_layers[i].register_forward_hook(save_outputs_tta[i])
                            for i in range(n_chosen_layers)]

            inputs = inputs.cuda()
            out = net(inputs)
            act_mean_batch_tta = []
            act_var_batch_tta = []
            for yy in range(n_chosen_layers):
                save_outputs_tta[yy].statistics_update()
                act_mean_batch_tta.append(save_outputs_tta[yy].pop_mean())
                act_var_batch_tta.append(save_outputs_tta[yy].pop_var())

            for z in range(n_chosen_layers):
                save_outputs_tta[z].clear()
                hooks_list_tta[z].remove()

            loss_mean = torch.tensor(0, requires_grad=True, dtype=torch.float).float().cuda()
            loss_var = torch.tensor(0, requires_grad=True, dtype=torch.float).float().cuda()
            for i in range(n_chosen_layers):
                loss_mean += l1_loss(act_mean_batch_tta[i].cuda(), clean_mean[i].cuda())
                loss_var += l1_loss(act_var_batch_tta[i].cuda(), clean_var[i].cuda())
            loss = (loss_mean + loss_var) * 0.5

            loss.backward()
            optimizer.step()

            err_cls = 100 - (test(te_loader, net) * 100)
            err_corr.append(err_cls)

            print(f'{idx} \t\t\t {loss:.3f} \t\t {err_cls: .1f}')

        min_err_corr = min(err_corr)
        log.info(f'Minimum Error after adaptation: {min_err_corr:.1f}')
        results.add_result('ActMAD', f'{args.task}', min_err_corr, args.dataset)
        all_res.append(min_err_corr)
    log.info(f'Mean Error: {sum(all_res) / len(all_res)}')
