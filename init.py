import logging
import logging.config
import os
from os.path import exists, join, realpath, split

import torchvision.models as tv_models
from torch import nn, save
from torchvision import __version__ as torchvision_version

import globals
import config
from models.experimental import attempt_load
from models.resnet_26 import ResNetCifar
from models.wide_resnet import WideResNet
from utils.general import check_img_size, increment_path
from utils.torch_utils import select_device
from utils.training import train
from utils.training_yolov3 import train as train_yolo

log = logging.getLogger('MAIN')


def set_paths(args):
    args.dataroot = config.PATHS[args.usr][args.dataset]['root']
    args.ckpt_path = config.PATHS[args.usr][args.dataset]['ckpt']


def init_net(args):
    device = select_device(args.device, batch_size=args.batch_size)

    if args.group_norm == 0:
        norm_layer = nn.BatchNorm2d
    else:
        def gn_helper(planes):
            return nn.GroupNorm(args.group_norm, planes)
        norm_layer = gn_helper

    def get_heads_classification(self):
        # returns last layer
        for m in self.modules(): pass
        return m

    if args.model == 'wrn':
        net = WideResNet(widen_factor=2, depth=40, num_classes=args.num_classes)
        WideResNet.get_heads = get_heads_classification

    elif args.model == 'res':
        net = ResNetCifar(args.depth, args.width, channels=3, classes=10,
                          norm_layer=norm_layer)
        ResNetCifar.get_heads = get_heads_classification

    elif args.model == 'res18':
        num_classes = 200 if args.dataset == 'tiny-imagenet' else 1000
        # if no checkpoint provided start from the pretrained one
        if not args.ckpt_path:
            if torchvision_version.startswith(('0.11', '0.12')):
                net = tv_models.resnet18(pretrained=True, norm_layer=norm_layer, num_classes=num_classes)
            else:
                net = tv_models.resnet18(weights='DEFAULT', norm_layer=norm_layer, num_classes=num_classes)
        else:
            net = tv_models.resnet18(norm_layer=norm_layer, num_classes=num_classes)
        # save(net.state_dict(), 'res18_imgnet_state_dict_ckpt.pt')
        # args.ckpt_path = join(split(realpath(__file__))[0], 'res18_imgnet_state_dict_ckpt.pt')
        tv_models.resnet.ResNet.get_heads = get_heads_classification

    elif args.model == 'yolov3':
        if hasattr(args, 'orig_ckpt_path'):
            args.ckpt_path = args.orig_ckpt_path
        if exists(args.ckpt_path):
            args.orig_ckpt_path = args.ckpt_path
            net = attempt_load(args.ckpt_path, map_location=device)
            args.gs = max(int(net.stride.max()), 32)
            args.img_size = [check_img_size(x, args.gs) for x in args.img_size]
        else:
            net = init_yolov3(args, device)
            args.gs = max(int(net.stride.max()), 32)
            args.img_size = [check_img_size(x, args.gs) for x in args.img_size]
            train_initial(args, net)
        save(net.state_dict(), 'yolo_kitti_state_dict_ckpt.pt')
        args.ckpt_path = join(split(realpath(__file__))[0], 'yolo_kitti_state_dict_ckpt.pt')

    else:
        raise Exception(f'Invalid model argument: {args.model}')

    net = net.to(device)
    setattr(net.__class__, 'bn_stats', {})

    if not args.ckpt_path or not exists(args.ckpt_path):
        train_initial(args, net)

    return net


def init_yolov3(args, device):
    import torch

    from models.yolo import Model
    from utils.google_utils import attempt_download
    from utils.torch_utils import intersect_dicts, torch_distributed_zero_first

    log.info('Loading yolov3.pt weights.')
    hyp = args.yolo_hyp()
    with torch_distributed_zero_first(args.global_rank):
        attempt_download('yolov3.pt')  # download if not found locally
    ckpt = torch.load('yolov3.pt', map_location=device)  # load checkpoint
    if hyp.get('anchors'):
        ckpt['model'].yaml['anchors'] = round(hyp['anchors'])  # force autoanchor
    net = Model(args.cfg or ckpt['model'].yaml, ch=3, nc=args.nc).to(device)  # create
    exclude = ['anchor'] if args.cfg or hyp.get('anchors') else []  # exclude keys
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, net.state_dict(), exclude=exclude)  # intersect
    net.load_state_dict(state_dict, strict=False)  # load
    net.to(device)
    return net


def train_initial(args, net):
    args.epochs = 350
    log.info('Checkpoint trained on initial task not found - Starting training.')
    args.task = 'initial'
    save_dir_path = join('checkpoints', args.dataset, args.model, 'initial')

    if args.model == 'yolov3':
        device = select_device(args.device, batch_size=args.batch_size)
        args.save_dir = save_dir_path
        train_yolo(args.yolo_hyp(), args, device, model=net)
        args.ckpt_path = join(split(realpath(__file__))[0], save_dir_path, 'weights', 'best.pt')
    else:
        save_file_name = f'{args.dataset}_initial.pt'
        path = join(save_dir_path, save_file_name)
        args.severity = 0
        train(net, args, path)
        args.ckpt_path = join(split(realpath(__file__))[0], path)
    log.info(f'Checkpoint trained on initial task saved at {args.ckpt_path}')


def init_settings(args):
    os.makedirs('results', exist_ok=True)

    # Set default learning rate if none provided
    if not args.lr:
        if args.dataset == 'cifar10':
            args.lr = 1e-2
        elif args.dataset == 'cifar100':
            args.lr = 1e-3
        elif args.dataset == 'imagenet':
            args.lr = 25e-5
        elif args.dataset == 'kitti':
            args.lr = 1e-4
        else:
            raise Exception(f'No default lr for {args.dataset}! Check dataset name or set --lr')
        log.info(f'Set learning rate to {args.lr}')

    # Set default batch size if none provided
    if not args.batch_size:
        if args.dataset == 'cifar10':
            args.batch_size = 128
        elif args.dataset == 'cifar100':
            args.batch_size = 128
        elif args.dataset == 'imagenet':
            args.batch_size = 128
        elif args.dataset == 'kitti':
            args.batch_size = 30
        else:
            raise Exception(f'No default batch size for {args.dataset}! Check dataset name or set --batch_size')
        log.info(f'Set batch size to {args.batch_size}')

    if args.dataset == 'kitti':
        if not args.model:
            args.model = 'yolov3'
        if args.tasks:
            globals.TASKS = args.tasks
        else:
            globals.TASKS = config.KITTI_TASKS
        args.num_severities = max([len(args.fog_severities),
                                   len(args.rain_severities),
                                   len(args.snow_severities)])
        globals.KITTI_SEVERITIES['fog'] = args.fog_severities
        globals.KITTI_SEVERITIES['rain'] = args.rain_severities
        globals.KITTI_SEVERITIES['snow'] = args.snow_severities

        # set args.yolo_hyp to a function returning a copy of globals.YOLO_HYP
        # as some values get changed during training, which would lead to
        # false values if multiple training sessions are started within one
        # execution of the script
        def get_yolo_hyp():
            return config.YOLO_HYP.copy()
        config.YOLO_HYP['lr0'] = args.lr
        args.yolo_hyp = get_yolo_hyp

        # opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        # opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
        args.world_size = 1
        args.global_rank = -1

        args.img_size.extend([args.img_size[-1]] * (2 - len(args.img_size)))  # extend to 2 sizes (train, test)
        args.total_batch_size = args.batch_size
        args.nc = 8
        args.names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
                      'Cyclist', 'Tram', 'Misc']
    else:
        if args.tasks:
            globals.TASKS = args.tasks
        else:
            globals.TASKS = config.ROBUSTNESS_TASKS
        if args.dataset in ['imagenet', 'imagenet-mini']:
            from utils.datasets import ImgNet
            ImgNet.initial_dir = args.dataset
        args.num_severities = len(args.robustness_severities)
        args.severity = None
        config.ROBUSTNESS_SEVERITIES = args.robustness_severities
        if args.dataset in ['cifar10', 'cifar100'] and not args.model:
            args.num_classes = 10 if args.dataset == 'cifar10' else 100
            args.model = 'wrn'
        elif args.dataset in ['imagenet', 'imagenet-mini'] and not args.model:
            args.model = 'res18'

