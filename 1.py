import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import datetime  # 用于超时设置的标准库
import torch
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import tensorboard_logger as tb_logger
import numpy as np
from model import PiCO
from resnet import *
from utils.utils_algo import *
from utils.utils_loss import partial_loss, SupConLoss
from utils.cub200 import load_cub200
from utils.cifar10 import load_cifar10
from utils.cifar100 import load_cifar100

torch.set_printoptions(precision=2, sci_mode=False)

parser = argparse.ArgumentParser(description='PyTorch implementation of ICLR 2022 Oral paper PiCO')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100', 'cub200'],
                    help='dataset name (cifar10)')
parser.add_argument('--exp-dir', default='experiment/PiCO', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=['resnet18'],
                    help='network architecture (only resnet18 used in PiCO)')
parser.add_argument('-j', '--workers', default=8, type=int,  # 降低worker数，避免多卡资源竞争
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=500, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,  # 降低单卡batch size，避免OOM
                    help='mini-batch size per GPU (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate (per GPU)', dest='lr')
parser.add_argument('-lr_decay_epochs', type=str, default='300,400',  # 适配500轮训练的衰减点
                    help='where to decay lr, can be a list')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training (auto-set by torchrun)')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training (auto-set by torchrun)')
parser.add_argument('--dist-url', default='env://', type=str,  # 关键：用环境变量自动配置通信
                    help='url used to set up distributed training (env:// for torchrun)')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend (nccl for multi-GPU)')
parser.add_argument('--seed', default=42, type=int,  # 固定seed，确保训练可复现
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use (auto-set by torchrun, do not specify manually)')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=True,  # 默认启用多进程分布式
                    help='Use multi-processing distributed training (required for multi-GPU)')
parser.add_argument('--num-class', default=10, type=int,
                    help='number of class')
parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')
parser.add_argument('--moco_queue', default=8192, type=int,
                    help='queue size; number of negative samples')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='momentum for updating momentum encoder')
parser.add_argument('--proto_m', default=0.99, type=float,
                    help='momentum for computing the moving average of prototypes')
parser.add_argument('--loss_weight', default=0.5, type=float,
                    help='contrastive loss weight')
parser.add_argument('--conf_ema_range', default='0.95,0.8', type=str,
                    help='pseudo target updating coefficient (phi)')
parser.add_argument('--prot_start', default=80, type=int,
                    help='Start Prototype Updating')
parser.add_argument('--partial_rate', default=0.1, type=float,
                    help='ambiguity level (q)')
parser.add_argument('--hierarchical', action='store_true',
                    help='for CIFAR-100 fine-grained training')


def main():
    args = parser.parse_args()
    args.conf_ema_range = [float(item) for item in args.conf_ema_range.split(',')]
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = [int(it) for it in iterations]  # 简化列表生成
    print(f"Training Args: {args}")

    # 固定随机种子
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('Seeded training enabled: CUDNN deterministic mode is on (may slow down training)')

    # 禁止手动指定GPU（由torchrun自动分配）
    if args.gpu is not None:
        raise ValueError("Do not specify --gpu manually when using torchrun (auto-assigned)")

    # 从环境变量读取分布式配置（torchrun自动设置）
    if args.dist_url == "env://":
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))
        args.rank = int(os.environ.get("RANK", 0))
        args.gpu = int(os.environ.get("LOCAL_RANK", 0))  # 每个进程的本地GPU编号
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # 仅主进程（rank=0）创建实验目录，避免多进程冲突
    if args.rank == 0:
        model_path = 'ds_{ds}_pr_{pr}_lr_{lr}_ep_{ep}_ps_{ps}_lw_{lw}_pm_{pm}_arch_{arch}_heir_{heir}_sd_{seed}'.format(
            ds=args.dataset, pr=args.partial_rate, lr=args.lr,
            ep=args.epochs, ps=args.prot_start, lw=args.loss_weight,
            pm=args.proto_m, arch=args.arch, seed=args.seed,
            heir=args.hierarchical)
        args.exp_dir = os.path.join(args.exp_dir, model_path)
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)
            print(f"Experiment directory created: {args.exp_dir}")

    # 多卡训练：用torchrun启动时，无需mp.spawn（torchrun已管理进程）
    main_worker(args.gpu, args.world_size, args)


def main_worker(gpu, world_size, args):
    cudnn.benchmark = True  # 非确定性加速（关闭seed时生效）
    args.gpu = gpu
    args.world_size = world_size

    # 仅主进程打印GPU分配信息
    if args.rank == 0:
        print(f"Using {args.world_size} GPUs (local rank: {args.gpu})")

    # 初始化分布式进程组（关键修复：适配torchrun环境变量）
    if args.distributed:
        try:
            dist.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
                timeout=datetime.timedelta(seconds=60)  # 延长超时，避免通信超时
            )
            print(f"Distributed initialized: rank {args.rank}/{args.world_size}, GPU {args.gpu}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize distributed: {str(e)}") from e

    # 创建模型（DistributedDataParallel包装）
    print(f"Rank {args.rank}: Creating model '{args.arch}'")
    model = PiCO(args, SupConResNet)
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # 多卡同步：所有进程等待模型初始化完成
        dist.barrier()
        # 包装DDP，find_unused_parameters=True避免未使用参数报错
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
    else:
        model = model.cuda(args.gpu)
        raise NotImplementedError("Only DistributedDataParallel is supported for multi-GPU")

    # 优化器配置（单卡参数，多卡自动同步）
    optimizer = torch.optim.SGD(
        model.parameters(), args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # 恢复 checkpoint（仅主进程加载，多卡自动同步）
    if args.resume and args.rank == 0:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=f'cuda:{args.gpu}')
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            print(f"Checkpoint not found: {args.resume}")
    # 多卡同步：等待主进程加载checkpoint
    if args.distributed:
        dist.barrier()

    # 加载数据集（多卡用DistributedSampler）
    print(f"Rank {args.rank}: Loading dataset {args.dataset}")
    if args.dataset == 'cub200':
        input_size = 224
        train_loader, train_givenY, train_sampler, test_loader = load_cub200(
            input_size=input_size, partial_rate=args.partial_rate,
            batch_size=args.batch_size, distributed=args.distributed
        )
    elif args.dataset == 'cifar10':
        train_loader, train_givenY, train_sampler, test_loader = load_cifar10(
            partial_rate=args.partial_rate, batch_size=args.batch_size,
            distributed=args.distributed
        )
    elif args.dataset == 'cifar100':
        train_loader, train_givenY, train_sampler, test_loader = load_cifar100(
            partial_rate=args.partial_rate, batch_size=args.batch_size,
            hierarchical=args.hierarchical, distributed=args.distributed
        )
    else:
        raise NotImplementedError(f"Unsupported dataset: {args.dataset}")

    # 计算初始置信度（多卡同步数据）
    print(f"Rank {args.rank}: Calculating uniform targets")
    tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
    confidence = train_givenY.float() / tempY
    confidence = confidence.cuda(args.gpu)

    # 损失函数（仅主进程初始化日志）
    loss_fn = partial_loss(confidence)
    loss_cont_fn = SupConLoss()
    logger = None
    if args.rank == 0:
        logger = tb_logger.Logger(
            logdir=os.path.join(args.exp_dir, 'tensorboard'),
            flush_secs=2
        )
        print("\nStart Training\n")

    # 训练主循环
    best_acc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)  # 多卡shuffle同步

        # 调整学习率（所有进程同步执行）
        adjust_learning_rate(args, optimizer, epoch)
        # 训练一轮
        train(train_loader, model, loss_fn, loss_cont_fn, optimizer, epoch, args, logger)
        # 更新伪标签置信度系数
        loss_fn.set_conf_ema_m(epoch, args)
        # 测试一轮
        acc_test = test(model, test_loader, args, epoch, logger)
        # 计算平均最大置信度
        mmc = loss_fn.confidence.max(dim=1)[0].mean().item()

        # 仅主进程保存日志和 checkpoint
        if args.rank == 0:
            # 写入日志
            with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
                f.write(f"Epoch {epoch:3d}: Acc {acc_test:.2f}%, Best Acc {best_acc:.2f}%, "
                        f"LR {optimizer.param_groups[0]['lr']:.6f}, MMC {mmc:.4f}\n")
            # 更新最佳精度并保存 checkpoint
            is_best = acc_test > best_acc
            if is_best:
                best_acc = acc_test
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc
            }, is_best=is_best,
                filename=f"{args.exp_dir}/checkpoint.pth.tar",
                best_file_name=f"{args.exp_dir}/checkpoint_best.pth.tar")


def train(train_loader, model, loss_fn, loss_cont_fn, optimizer, epoch, args, logger):
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    acc_proto = AverageMeter('Acc@Proto', ':2.2f')
    loss_cls_log = AverageMeter('Loss@Cls', ':2.2f')
    loss_cont_log = AverageMeter('Loss@Cont', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_cls, acc_proto, loss_cls_log, loss_cont_log],
        prefix=f"Epoch: [{epoch}]"
    )

    model.train()
    end = time.time()
    start_upd_prot = epoch >= args.prot_start  # 原型更新启动标志

    for i, (images_w, images_s, labels, true_labels, index) in enumerate(train_loader):
        # 测量数据加载时间
        data_time.update(time.time() - end)

        # 数据移到GPU
        images_w = images_w.cuda(args.gpu, non_blocking=True)
        images_s = images_s.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)
        true_labels = true_labels.long().cuda(args.gpu, non_blocking=True)
        index = index.cuda(args.gpu, non_blocking=True)

        # 前向传播
        cls_out, features_cont, pseudo_target_cont, score_prot = model(
            images_w, images_s, labels, args
        )
        batch_size = cls_out.shape[0]
        pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1)

        # 原型更新（仅训练后期启动）
        if start_upd_prot:
