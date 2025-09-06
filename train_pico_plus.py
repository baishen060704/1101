import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn
import torch.backends.cudnn as cudnn
import numpy as np
import tensorboard_logger as tb_logger
from model import PiCO_PLUS
from resnet import *
from utils_plus.utils_algo import *
from utils_plus.utils_loss import partial_loss, SupConLoss, ce_loss
from utils_plus.cub200 import load_cub200
from utils_plus.cifar10 import load_cifar10
from utils_plus.cifar100 import load_cifar100
import copy

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(precision=2, sci_mode=False)

parser = argparse.ArgumentParser(description='PyTorch implementation of PiCO (single GPU version)')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100', 'cub200'],
                    help='dataset name (cifar10)')
parser.add_argument('--exp-dir', default='experiment/PiCO', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=['resnet18'],
                    help='network architecture (only resnet18 used in PiCO)')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=2, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-lr_decay_epochs', type=str, default='700,800,900',
                    help='where to decay lr, can be a list')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=123, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use (-1 for CPU)')
parser.add_argument('--num-class', default=10, type=int,
                    help='number of class')
parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')
parser.add_argument('--moco_queue', default=8192, type=int,
                    help='queue size; number of negative samples')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='momentum for updating momentum encoder')
parser.add_argument('--proto_m', default=0.99, type=float,
                    help='momentum for computing the momving average of prototypes')
parser.add_argument('--loss_weight', default=0.5, type=float,
                    help='contrastive loss weight')
parser.add_argument('--conf_ema_range', default='0.95,0.8', type=str,
                    help='pseudo target updating coefficient (phi)')
parser.add_argument('--prot_start', default=50, type=int,
                    help='Start Prototype Updating')
parser.add_argument('--partial_rate', default=0.1, type=float,
                    help='ambiguity level (q)')

# 噪声等级 让有些样本有noisy_rate的概率候选标签集合里面没有真值
parser.add_argument('--noisy_rate', default=0.2, type=float,
                    help='noisy level')
parser.add_argument('--hierarchical', action='store_true',
                    help='for CIFAR-100 fine-grained training')
# 纯净率
parser.add_argument('--pure_ratio', default='0.6', type=float,
                    help='selection ratio')

parser.add_argument('--knn_start', default=100, type=int,
                    help='when to start kNN')
parser.add_argument('--chosen_neighbors', default=5, type=int,
                    help='chosen neighbors')
parser.add_argument('--temperature_guess', default=0.07, type=float,
                    help='temperature for label guessing')

# 用来计算不可靠样本的损失权重
parser.add_argument('--ur_weight', default='0.1', type=float,
                    help='weights for the losses of unreliable examples')
parser.add_argument('--cls_weight', default=2, type=float,
                    help='weights for the losses of mixup loss')


def main():
    args = parser.parse_args()
    args.conf_ema_range = [float(item) for item in args.conf_ema_range.split(',')]
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    print(args)

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # 设置GPU或CPU
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    model_path = 'ds{ds}p{pr}n{nr}_ps{ps}_lw{lw}_pm{pm}_he_{heir}_sel{sel}_k{k}s{ks}_uw{uw}_sd{seed}'.format(
        ds=args.dataset,
        pr=args.partial_rate,
        ps=args.prot_start,
        lw=args.loss_weight,
        pm=args.proto_m,
        seed=args.seed,
        sel=args.pure_ratio,
        k=args.chosen_neighbors,
        ks=args.knn_start,
        nr=args.noisy_rate,
        uw=args.ur_weight,
        heir=args.hierarchical)
    args.exp_dir = os.path.join(args.exp_dir, model_path)
    # 确保实验目录存在
    os.makedirs(args.exp_dir, exist_ok=True)

    # 创建模型
    print("=> creating model '{}'".format(args.arch))
    model = PiCO_PLUS(args, SupConResNet)
    model = model.to(device)

    # 设置优化器
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # 可选地从检查点恢复
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # 加载数据
    if args.dataset == 'cub200':
        input_size = 224  # 固定为224
        train_loader, train_givenY, test_loader = load_cub200(
            input_size=input_size,
            partial_rate=args.partial_rate,
            noisy_rate=args.noisy_rate,
            batch_size=args.batch_size
        )
    elif args.dataset == 'cifar10':
        train_loader, train_givenY, test_loader = load_cifar10(
            partial_rate=args.partial_rate,
            batch_size=args.batch_size,
            noisy_rate=args.noisy_rate
        )
    elif args.dataset == 'cifar100':
        train_loader, train_givenY, test_loader = load_cifar100(
            partial_rate=args.partial_rate,
            batch_size=args.batch_size,
            hierarchical=args.hierarchical,
            noisy_rate=args.noisy_rate
        )
    else:
        raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")

    print('Calculating uniform targets...')
    # num_instance记录样本数量
    num_instance = train_givenY.shape[0]
    tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
    confidence = train_givenY.float() / tempY
    confidence = confidence.to(device)
    # 计算置信度

    loss_fn = partial_loss(confidence)
    loss_cont_fn = SupConLoss()
    # 设置损失函数

    logger = tb_logger.Logger(logdir=os.path.join(args.exp_dir, 'tensorboard'), flush_secs=2)

    print('\nStart Training\n')

    best_acc = 0
    mmc = 0  # 平均最大置信度
    # dist:样本长度的tensor张量  is_rel:样本长度的布尔型tensor张量(判断哪个最可靠)
    sel_stats = {
        'dist': torch.zeros(num_instance).to(device),
        'is_rel': torch.ones(num_instance).bool().to(device),
    }

    # 用于存储伪标签和真实标签的列表，确保在整个训练过程中可用
    pseudo_labels_list = []
    target_list = []

    for epoch in range(args.start_epoch, args.epochs):
        is_best = False

        adjust_learning_rate(args, optimizer, epoch)
        # 更新prototype 选择可以用来更新的样本集合
        if epoch >= args.prot_start:
            reliable_set_selection(args, epoch, sel_stats)
            # 预热5个epoch后开始选择
        # 将列表作为参数传递，以便在训练过程中累积数据
        train(args, train_loader, model, loss_fn, loss_cont_fn, optimizer, epoch, logger, sel_stats, device,
              pseudo_labels_list, target_list)
        loss_fn.set_conf_ema_m(args, epoch)
        # 重置phi

        acc_test = test(args, model, test_loader, epoch, logger, device)
        mmc = loss_fn.confidence.max(dim=1)[0].mean()

        with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
            f.write('Epoch {}: Acc {}, Best Acc {}. (lr {}, MMC {})\n'.format(epoch
                                                                              , acc_test, best_acc,
                                                                              optimizer.param_groups[0]['lr'], mmc))
            print('success save result.log...')
        if acc_test > best_acc:
            best_acc = acc_test
            is_best = True

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(args.exp_dir),
            best_file_name='{}/checkpoint_best.pth.tar'.format(args.exp_dir))
        print('success save checkpoint.pth.tar/checkpoint_best.pth.tar...')

    # 训练结束后保存伪标签和真实标签，无论是否是最后一个epoch
    if pseudo_labels_list and target_list:
        with open(os.path.join(args.exp_dir, 'pseudo_labels.txt'), 'w') as f:
            for line in pseudo_labels_list:
                for val in line:
                    f.write(str(val) + '\t')
                f.write('\n')
        with open(os.path.join(args.exp_dir, 'labels.txt'), 'w') as f:
            for x in target_list:
                f.write(str(x) + '\n')
        print('success save pseudo_labels and labels...')
    else:
        print('No data to save for pseudo labels and targets')


def train(args, train_loader, model, loss_fn, loss_cont_fn, optimizer, epoch, tb_logger, sel_stats, device,
          pseudo_labels_list, target_list):
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    acc_proto = AverageMeter('Acc@Proto', ':2.2f')
    loss_cls_log = AverageMeter('Loss@Cls', ':2.2f')
    loss_cont_log = AverageMeter('Loss@Cont', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_cls, acc_proto, loss_cls_log, loss_cont_log],
        prefix="Epoch: [{}]".format(epoch))

    # 更新的标志
    start_upd_prot = epoch >= args.prot_start

    # 切换到训练模式
    model.train()
    end = time.time()

    for i, (images_w, images_s, labels, true_labels, index) in enumerate(train_loader):
        # 测量数据加载时间
        data_time.update(time.time() - end)

        X_w, X_s, Y, index = images_w.to(device), images_s.to(device), labels.to(device), index.to(device)
        Y_true = true_labels.long().detach().to(device)
        # 用于显示训练精度，不用于训练
        is_rel = sel_stats['is_rel'][index]
        batch_weight = is_rel.float()

        cls_out, features_cont, pseudo_labels, score_prot, distance_prot, is_rel_queue, target \
            = model(X_w, X_s, Y, Y_cor=None, is_rel=is_rel, args=args)

        # 收集所有epoch的数据，而不仅仅是最后一个
        t1 = time.time()
        # 获取实际的batch size
        actual_batch_size = features_cont.size(0)
        # 确保target的维度正确，取前actual_batch_size个元素
        current_targets = target[:actual_batch_size].cpu().detach().numpy().tolist()

        # 分别添加特征和目标标签
        for feat in features_cont.cpu().detach().numpy().tolist():
            pseudo_labels_list.append(feat)
        for t in current_targets:
            target_list.append(t)
        print('time:', time.time() - t1)

        batch_size = cls_out.shape[0]
        pseudo_target_cont = pseudo_labels.contiguous().view(-1, 1)

        if start_upd_prot:
            loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=Y)
            # 预热结束
            mask_all = torch.eq(pseudo_target_cont[:batch_size], pseudo_target_cont.T).float().to(device)
            loss_cont_all = loss_cont_fn(features=features_cont, mask=mask_all, batch_size=batch_size, weights=None)
            # 这是一个宽松版本，应该分配较低的权重

            mask = copy.deepcopy(mask_all).detach()
            mask = batch_weight.unsqueeze(1).repeat(1, mask.shape[1]) * mask
            # 移除行方向不可靠的掩码
            mask = is_rel_queue.view(1, -1).repeat(mask.shape[0], 1) * mask
            # 移除列方向不可靠的掩码

            # 通过对比预测标签获取正集
            if epoch >= args.knn_start:
                cosine_corr = features_cont[:batch_size] @ features_cont.T
                _, kNN_index = torch.topk(cosine_corr, k=args.chosen_neighbors, dim=-1, largest=True)
                # 最大余弦相关性表示更近的l2距离
                mask_kNN = torch.scatter(torch.zeros(mask.shape).to(device), 1, kNN_index, 1)
                # 上面：通过kNN设置剩余掩码
                mask[~is_rel] = mask_kNN[~is_rel]

            mask[:, batch_size:batch_size * 2] = (
                    (mask[:, batch_size:batch_size * 2] + torch.eye(batch_size).to(device)) > 0).float()
            mask[:, :batch_size] = ((mask[:, :batch_size] + torch.eye(batch_size).to(device)) > 0).float()
            # 重置查询/键的正性

            # 开始knn聚类
            # 对于干净的数据，我们使用原始损失重量。对于不可靠的数据，我们使用较低的权重计算knn-cont损失
            if epoch >= args.knn_start:
                weights = args.loss_weight * batch_weight + args.ur_weight * (1 - batch_weight)
                # 对于干净数据，使用原始损失权重
                # 对于不可靠数据，使用较低权重计算knn-cont损失
                loss_cont_rel_knn = loss_cont_fn(features=features_cont, mask=mask, batch_size=batch_size,
                                                 weights=weights)
                # 联合计算干净/基于knn的对比损失，但为knn损失分配较低的权重
            else:
                loss_cont_rel_knn = loss_cont_fn(features=features_cont, mask=mask, batch_size=batch_size, weights=None)
            # 上面：可靠样本的对比损失

            loss_cont = loss_cont_rel_knn + args.ur_weight * loss_cont_all

            # 分类损失
            loss_cls = loss_fn(cls_out, index, is_rel)

            # 对不可靠样本进行标签猜测
            sp_temp_scale = score_prot ** (1 / args.temperature_guess)
            targets_guess = sp_temp_scale / sp_temp_scale.sum(dim=1, keepdim=True)
            _, loss_cls_ur = ce_loss(cls_out, targets_guess, sel=~is_rel)
            # 不可靠样本的标签猜测

            l = np.random.beta(4, 4)
            l = max(l, 1 - l)
            pseudo_label = loss_fn.confidence[index]
            pseudo_label[~is_rel] = targets_guess[~is_rel]
            # 拼接干净标签和猜测的噪声标签
            idx = torch.randperm(X_w.size(0))
            X_w_rand = X_w[idx]
            pseudo_label_rand = pseudo_label[idx]
            X_w_mix = l * X_w + (1 - l) * X_w_rand
            pseudo_label_mix = l * pseudo_label + (1 - l) * pseudo_label_rand
            logits_mix, _ = model.encoder_q(X_w_mix)
            # 使用encoder q避免DDP错误
            _, loss_mix = ce_loss(logits_mix, targets=pseudo_label_mix)
            # mixup损失

            loss_cls = loss_mix + args.cls_weight * loss_cls + args.ur_weight * loss_cls_ur
            # 我们使用loss_mix作为锚点，因为它使用了所有数据样本
            loss = loss_cls + loss_cont
        else:
            loss_cls = loss_fn(cls_out, index, is_rel=None)
            loss_cont = loss_cont_fn(features=features_cont, mask=None, batch_size=batch_size)
            # 使用MoCo预热

            loss = loss_cls + args.loss_weight * loss_cont
        # 最终损失

        sel_stats['dist'][index] = copy.deepcopy(distance_prot.clone().detach())
        # 更新数据选择的距离

        loss_cls_log.update(loss_cls.item())
        loss_cont_log.update(loss_cont.item())

        # 记录精度
        acc = accuracy(cls_out, Y_true)[0]
        acc_cls.update(acc[0])
        acc = accuracy(score_prot, Y_true)[0]
        acc_proto.update(acc[0])

        # 计算梯度并执行SGD步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 测量经过的时间

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)

    tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
    tb_logger.log_value('Prototype Acc', acc_proto.avg, epoch)
    tb_logger.log_value('Classification Loss', loss_cls_log.avg, epoch)
    tb_logger.log_value('Contrastive Loss', loss_cont_log.avg, epoch)


# 选择最可靠干净的最近的原型向量
def reliable_set_selection(args, epoch, sel_stats):
    dist = sel_stats['dist']
    n = dist.shape[0]
    is_rel = torch.zeros(n).bool().to(dist.device)
    # 按样本与原型向量的距离dist升序返回索引
    sorted_idx = torch.argsort(dist)
    # 根据pure_ratio选择需要的样本数量
    chosen_num = int(n * args.pure_ratio)
    # is_rel表示那些样本需要选
    is_rel[sorted_idx[:chosen_num]] = True
    # 更新sel_stats['is_rel']
    sel_stats['is_rel'] = is_rel
    # 选择接近原型的样本作为可靠和干净的


def test(args, model, test_loader, epoch, tb_logger, device):
    with torch.no_grad():
        print('==> Evaluation...')
        model.eval()
        top1_acc = AverageMeter("Top1")
        top5_acc = AverageMeter("Top5")
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, args=args, eval_only=True)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1_acc.update(acc1[0])
            top5_acc.update(acc5[0])

        print('Accuracy is %.2f%% (%.2f%%)' % (top1_acc.avg, top5_acc.avg))
        tb_logger.log_value('Top1 Acc', top1_acc.avg, epoch)
        tb_logger.log_value('Top5 Acc', top5_acc.avg, epoch)
    return top1_acc.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def Pico_TSNE(data, target, args):
    """画出特征投影图"""
    t_sne_features = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(data)
    plt.scatter(x=t_sne_features[:, 0], y=t_sne_features[:, 1], c=target, cmap='jet')
    plt.savefig(os.path.join(args.exp_dir, 'tsne.pdf'), dpi=800)
    plt.show()


def plot_TSNE(args):
    # 检查文件是否存在
    pseudo_labels_path = os.path.join(args.exp_dir, 'pseudo_labels.txt')
    labels_path = os.path.join(args.exp_dir, 'labels.txt')

    if not os.path.exists(pseudo_labels_path) or not os.path.exists(labels_path):
        print(f"Warning: Could not find {pseudo_labels_path} or {labels_path}, skipping TSNE plot.")
        return

    with open(pseudo_labels_path, 'r') as f:
        p_lines = f.readlines()
    with open(labels_path, 'r') as f:
        l_lines = f.readlines()
    x_list = []
    y_list = []
    for line in p_lines:
        line = line.strip('\t')
        line = line.strip('\n')
        line = line.strip('')
        tem_list = []
        for x in line.split('\t'):
            if x != '':
                tem_list.append(float(x))
        if len(tem_list) == args.low_dim:  # 使用参数中的低维数，更通用
            x_list.append(np.array(tem_list))

    for target in l_lines:
        if target.strip():  # 跳过空行
            y_list.append(int(target.strip()))

    if x_list and y_list:  # 确保有数据
        x_list = np.array(x_list)
        y_list = np.array(y_list)
        Pico_TSNE(x_list, y_list, args)
    else:
        print("No valid data for TSNE plot.")


if __name__ == '__main__':
    args = parser.parse_args()
    main()
    plot_TSNE(args=args)