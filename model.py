import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# 移除分布式相关导入

# 修正：单GPU版本的concat_all_gather，直接返回输入张量
@torch.no_grad()
def concat_all_gather(tensor):
    """单GPU环境下无需聚集，直接返回张量"""
    return tensor


class PiCO(nn.Module):

    def __init__(self, args, base_encoder):
        super().__init__()

        pretrained = args.dataset == 'cub200'
        # 定义两个Encoder
        self.encoder_q = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch,
                                      pretrained=pretrained)
        # momentum encoder
        self.encoder_k = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch,
                                      pretrained=pretrained)

        # encoder_k的参数不可梯度更新
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # 初始化
            param_k.requires_grad = False  # 不通过梯度更新

        # 创建各条队列 /负样本/伪标签/ptr(队列指针)/原型向量
        self.register_buffer("queue", torch.randn(args.moco_queue, args.low_dim))
        self.register_buffer("queue_pseudo", torch.randn(args.moco_queue))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("prototypes", torch.zeros(args.num_class, args.low_dim))
        self.queue = F.normalize(self.queue, dim=0)

    # 动量更新encoder_k
    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """更新动量编码器"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)

    # 负样本入列与出列操作
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, args):
        # 单GPU无需聚集数据
        batch_size = keys.shape[0]

        # 判断队列的大小是否为batch大小的整数倍
        ptr = int(self.queue_ptr)
        assert args.moco_queue % batch_size == 0  # 为简单起见

        # 以指针的方式替换keys
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_pseudo[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % args.moco_queue  # 更新指针

        self.queue_ptr[0] = ptr

    # 修正：单GPU版本不需要打乱batch
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """单GPU环境下无需打乱batch"""
        return x, torch.arange(x.size(0), device=x.device)

    # 修正：单GPU版本不需要恢复batch顺序
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """单GPU环境下无需恢复batch顺序"""
        return x

    # 模型前向传播
    def forward(self, img_q, im_k=None, partial_Y=None, args=None, eval_only=False):

        output, q = self.encoder_q(img_q)
        if eval_only:
            return output
        # 测试模式

        # 通过伪标签的概率分布softmax得到 样本伪标签pseudo_labels_b
        predicted_scores = torch.softmax(output, dim=1) * partial_Y
        max_scores, pseudo_labels_b = torch.max(predicted_scores, dim=1)
        # 使用部分标签过滤负标签

        # 计算原型logits
        prototypes = self.prototypes.clone().detach()
        logits_prot = torch.mm(q, prototypes.t())
        score_prot = torch.softmax(logits_prot, dim=1)

        # 用伪标签动量更新原型（单GPU版本）
        for feat, label in zip(q, pseudo_labels_b):
            self.prototypes[label] = self.prototypes[label] * args.proto_m + (1 - args.proto_m) * feat
        # 归一化原型
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

        # 计算key特征（恢复BN优化）
        with torch.no_grad():  # 无梯度
            self._momentum_update_key_encoder(args)  # 更新动量编码器
            # 单GPU无需打乱批次
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            _, k = self.encoder_k(im_k)
            # 单GPU无需恢复顺序
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)
        # 用于计算SupCon Loss

        # 入队和出队
        self._dequeue_and_enqueue(k, pseudo_labels_b, args)

        return output, features, pseudo_labels, score_prot, pseudo_labels_b


class PiCO_PLUS(PiCO):
    '''PiCO+是PiCO的扩展，能够缓解嘈杂的部分标签学习问题'''

    def __init__(self, args, base_encoder):
        super().__init__(args, base_encoder)
        # 相关性队列
        self.register_buffer("queue_rel", torch.zeros(args.moco_queue, dtype=torch.bool))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, is_rel, args):
        # 单GPU版本入队操作
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # 调用父类方法更新queue和queue_pseudo
        super()._dequeue_and_enqueue(keys, labels, args)
        # 更新相关性队列
        self.queue_rel[ptr:ptr + batch_size] = is_rel
        # 更新指针（父类已经更新，这里不需要重复）

    def forward(self, img_q, im_k=None, Y_ori=None, Y_cor=None, is_rel=None, args=None, eval_only=False):

        output, q = self.encoder_q(img_q)
        if eval_only:
            return output
        # 测试模式

        batch_weight = is_rel.float()
        with torch.no_grad():  # 无梯度
            predicted_scores = torch.softmax(output, dim=1)
            # 一个batch里面的所有类别
            _, within_max_cls = torch.max(predicted_scores * Y_ori, dim=1)
            _, all_max_cls = torch.max(predicted_scores, dim=1)
            # 保留加权逻辑，删除无效覆盖
            pseudo_labels_b = batch_weight * within_max_cls + (1 - batch_weight) * all_max_cls
            pseudo_labels_b = pseudo_labels_b.long()
            # 对于干净数据，使用部分标签过滤负标签
            # 对于噪声数据，我们启用全套伪标签选择

            # 计算原型logits
            prototypes = self.prototypes.clone().detach()
            logits_prot = torch.mm(q, prototypes.t())
            score_prot = torch.softmax(logits_prot, dim=1)

            # 使用分类器预测的原始集合原型内的距离来检测候选标签集是否有噪声
            _, within_max_cls_ori = torch.max(predicted_scores * Y_ori, dim=1)
            distance_prot = - (q * prototypes[within_max_cls_ori]).sum(dim=1)

            # 用伪标签动量更新原型（仅使用可靠样本）
            for feat, label in zip(q[is_rel], pseudo_labels_b[is_rel]):
                self.prototypes[label] = self.prototypes[label] * args.proto_m + (1 - args.proto_m) * feat
            # 归一化原型
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

            # 计算key特征（复用父类的BN优化）
            self._momentum_update_key_encoder(args)  # 更新动量编码器
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            _, k = self.encoder_k(im_k)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)
        is_rel_queue = torch.cat((is_rel, is_rel, self.queue_rel.clone().detach()), dim=0)
        # 用于计算SupCon Loss

        # 入队和出队
        self._dequeue_and_enqueue(k, pseudo_labels_b, is_rel, args)

        return output, features, pseudo_labels, score_prot, distance_prot, is_rel_queue, within_max_cls