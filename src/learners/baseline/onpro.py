
"""Code adapted from https://github.com/weilllllls/OnPro
"""
import torch
import wandb
import time
import torch.nn as nn
import sys
import logging as lg
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import math
import numpy
import pickle
import itertools
import torch.cuda.amp as amp
import matplotlib.pyplot as plt 

from torch.utils.data import DataLoader
from torchvision import transforms
from copy import deepcopy
from sklearn.metrics import accuracy_score
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, RandomGrayscale, RandomMixUpV2
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.distributions import Categorical
from sync_batchnorm import patch_replication_callback

from src.misc import onpro_transforms as TL
from src.misc.onpro_transforms import Rotation
from src.learners.baseline.base import BaseLearner
from src.utils.losses import SupConLoss
from src.utils import name_match
from src.models.resnet import OCMResnet
from src.models.onproresnet import resnet18, imagenet_resnet18
from src.models.Mamba_onproresnet import M_resnet18
from src.utils.metrics import forgetting_line
from src.utils.utils import get_device
from src.buffers.onprobuf import Buffer
from src.models.etf_classifier import etf_initialize,dot_regression_accuracy, dynamic_etf_initialize, DotRegressionLoss

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import time


device = get_device()

pdist = torch.nn.PairwiseDistance(p=2).cuda()

class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        if name == 'module':
            return self._modules['module']
        else:
            return getattr(self.module, name)

class OPELoss(nn.Module):
    def __init__(self, class_per_task, temperature=0.5, only_old_proto=False):
        super(OPELoss, self).__init__()
        self.class_per_task = class_per_task
        self.temperature = temperature
        self.only_old_proto = only_old_proto

    def cal_prototype(self, z1, z2, y, current_task_id):
        start_i = 0
        end_i = (current_task_id + 1) * self.class_per_task
        dim = z1.shape[1]
        current_classes_mean_z1 = torch.zeros((end_i, dim), device=z1.device)
        current_classes_mean_z2 = torch.zeros((end_i, dim), device=z1.device)
        for i in range(start_i, end_i):
            indices = (y == i)
            if not any(indices):
                continue
            t_z1 = z1[indices]
            t_z2 = z2[indices]

            mean_z1 = torch.mean(t_z1, dim=0)
            mean_z2 = torch.mean(t_z2, dim=0)

            current_classes_mean_z1[i] = mean_z1
            current_classes_mean_z2[i] = mean_z2

        return current_classes_mean_z1, current_classes_mean_z2

    def forward(self, z1, z2, labels, task_id, is_new=False):
        prototype_z1, prototype_z2 = self.cal_prototype(z1, z2, labels, task_id)

        if not self.only_old_proto or is_new:
            nonZeroRows = torch.abs(prototype_z1).sum(dim=1) > 0
            nonZero_prototype_z1 = prototype_z1[nonZeroRows]
            nonZero_prototype_z2 = prototype_z2[nonZeroRows]
        else:
            old_prototype_z1 = prototype_z1[:task_id * self.class_per_task]
            old_prototype_z2 = prototype_z2[:task_id * self.class_per_task]
            nonZeroRows = torch.abs(old_prototype_z1).sum(dim=1) > 0
            nonZero_prototype_z1 = old_prototype_z1[nonZeroRows]
            nonZero_prototype_z2 = old_prototype_z2[nonZeroRows]

        nonZero_prototype_z1 = F.normalize(nonZero_prototype_z1)
        nonZero_prototype_z2 = F.normalize(nonZero_prototype_z2)

        device = nonZero_prototype_z1.device

        class_num = nonZero_prototype_z1.size(0)
        z = torch.cat((nonZero_prototype_z1, nonZero_prototype_z2), dim=0)

        logits = torch.einsum("if, jf -> ij", z, z) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        pos_mask = torch.zeros((2 * class_num, 2 * class_num), dtype=torch.bool, device=device)
        pos_mask[:, class_num:].fill_diagonal_(True)
        pos_mask[class_num:, :].fill_diagonal_(True)

        logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)

        exp_logits = torch.exp(logits) * logit_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positives
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)

        loss = -mean_log_prob_pos.mean()

        return loss, prototype_z1, prototype_z2

# 根据不同类别之间的相似度，对mem中的图像进行有选择性的mixup。大量相似图片数据的分类有利于扩大相似类别之间的决策边界
class AdaptivePrototypicalFeedback(nn.Module):
    def __init__(self, buffer, mixup_base_rate, mixup_p, mixup_lower, mixup_upper, mixup_alpha,
                 class_per_task):
        super(AdaptivePrototypicalFeedback, self).__init__()
        self.buffer = buffer
        self.class_per_task = class_per_task
        self.mixup_base_rate = mixup_base_rate
        self.mixup_p = mixup_p
        self.mixup_lower = mixup_lower
        self.mixup_upper = mixup_upper
        self.mixup_alpha = mixup_alpha
        self.mixup = RandomMixUpV2(p=mixup_p, lambda_val=(mixup_lower, mixup_upper),
                                   data_keys=["input", "class"]).cuda()

    def forward(self, mem_x, mem_y, buffer_batch_size, classes_mean, task_id):
        base_rate = self.mixup_base_rate
        base_sample_num = int(buffer_batch_size * base_rate)

        indices = torch.from_numpy(np.random.choice(mem_x.size(0), base_sample_num, replace=False)).cuda()
        mem_x_base = mem_x[indices]
        mem_y_base = mem_y[indices]

        mem_x_base_mix, mem_y_base_mix = self.mixup(mem_x_base, mem_y_base)

        prob_sample_num = buffer_batch_size - base_sample_num
        if prob_sample_num != 0:
            nonZeroRows = torch.abs(classes_mean).sum(dim=1) > 0
            ZeroRows = torch.abs(classes_mean).sum(dim=1) == 0
            class_num = classes_mean.shape[0]
            nonZero_class = torch.arange(class_num)[nonZeroRows.cpu()]
            Zero_class = torch.arange(class_num)[ZeroRows.cpu()]

            classes_mean = classes_mean[nonZeroRows]

            dis = torch.pdist(classes_mean)  # K*(K-1)/2

            sample_p = F.softmax(1 / dis, dim=0)

            mix_x_by_prob, mix_y_by_prob = self.make_mix_pair(sample_p, prob_sample_num, nonZero_class, Zero_class,
                                                              task_id)

            mem_x = torch.cat([mem_x_base_mix, mix_x_by_prob])
            mem_y_mix = torch.cat([mem_y_base_mix, mix_y_by_prob])

            origin_mem_y, mix_mem_y, mix_lam = mem_y_mix[:, 0], mem_y_mix[:, 1], mem_y_mix[:, 2]
            new_mem_y = (1 - mix_lam) * origin_mem_y + mix_lam * mix_mem_y
            mem_y = new_mem_y
        else:
            mem_x = mem_x_base_mix
            origin_mem_y, mix_mem_y, mix_lam = mem_y_base_mix[:, 0], mem_y_base_mix[:, 1], mem_y_base_mix[:, 2]
            new_mem_y = (1 - mix_lam) * origin_mem_y + mix_lam * mix_mem_y
            mem_y = new_mem_y
            mem_y_mix = mem_y_base_mix

        return mem_x, mem_y, mem_y_mix

    def make_mix_pair(self, sample_prob, prob_sample_num, nonZero_class, Zero_class, current_task_id):
        start_i = 0
        end_i = (current_task_id + 1) * self.class_per_task
        sample_num_per_class_pair = (sample_prob * prob_sample_num).round()
        diff_num = int((prob_sample_num - sample_num_per_class_pair.sum()).item())
        if diff_num > 0:
            add_idx = torch.randperm(sample_num_per_class_pair.shape[0])[:diff_num]
            sample_num_per_class_pair[add_idx] += 1
        elif diff_num < 0:
            reduce_idx = torch.nonzero(sample_num_per_class_pair, as_tuple=True)[0]
            reduce_idx_ = torch.randperm(reduce_idx.shape[0])[:-diff_num]
            reduce_idx = reduce_idx[reduce_idx_]
            sample_num_per_class_pair[reduce_idx] -= 1

        assert sample_num_per_class_pair.sum() == prob_sample_num

        x_indices = torch.arange(self.buffer.x.shape[0])
        y_indices = torch.arange(self.buffer.y.shape[0])
        y = self.buffer.y.cuda()
        _, y = torch.max(y, dim=1)

        class_x_list = []
        class_y_list = []
        class_id_map = {}
        for task_id in range(start_i, end_i):
            if task_id in Zero_class:
                continue
            indices = (y == task_id)
            if not any(indices):
                continue

            class_x_list.append(x_indices[indices.cpu()])
            class_y_list.append(y_indices[indices.cpu()])
            class_id_map[task_id] = len(class_y_list) - 1

        mix_images = []
        mix_labels = []

        for idx, class_pair in enumerate(itertools.combinations(nonZero_class.tolist(), 2)):
            n = int(sample_num_per_class_pair[idx].item())
            if n == 0:
                continue
            first_class_y = class_pair[0]
            second_class_y = class_pair[1]

            if first_class_y not in class_id_map:
                first_class_y = np.random.choice(list(class_id_map.keys()), 1)[0]
                first_class_y = int(first_class_y)
            if second_class_y not in class_id_map:
                second_class_y = np.random.choice(list(class_id_map.keys()), 1)[0]
                second_class_y = int(second_class_y)

            first_class_idx = class_id_map[first_class_y]
            second_class_idx = class_id_map[second_class_y]

            first_class_sample_idx = torch.from_numpy(np.random.choice(class_x_list[first_class_idx].tolist(), n)).long()
            second_class_sample_idx = torch.from_numpy(np.random.choice(class_x_list[second_class_idx].tolist(), n)).long()

            first_class_x = self.buffer.x[first_class_sample_idx].cuda()
            second_class_x = self.buffer.x[second_class_sample_idx].cuda()

            mix_pair, mix_lam = self.mixup_by_input_pair(first_class_x, second_class_x, n)
            mix_y = torch.zeros(n, 3)
            mix_y[:, 0] = first_class_y
            mix_y[:, 1] = second_class_y
            mix_y[:, 2] = mix_lam

            mix_images.append(mix_pair)
            mix_labels.append(mix_y)

        mix_images_by_prob = torch.cat(mix_images).cuda()
        mix_labels_by_prob = torch.cat(mix_labels).cuda()

        return mix_images_by_prob, mix_labels_by_prob

    def mixup_by_input_pair(self, x1, x2, n):
        if torch.rand([]) <= self.mixup_p:
            lam = torch.from_numpy(np.random.beta(self.mixup_alpha, self.mixup_alpha, n)).cuda()
            lam_ = lam.unsqueeze(0).unsqueeze(0).unsqueeze(0).view(-1, 1, 1, 1)
        else:
            lam = 0
            lam_ = 0
        lam = torch.tensor(lam, dtype=x1.dtype)
        lam_ = torch.tensor(lam_, dtype=x1.dtype)
        image = (1 - lam_) * x1 + lam_ * x2
        return image, lam

def Supervised_NT_xent_n(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8):
    """
        Code from OCM : https://github.com/gydpku/OCM
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    """
    device = sim_matrix.device
    labels1 = labels.repeat(2)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)

    sim_matrix = sim_matrix - logits_max.detach()
    B = sim_matrix.size(0) // chunk

    eye = torch.zeros((B * chunk, B * chunk), dtype=torch.bool, device=device)
    eye[:, :].fill_diagonal_(True)
    sim_matrix = torch.exp(sim_matrix / temperature) * (~eye)

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)

    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)
    labels1 = labels1.contiguous().view(-1, 1)

    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)

    loss2 = 2 * torch.sum(Mask1 * sim_matrix) / (2 * B)
    loss1 = torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)

    return loss1 + loss2

def Supervised_NT_xent_uni(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8):
    """
        Code from OCM: https://github.com/gydpku/OCM
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    """

    device = sim_matrix.device
    labels1 = labels.repeat(2)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)

    sim_matrix = sim_matrix - logits_max.detach()
    B = sim_matrix.size(0) // chunk

    sim_matrix = torch.exp(sim_matrix / temperature)
    denom = torch.sum(sim_matrix, dim=1, keepdim=True)

    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)
    labels1 = labels1.contiguous().view(-1, 1)

    Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
    Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)

    return torch.sum(Mask1 * sim_matrix) / (2 * B)

class OnProLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.oop_base = self.params.n_classes
        self.oop = 16
        self.n_classes_num = self.params.n_classes
        self.fea_dim = self.params.proj_dim # 这里为什么是128
        self.classes_mean = torch.zeros((self.n_classes_num, self.fea_dim), requires_grad=False).cuda()
        self.class_per_task = int(self.params.n_classes / self.params.n_tasks)
        self.class_holder = []
        self.ins_t = 0.07
        self.proto_t = 0.5
        self.mem = None

        self.buffer = Buffer(
                mem_size=self.params.mem_size,
                img_size=self.params.img_size,
                nb_ch=self.params.nb_channels,
                n_classes=self.params.n_classes,
                drop_method=self.params.drop_method,
                device=device
            )
        self.buffer_batch_size = 64
        self.buffer_per_class = 7

        self.OPELoss = OPELoss(self.class_per_task, temperature=self.proto_t)

        if self.n_classes_num == 10:
            self.taskid_max = 5
            self.sim_lambda = 0.5
            self.total_samples = 10000
            self.mixup_base_rate = 0.75
            self.mixup_p = 0.6
            # 1
            self.extra = 1 #72.39
            self.mul_ce_loss = 1
            self.mul_etf_loss = 1 * self.extra
            self.mul_diff_loss = 1 * self.extra
            self.mul_cons_loss = 5 * self.extra
            self.mul_load_loss = 1 * self.extra
            self.num_experts = 10
            self.top_k = 1
        elif self.n_classes_num == 100:
            self.taskid_max = 10
            self.sim_lambda = 1.0
            self.total_samples = 5000
            self.mixup_base_rate = 0.9
            self.mixup_p = 0.2
            self.extra = 1/2 #1/2
            self.mul_ce_loss = 1
            self.mul_etf_loss = 10 * self.extra
            self.mul_diff_loss = 10 * self.extra #10
            self.mul_cons_loss = 50 * self.extra #这个参数万万不能动
            self.mul_load_loss = 1 * self.extra
            # 5
            self.num_experts = 5
            self.top_k = 1
        elif self.n_classes_num == 200:
            # e1d1best e1
            # c50best
            self.taskid_max = 100
            self.sim_lambda = 1.0
            self.total_samples = 10000
            self.mixup_base_rate = 0.9
            self.mixup_p = 0.2
            self.extra = 1
            self.top_k = 10
            self.mul_ce_loss = 1
            self.mul_etf_loss = 1 * self.extra
            self.mul_diff_loss = 1 * self.extra
            self.mul_cons_loss = 50  * self.extra
            self.mul_load_loss = 0 * self.extra
            self.num_experts = 1

        # need different parameters for initialization
        self.model = self.load_model()
        self.optim = self.load_optim()

        self.print_num = self.total_samples // 10

        self.oop = 16
        self.iter = 0
        # self.dim_neckout = 512
        self.dim_neckout = 128
        self.use_cons_loss = True
        # etf 
        self.alpha_k = torch.ones(1).cuda()
        self.beta_k = torch.zeros(self.dim_neckout).cuda()
        self.etf_criterion = DotRegressionLoss(reduction="none")
        self.register_buffer("etf_classifier", etf_initialize(self.dim_neckout, self.n_classes_num).cuda()) # 512 10

        self.print_num = self.total_samples // 10

        hflip = TL.HorizontalFlipLayer().cuda()
        with torch.no_grad():
            input_size = [3, self.params.img_size, self.params.img_size]
            resize_scale = (0.3, 1.0)
            color_gray = TL.RandomColorGrayLayer(p=0.25).cuda()
            resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=[input_size[1], input_size[2], input_size[0]]).cuda()
            self.transform = torch.nn.Sequential(
                hflip,
                color_gray,
                resize_crop)

        self.APF = AdaptivePrototypicalFeedback(self.buffer, self.mixup_base_rate, self.mixup_p, 0, 0.6,
                                  0.4, self.class_per_task)

        self.scaler = amp.GradScaler()

    def myprint(self):
        # print("random max!")
        print("mul_ce_loss:", self.mul_ce_loss, flush=True)
        print("mul_etf_loss:", self.mul_etf_loss, flush=True)
        print("mul_diff_loss:", self.mul_diff_loss, flush=True)
        print("mul_cons_loss:", self.mul_cons_loss, flush=True)
        print("mul_load_loss:", self.mul_load_loss, flush=True)
        print("num_experts:", self.num_experts, flush=True)
        print("top_k:", self.top_k, flush=True)
        print("use_cons_loss:", self.use_cons_loss, flush=True)
        print("pro=12")

    # etf
    # ETF变换
    def etf_transform(self, features):
        return self.alpha_k * features + self.beta_k
    # 归一化特征
    def pre_logits(self, x):
        return F.normalize(x, dim=1)

    def load_criterion(self):
        pass

    def load_model(self, **kwargs):
        if self.params.dataset == 'cifar10' or self.params.dataset == 'cifar100' or self.params.dataset == 'tiny':
            model = M_resnet18(nclasses=self.params.n_classes, choose_mamba='mamba modulized', num_experts=self.num_experts, top_k=self.top_k)
            return model.to(device)
        elif self.params.dataset == 'imagenet' or self.params.dataset == 'imagenet100':
            # for imagenet experiments, the 80 gig memory is not enough, so do it in a data parallel way
            #haven't done yet！！！
            model = MyDataParallel(imagenet_resnet18(nclasses=self.params.n_classes))
            patch_replication_callback(model)
            return model.to(device)

    def train(self, dataloader, **kwargs):
        self.model.train()
        if self.params.training_type == "inc":
            self.train_inc(dataloader, **kwargs)
        elif self.params.training_type == "blurry":
            self.train_blurry(dataloader, **kwargs)

    def train_inc(self, dataloader, **kwargs):
        start_time = time.time()
        task_id = kwargs.get('task_id', None)
        dataloaders = kwargs.get('dataloaders', None)
        task_name = kwargs.get('task_name', None)
        present = torch.LongTensor(size=(0,)).to(device)

        if task_id == 0:
            num_d = 0
            for batch_idx, batch in enumerate(dataloader):
                x, y = batch[0], batch[1]
                ybuf = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in y])
                y = ybuf.to(device)
                num_d += x.shape[0]
                loss = 0

                Y = deepcopy(y)
                for j in range(len(Y)):
                    if Y[j] not in self.class_holder: # ??? class_holder是什么
                        self.class_holder.append(Y[j].detach())

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                    # x = x.requires_grad_()

                    rot_x = Rotation(x)
                    rot_x_aug = self.transform(rot_x)
                    images_pair = torch.cat([rot_x, rot_x_aug], dim=0)

                    rot_sim_labels = torch.cat([y + self.oop_base * i for i in range(self.oop)], dim=0)

                    features, projections = self.model(images_pair, is_simclr=True)
                    projections = F.normalize(projections)

                    # instance-wise contrastive loss in OCM
                    features = F.normalize(features)
                    dim_diff = features.shape[1] - projections.shape[1]  # 512 - 128
                    dim_begin = torch.randperm(dim_diff)[0]
                    dim_len = projections.shape[1]
                    sim_matrix = torch.matmul(projections, features[:, dim_begin:dim_begin + dim_len].t())
                    sim_matrix += torch.mm(projections, projections.t())
                    ins_loss = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=self.ins_t)    
                    loss += ins_loss            
                    
                    # # OPEloss##############################################
                    # OPE_loss = 0
                    if batch_idx != 0:
                        buffer_x, buffer_y = self.sample_from_buffer_for_prototypes()
                        # buffer_x.requires_grad = True
                        buffer_x, buffer_y = buffer_x.cuda(), buffer_y.cuda()
                        buffer_x_pair = torch.cat([buffer_x, self.transform(buffer_x)], dim=0)

                        proto_seen_loss, _, _, _ = self.cal_buffer_proto_loss(buffer_x, buffer_y, buffer_x_pair, task_id)
                    else:
                        proto_seen_loss = 0

                    z = projections[:rot_x.shape[0]]
                    zt = projections[rot_x.shape[0]:]
                    proto_new_loss, cur_new_proto_z, cur_new_proto_zt = self.OPELoss(z[:x.shape[0]], zt[:x.shape[0]], y, task_id, True)

                    OPE_loss = proto_new_loss + proto_seen_loss
                    loss += OPE_loss
                    # OPEloss##############################################

                    y_pred_linear = self.model.logits(self.transform(x))
                    ce_loss = F.cross_entropy(y_pred_linear, y)
                    wandb.log({"ce_loss": ce_loss.item()})
                    ce_loss *= self.mul_ce_loss
                    loss += ce_loss

                    #etf_loss#############################
                    features, outputs = self.model(self.transform_train(x), labels=y, is_outputs=True)
                    features = self.pre_logits(self.etf_transform(features))
                    etf_loss = self.etf_criterion(features, self.etf_classifier[:, y].t())
                    etf_loss = etf_loss.mean() #或许可以测试一个不mean的结果
                    etf_loss *= self.mul_etf_loss
                    loss += etf_loss

                    load_balancing_loss = 0
                    # #load_balancing_loss#############################
                    load_balancing_loss = outputs['load_balancing_loss']
                    load_balancing_loss *= self.mul_load_loss
                    loss += load_balancing_loss

                    #diff_loss#############################
                    y_pred_etf = self.etf_predict(features, self.etf_classifier)   
                    diff_loss = F.kl_div(F.log_softmax(y_pred_linear, dim=1), y_pred_etf, reduction='batchmean')
                    diff_loss *= self.mul_diff_loss
                    loss += diff_loss

                    #cons_loss#############################
                    if self.use_cons_loss:
                        cons_loss = self.calculate_cons_loss(outputs, y)
                        cons_loss *= self.mul_cons_loss
                        loss += cons_loss
                    else:
                        cons_loss = 0

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad()
                self.iter += 1
                self.buffer.add_reservoir(x=batch[0].detach(), y=ybuf.detach(), logits=None, t=task_id)

                if num_d % self.print_num == 0 or batch_idx == 1:
                    print('==>>> it: {}, loss: ce {:.2f} + ins {:.4f} + ETF {:.4f} + cons {:.4f} + diff {:.4f} + load {:.4f}= {:.6f}, {}%'
                        .format(batch_idx, ce_loss, ins_loss, etf_loss, cons_loss, diff_loss, load_balancing_loss, loss, 100 * (num_d / self.total_samples)),flush=True)
                    self.myprint()

        else: 
            num_d = 0
            for batch_idx, batch in enumerate(dataloader):
                # self.optim.zero_grad() ？？？？？？？？？？？？？？？？？？？？？？？
                x, y = batch[0], batch[1]
                ybuf = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in y])
                y = ybuf.to(device)
                num_d += x.shape[0]
                loss = 0

                Y = deepcopy(y)
                for j in range(len(Y)):
                    if Y[j] not in self.class_holder:
                        self.class_holder.append(Y[j].detach())

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                    # x = x.requires_grad_()
                    buffer_batch_size = min(self.buffer_batch_size, self.buffer_per_class * len(self.class_holder))

                    ori_mem_x, ori_mem_y, bt = self.buffer.sample(buffer_batch_size, exclude_task=None)
                    ori_mem_x, ori_mem_y = ori_mem_x.to(device), ori_mem_y.to(device)
                    if batch_idx != 0:
                        mem_x, mem_y, mem_y_mix = self.APF(ori_mem_x, ori_mem_y, buffer_batch_size, self.classes_mean, task_id)
                        rot_sim_labels = torch.cat([y + self.oop_base * i for i in range(self.oop)], dim=0)
                        rot_sim_labels_r = torch.cat([mem_y + self.oop_base * i for i in range(self.oop)], dim=0)
                        rot_mem_y_mix = torch.zeros(rot_sim_labels_r.shape[0], 3).cuda()
                        rot_mem_y_mix[:, 0] = torch.cat([mem_y_mix[:, 0] + self.oop_base * i for i in range(self.oop)], dim=0)
                        rot_mem_y_mix[:, 1] = torch.cat([mem_y_mix[:, 1] + self.oop_base * i for i in range(self.oop)], dim=0)
                        rot_mem_y_mix[:, 2] = mem_y_mix[:, 2].repeat(self.oop)
                    else:
                        mem_x = ori_mem_x
                        mem_y = ori_mem_y

                        rot_sim_labels = torch.cat([y + self.oop_base * i for i in range(self.oop)], dim=0)
                        rot_sim_labels_r = torch.cat([mem_y + self.oop_base * i for i in range(self.oop)], dim=0)

                    mem_x = mem_x.requires_grad_()

                    rot_x = Rotation(x)
                    rot_x_r = Rotation(mem_x)
                    rot_x_aug = self.transform(rot_x)
                    rot_x_r_aug = self.transform(rot_x_r)
                    images_pair = torch.cat([rot_x, rot_x_aug], dim=0)
                    images_pair_r = torch.cat([rot_x_r, rot_x_r_aug], dim=0)

                    all_images = torch.cat((images_pair, images_pair_r), dim=0)

                    features, projections = self.model(all_images, is_simclr=True)

                    projections_x = projections[:images_pair.shape[0]]
                    projections_x_r = projections[images_pair.shape[0]:]

                    projections_x = F.normalize(projections_x)
                    projections_x_r = F.normalize(projections_x_r)

                    # instance-wise contrastive loss in OCM
                    #ins_loss###########################################################
                    features_x = F.normalize(features[:images_pair.shape[0]])
                    features_x_r = F.normalize(features[images_pair.shape[0]:])

                    dim_diff = features_x.shape[1] - projections_x.shape[1]
                    dim_begin = torch.randperm(dim_diff)[0]
                    dim_begin_r = torch.randperm(dim_diff)[0]
                    dim_len = projections_x.shape[1]

                    sim_matrix = self.sim_lambda * torch.matmul(projections_x, features_x[:, dim_begin:dim_begin + dim_len].t())
                    sim_matrix_r = self.sim_lambda * torch.matmul(projections_x_r, features_x_r[:, dim_begin_r:dim_begin_r + dim_len].t())

                    sim_matrix += self.sim_lambda * torch.mm(projections_x, projections_x.t())
                    sim_matrix_r += self.sim_lambda * torch.mm(projections_x_r, projections_x_r.t())

                    loss_sim_r = Supervised_NT_xent_uni(sim_matrix_r, labels=rot_sim_labels_r, temperature=self.ins_t)
                    loss_sim = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=self.ins_t)
                    
                    ins_loss = loss_sim_r + loss_sim
                    loss += ins_loss
                    
                    ######################################################################
                    # #OPE_loss#############################################################
                    # OPE_loss = 0
                    buffer_x = ori_mem_x
                    buffer_y = ori_mem_y
                    buffer_x_pair = torch.cat([buffer_x, self.transform(buffer_x)], dim=0)
                    proto_seen_loss, cur_buffer_z1_proto, cur_buffer_z2_proto, cur_buffer_z = self.cal_buffer_proto_loss(buffer_x, buffer_y, buffer_x_pair, task_id)

                    z = projections_x[:rot_x.shape[0]]
                    zt = projections_x[rot_x.shape[0]:]
                    proto_new_loss, cur_new_proto_z, cur_new_proto_zt = self.OPELoss(z[:x.shape[0]], zt[:x.shape[0]], y, task_id, True)

                    OPE_loss = proto_new_loss + proto_seen_loss
                    loss += OPE_loss
                    ########################################################################]
                    #ce_loss###############################################################
                    y_pred_linear = self.model.logits(self.transform(mem_x))

                    if batch_idx != 0:
                        ce_loss = self.loss_mixup(y_pred_linear, mem_y_mix)
                    else:
                        ce_loss = F.cross_entropy(y_pred_linear, mem_y)

                    wandb.log({"ce_loss": ce_loss.item()})
                    ce_loss *= self.mul_ce_loss
                    loss += ce_loss
                    ########################################################################
                    # etf_loss 简单版本
                    # 这里只用了旧数据，导致在新的数据上学的不好
                    if batch_idx != 0:
                        mem_y_mix = mem_y_mix.long()
                        input = self.model(self.transform(mem_x), labels=mem_y_mix[:,0])
                        input = self.pre_logits(self.etf_transform(input))
                        etf_loss = self.etfloss_mixup(input, mem_y_mix)
                    else:
                        mem_y = mem_y.long()
                        input = self.model(self.transform(mem_x), labels=mem_y)
                        input = self.pre_logits(self.etf_transform(input))
                        etf_loss = self.etf_criterion(input, self.etf_classifier[:, mem_y].t())
                        etf_loss = etf_loss.mean()
                    etf_loss *= self.mul_etf_loss
                    loss += etf_loss
                    ########################################################################
                    # # diff_loss   
                    y_pred_etf = self.etf_predict(self.model(self.transform(mem_x)), self.etf_classifier)                    
                    diff_loss = F.kl_div(F.log_softmax(y_pred_linear, dim=1), y_pred_etf, reduction='batchmean')
                    diff_loss *= self.mul_diff_loss
                    loss += diff_loss
                    ########################################################################
                    # cons_loss
                    cons_loss = 0
                    load_balancing_loss = 0
                    load_balancing_loss2 = 0
                    if self.use_cons_loss:
                        _, outputs = self.model(self.transform(x), labels=y, is_outputs=True)
                        cons_loss = self.calculate_cons_loss(outputs, y)
                        cons_loss *= self.mul_cons_loss
                        if batch_idx != 0:
                            labels = mem_y_mix[:,0].long()
                        else:
                            lables = mem_y.long()
                        _, mem_outputs = self.model(self.transform_train(mem_x), labels=lables, is_outputs=True)
                        loss += cons_loss

                        #load_balancing_loss#############################
                        load_balancing_loss = mem_outputs['load_balancing_loss']
                        load_balancing_loss *= self.mul_load_loss
                        loss += load_balancing_loss

                        load_balancing_loss2 = outputs['load_balancing_loss']
                        load_balancing_loss2 *= self.mul_load_loss
                        loss += load_balancing_loss2
                    

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad()
                self.iter += 1

                self.buffer.add_reservoir(x=batch[0].detach(), y=ybuf.detach(), logits=None, t=task_id)

                if num_d % self.print_num == 0 or batch_idx == 1:
                    print('==>>> it: {}, loss: ce {:.2f} + ins {:.4f} + OPE {:.4f} + ETF {:.4f} + cons {:.4f} + diff {:.4f}= {:.6f}, {}%'
                        .format(batch_idx, ce_loss, ins_loss, OPE_loss, etf_loss, cons_loss, diff_loss, loss, 100 * (num_d / self.total_samples)),flush=True)
                    self.myprint()

        # if task_id == self.taskid_max-1:
        # if task_id != -1:
        #     x, y = self.buffer.get_all()
        #     # print("x", x.shape)
        #     x = x[:999].cuda()
        #     y = y[:999].cuda()
        #     x = self.model.after_backbone(self.transform(x))
        #     # print("x", x.shape)
        #     self.plot_embedding_2d(x, y, save_path=f"embedding_plot_onpro_mod_c100v2.pdf")

        end_time = time.time()
        print(f"Training ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Total training time: {end_time - start_time:.2f} seconds")
            

    # 通过模型得到特征后，利用缓冲区中的数据计算原型损失
    def cal_buffer_proto_loss(self, buffer_x, buffer_y, buffer_x_pair, task_id):
        buffer_fea, buffer_z = self.model(buffer_x_pair, is_simclr=True)
        buffer_z_norm = F.normalize(buffer_z)
        buffer_z1 = buffer_z_norm[:buffer_x.shape[0]]
        buffer_z2 = buffer_z_norm[buffer_x.shape[0]:]

        buffer_proto_loss, buffer_z1_proto, buffer_z2_proto = self.OPELoss(buffer_z1, buffer_z2, buffer_y, task_id)
        self.classes_mean = (buffer_z1_proto + buffer_z2_proto) / 2

        return buffer_proto_loss, buffer_z1_proto, buffer_z2_proto, buffer_z_norm

    # 从缓冲区中抽样数据，并返回抽样的特征和标签
    def sample_from_buffer_for_prototypes(self):
        b_num = self.buffer.x.shape[0]
        if b_num <= self.buffer_batch_size:
            buffer_x = self.buffer.x
            buffer_y = self.buffer.y
            _, buffer_y = torch.max(buffer_y, dim=1)
        else:
            buffer_x, buffer_y, _ = self.buffer.sample(self.buffer_batch_size, exclude_task=None)

        return buffer_x, buffer_y

    # 计算基于mixup的损失函数。mixup是一种通过在输入数据和标签之间进行线性插值来生成新样本的数据增强方法，可以提高模型的泛化能力
    def loss_mixup(self, logits, y):
        criterion = F.cross_entropy
        loss_a = criterion(logits, y[:, 0].long(), reduction='none')
        loss_b = criterion(logits, y[:, 1].long(), reduction='none')
        return ((1 - y[:, 2]) * loss_a + y[:, 2] * loss_b).mean()

    def etfloss_mixup(self, features, y):
        criterion = DotRegressionLoss(reduction="none")
        loss_a = criterion(features, self.etf_classifier[:, y[:, 0].long()].t()).mean()
        loss_b = criterion(features, self.etf_classifier[:, y[:, 1].long()].t()).mean()
        return ((1 - self.etf_classifier[:, y[:, 2]]) * loss_a + self.etf_classifier[:, y[:, 2]] * loss_b).mean()

    # 计算预测值，和label一起返回
    def encode_logits(self, dataloader, nbatches=-1):
        i = 0
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(device)
                logits = self.model.logits(self.transform_test(inputs))
                preds = logits.argmax(dim=1)
                preds = torch.tensor([self.params.labels_order[i] for i in preds])
                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat = preds.cpu().numpy()
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat = np.hstack([all_feat, preds.cpu().numpy()])
        return all_feat, all_labels

    # 评估模型在多任务学习场景下的聚类或分类性能
    # ?为什么只用mem的数据训练分类器
    def evaluate_clustering(self, dataloaders, task_id, eval_ema=False, **kwargs):
        with torch.no_grad():
            self.model.eval()

            # Train classifier on labeled data
            step_size = int(self.params.n_classes/self.params.n_tasks)
            mem_representations, mem_labels = self.get_mem_rep_labels(use_proj=self.params.eval_proj)

            mem_labels = torch.tensor([self.params.labels_order[i] for i in mem_labels])
            # UMAP visualization
            # reduction = self.umap_reduction(mem_representations.cpu().numpy())
            # plt.figure()
            # figure = plt.scatter(reduction[:, 0], reduction[:, 1], c=mem_labels, cmap='Spectral', s=1)
            # if not self.params.no_wandb:
            #     wandb.log({
            #         "umap": wandb.Image(figure),
            #         "task_id": task_id
            #     })
            
            classifiers= self.init_classifiers()
            classifiers= self.fit_classifiers(classifiers=classifiers, representations=mem_representations, labels=mem_labels)
            
            accs = []
            representations= {}
            targets= {}
            preds= []
            all_targets = []
            tag = ''

            for j in range(task_id + 1):
                test_representation, test_targets = self.encode_fea(dataloaders[f"test{j}"])
                representations[f"test{j}"] = test_representation
                targets[f"test{j}"] = test_targets

                test_preds= classifiers[0].predict(representations[f'test{j}'])

                acc= accuracy_score(targets[f"test{j}"], test_preds) 

                accs.append(acc)
                # Wandb logs
                if not self.params.no_wandb:
                    preds= np.concatenate([preds, test_preds])
                    all_targets = np.concatenate([all_targets, test_targets])
                    wandb.log({
                        tag + f"ncm_acc_{j}": acc,
                        "task_id": task_id
                    })
            
            # Make confusion matrix
            if not self.params.no_wandb:
                # re-index to have classes in task order
                all_targets = [self.params.labels_order.index(int(i)) for i in all_targets]
                preds= [self.params.labels_order.index(int(i)) for i in preds]
                cm= np.log(1 + confusion_matrix(all_targets, preds))
                fig = plt.matshow(cm)
                wandb.log({
                        tag + f"ncm_cm": fig,
                        "task_id": task_id
                    })

                
            for _ in range(self.params.n_tasks - task_id - 1):
                accs.append(np.nan)

            self.results_clustering.append(accs)
            
            line = forgetting_line(pd.DataFrame(self.results_clustering), task_id=task_id, n_tasks=self.params.n_tasks)
            line = line[0].to_numpy().tolist()
            self.results_clustering_forgetting.append(line)

            self.print_results(task_id)

            return np.nanmean(self.results_clustering[-1]), np.nanmean(self.results_clustering_forgetting[-1])

    def save(self, path):
        lg.debug("Saving checkpoint...")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pth'))

        self.mem = self.buffer.get_all()
        
        torch.save(self.mem, os.path.join(path, 'memory.pth'))
        # with open(os.path.join(path, f"memory.pkl"), 'wb') as memory_file:
            # pickle.dump(self.mem, memory_file)

    def resume(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
        self.mem = torch.load(os.path.join(path, 'memory.pth'))
        # with open(os.path.join(path, f"memory.pkl"), 'rb') as f:
            # self.mem = pickle.load(f)
        # f.close()
        torch.cuda.empty_cache()

    # 从模型的内存中提取所有存储的表示（representations）和对应的标签（labels）
    def get_mem_rep_labels_withmem(self, eval=True, use_proj=False):
        """Compute every representation -labels pairs from memory
        Args:
            eval (bool, optional): Whether to turn the mdoel in evaluation mode. Defaults to True.
        Returns:
            representation - labels pairs
        """
        if eval: 
            self.model.eval()
        mem_imgs, mem_labels = self.mem
        batch_s = 10
        n_batch = len(mem_imgs) // batch_s
        all_reps = []
        for i in range(n_batch):
            mem_imgs_b = mem_imgs[i*batch_s:(i+1)*batch_s].to(self.device)
            mem_imgs_b = self.transform_test(mem_imgs_b)
            mem_representations_b = self.model(mem_imgs_b)
            all_reps.append(mem_representations_b)
        mem_representations= torch.cat(all_reps, dim=0)
        return mem_representations, mem_labels

    def evaluate_etf(self, dataloaders, task_id, eval_ema=False, **kwargs):
        with torch.no_grad():
            self.model.eval()  # Set the model to evaluation mode
            accs = []
            preds = []
            all_targets = []
            tag = '' 
            for j in range(task_id + 1):
                # 获取模型的预测结果和真实标签
                test_preds, test_targets = self.encode_logits_etf(dataloaders[f"test{j}"])
                acc = accuracy_score(test_targets, test_preds)
                accs.append(acc)
                # Wandb logs
                if not self.params.no_wandb:
                    preds = np.concatenate([preds, test_preds])
                    all_targets = np.concatenate([all_targets, test_targets])
                    wandb.log({
                        tag + f"ETFacc_{j}": acc,
                        "task_id": task_id
                    })           
            # todo
            # 绘制混淆矩阵
            # 填充缺失的任务准确率
            for _ in range(self.params.n_tasks - task_id - 1):
                accs.append(np.nan)

            self.results_etf.append(accs)
            
            # 计算遗忘率
            line = forgetting_line(pd.DataFrame(self.results_etf), task_id=task_id, n_tasks=self.params.n_tasks)
            line = line[0].to_numpy().tolist()
            self.results_forgetting_etf.append(line)

            self.print_results(task_id)

            return np.nanmean(self.results_etf[-1]), np.nanmean(self.results_forgetting_etf[-1])

    def encode_logits_etf(self, dataloader, nbatches=-1):
        i = 0
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(device)
                # todo:加入etf_transform
                feat = self.pre_logits(self.etf_transform(self.model(self.transform_test(inputs))))
                scores = torch.matmul(feat, self.etf_classifier) 
                _, preds = torch.max(scores, dim=1)
                preds = torch.tensor([self.params.labels_order[i] for i in preds]) 
                # preds,_  = torch.max(scores, dim=1)

                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat= preds.cpu().numpy()
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat= np.hstack([all_feat, preds.cpu().numpy()])
        return all_feat, all_labels

    def etf_predict(self, feat, target):
        # 将点积值转化为预测结果
        # 假设我们有多个类别，我们希望获取每个类别的点积值：
        # 注意：假设 target 是 ETF 分类器的中心参数矩阵
        all_dots = torch.matmul(feat, target)  # [batch_size, num_classes]
        # 获取预测结果
        y_pred_etf = F.softmax(all_dots, dim=1)  # [batch_size, num_classes]
        return y_pred_etf
        


