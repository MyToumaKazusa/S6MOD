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
import torchvision

from sklearn.metrics import accuracy_score

from src.learners.baseline.base import BaseLearner
from src.buffers.logits_res import LogitsRes
from src.models.Mamba_onproresnet import M_resnet18
from src.utils.metrics import forgetting_line
from src.models.etf_classifier import etf_initialize,dot_regression_accuracy, dynamic_etf_initialize, DotRegressionLoss
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from src.utils.utils import get_device

device = get_device()

class DERppLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.results = []
        self.results_forgetting = []
        self.buffer = LogitsRes(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method
        )
        self.iter = 0
        self.n_classes_num = self.params.n_classes
        if self.n_classes_num == 10:
            self.taskid_max = 5
            self.total_samples = 10000
            self.extra = 1 # 0.6139 
            self.mul_ce_loss = 1
            self.mul_etf_loss = 1 * self.extra
            self.mul_diff_loss = 1 * self.extra
            self.mul_cons_loss = 5 * self.extra
            self.mul_load_loss = 1 * self.extra
            self.num_experts = 10
            self.top_k = 1
        elif self.n_classes_num == 100:
            self.taskid_max = 10
            self.total_samples = 5000
            self.extra = 1 
            self.mul_ce_loss = 1
            self.mul_etf_loss = 1 * self.extra
            self.mul_diff_loss = 1 * self.extra #10
            self.mul_cons_loss = 5 * self.extra #这个参数万万不能动
            self.mul_load_loss = 1 * self.extra
            self.num_experts = 8
            self.top_k = 1
        elif self.n_classes_num == 200:
            # e1d1best e1
            # c50best
            self.taskid_max = 100
            self.total_samples = 10000
            self.extra = 1/2
            self.mul_ce_loss = 1
            self.mul_etf_loss = 1 * self.extra
            self.mul_diff_loss = 1 * self.extra
            self.mul_cons_loss = 50  * self.extra
            self.mul_load_loss = 1 * self.extra
            self.num_experts = 10
            self.top_k = 10
        self.model = self.load_model()
        self.optim = self.load_optim()
        self.dim_neckout = 512
        self.print_num = self.total_samples // 10
        self.use_cons_loss = True
        # etf 
        self.alpha_k = torch.ones(1).cuda()
        self.beta_k = torch.zeros(self.dim_neckout).cuda()
        self.etf_criterion = DotRegressionLoss(reduction="none")
        self.register_buffer("etf_classifier", etf_initialize(self.dim_neckout, self.n_classes_num).cuda()) # 512 10
        self.myprint()

    def myprint(self):
        # print("random max!")
        print("extra:", self.extra, flush=True)
        print("mul_ce_loss:", self.mul_ce_loss, flush=True)
        print("mul_etf_loss:", self.mul_etf_loss, flush=True)
        print("mul_diff_loss:", self.mul_diff_loss, flush=True)
        print("mul_cons_loss:", self.mul_cons_loss, flush=True)
        print("mul_load_loss:", self.mul_load_loss, flush=True)
        print("num_experts:", self.num_experts, flush=True)
        print("top_k:", self.top_k, flush=True)
        print("use_cons_loss:", self.use_cons_loss, flush=True)
        print("pro=12")

    def etf_transform(self, features):
        return self.alpha_k * features + self.beta_k
    # 归一化特征
    def pre_logits(self, x):
        return F.normalize(x, dim=1)

    def load_model(self):
        if self.params.dataset == 'cifar10' or self.params.dataset == 'cifar100' or self.params.dataset == 'tiny':
            model = M_resnet18(nclasses=self.params.n_classes, choose_mamba='mamba modulized', num_experts=self.num_experts, top_k=self.top_k)
            return model.to(device)
        elif self.params.dataset == 'imagenet' or self.params.dataset == 'imagenet100':
            return ImageNet_ResNet18(
                dim_in=self.params.dim_in,
                nclasses=self.params.n_classes,
                nf=self.params.nf
            ).to(device)

    def load_criterion(self):
        return F.cross_entropy
    
    def train(self, dataloader, **kwargs):
        self.model = self.model.train()
        if self.params.training_type == 'inc':
            self.train_inc(dataloader, **kwargs)
        elif self.params.training_type == "uni":
            self.train_uni(dataloader, **kwargs)
        elif self.params.training_type == "blurry":
            self.train_blurry(dataloader, **kwargs)

    def train_uni(self, dataloader, **kwargs):
        raise NotImplementedError
        
    def train_inc(self, dataloader, task_name, **kwargs):
        """Adapted from https://github.com/aimagelab/mammoth/blob/master/models/derpp.py
        """
        task_id = kwargs.get('task_id', 0)
        dataloaders = kwargs.get('dataloaders', None)
        num_d = 0
        for batch_idx, batch in enumerate(dataloader):
            self.model = self.model.train()
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            num_d += len(batch[0])
            loss = 0
            
            for _ in range(self.params.mem_iters):
                self.optim.zero_grad()
                batch_y = batch_y.long()
                batch_x = batch_x.to(device)
                y_pred_linear = self.model.logits(self.transform_train(batch_x))
                loss += self.criterion(y_pred_linear, batch_y.to(device))
        
                # etf_loss
                features = self.model(self.transform_train(batch_x), labels=batch_y)
                features = self.pre_logits(self.etf_transform(features))
                etf_loss1 = self.etf_criterion(features, self.etf_classifier[:, batch_y].t())
                etf_loss1 = etf_loss1.mean()
                etf_loss1 *= self.mul_etf_loss
                loss += etf_loss1
                # diff_loss
                y_pred_etf = self.etf_predict(features, self.etf_classifier)
                diff_loss1 = F.kl_div(F.log_softmax(y_pred_linear, dim=1), y_pred_etf, reduction='batchmean')
                diff_loss1 *= self.mul_diff_loss
                loss += diff_loss1

                # cons_loss
                cons_loss = 0
                load_balancing_loss = 0
                if self.use_cons_loss:
                    _, cons_outputs = self.model(self.transform_train(batch_x), labels=batch_y, is_outputs=True)
                    cons_loss = self.calculate_cons_loss(cons_outputs, batch_y) # 可做消融
                    loss += cons_loss

                    #load_balancing_loss#############################
                    load_balancing_loss = cons_outputs['load_balancing_loss']
                    load_balancing_loss *= self.mul_load_loss
                    loss += load_balancing_loss


                mem_x, _, mem_logits = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                if mem_x.size(0) > 0:
                    mem_outputs = self.model.logits(self.transform_train(mem_x.to(device)))
                    loss += self.params.derpp_alpha * F.mse_loss(mem_outputs, mem_logits.to(device))

                    mem_x, mem_y, _ = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    mem_x = mem_x.to(device)
                    y_pred_linear = self.model.logits(self.transform_train(mem_x.to(device)))
                    mem_y = mem_y.long().to(device)
                    ce_loss_mem = self.params.derpp_beta * self.criterion(y_pred_linear, mem_y)
                    loss += ce_loss_mem
                    # etf_loss
                    features = self.model(self.transform_train(mem_x), labels=mem_y)
                    features = self.pre_logits(self.etf_transform(features))
                    etf_loss = self.etf_criterion(features, self.etf_classifier[:, mem_y].t())
                    etf_loss = etf_loss.mean()
                    etf_loss *= self.mul_etf_loss
                    loss += etf_loss
                    # diff_loss
                    y_pred_etf = self.etf_predict(features, self.etf_classifier)
                    diff_loss = F.kl_div(F.log_softmax(y_pred_linear, dim=1), y_pred_etf, reduction='batchmean')
                    diff_loss *= self.mul_diff_loss
                    loss += diff_loss

                loss.backward()
                self.optim.step()

                self.iter += 1

            # Update buffer
            self.buffer.update(imgs=batch_x.detach(), labels=batch_y.detach(), logits=y_pred_linear.detach())

            if num_d % self.print_num == 0 or batch_idx == 1:
                    print('==>>> it: {}, loss: ce {:.2f} + ETF {:.4f} + cons {:.4f} + diff {:.4f} + load1 {:.4f} = {:.6f}, {}%'
                        .format(batch_idx, ce_loss_mem, etf_loss, cons_loss, diff_loss, load_balancing_loss, loss, 100 * (num_d / self.total_samples)),flush=True)
                    self.myprint()

    def train_blurry(self, dataloader, **kwargs):
        raise NotImplementedError
    