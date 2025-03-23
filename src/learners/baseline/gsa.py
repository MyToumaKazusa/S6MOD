"""Code adapted from https://github.com/gydpku/GSA
Fair warning : the original code is one of the worst I've seen.
Sensitive developpers are advised to not click on the above link.
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
import torch.cuda.amp as amp

from torch.utils.data import DataLoader
from torchvision import transforms
from copy import deepcopy
from sklearn.metrics import accuracy_score
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from torch.distributions import Categorical

from src.learners.baseline.ocm import OCMLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.utils import name_match
from src.models.Mamba_onproresnet import M_resnet18
from src.utils.metrics import forgetting_line
from src.utils.utils import get_device
from src.models.etf_classifier import etf_initialize,dot_regression_accuracy, dynamic_etf_initialize, DotRegressionLoss

device = get_device()
scaler = amp.GradScaler()

class GSALearner(OCMLearner):
    def __init__(self, args):
        super().__init__(args)
        self.negative_logits_SUM = None
        self.positive_logits_SUM = None
        self.classes_per_task = self.params.n_classes // self.params.n_tasks
        # I know the variable naming is terrible. Please dont judge me it all comes from the authors terrible code
        # One day I will make it look better but its the best I can do rn
        self.Category_sum = None
        self.class_holder = []
        self.tf_gsa = nn.Sequential(
                        RandomResizedCrop(size=(self.params.img_size, self.params.img_size), scale=(0.6, 1.)),
                        RandomGrayscale(p=0.2)
                    ).to(device)
        self.flip_num=2
        self.extra = 1
        if self.n_classes_num == 10:
            self.taskid_max = 5
            self.total_samples = 10000
            self.extra = 1/10 # 1/10 0.6445
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
            self.extra = 1/10
            self.mul_ce_loss = 1
            self.mul_etf_loss = 10 * self.extra
            self.mul_diff_loss = 10 * self.extra#10
            self.mul_cons_loss = 50 * self.extra#这个参数万万不能动 50
            self.mul_load_loss =1 * self.extra# 1
            self.num_experts = 8
            self.top_k = 1
        elif self.n_classes_num == 200:
            # e1d1best e1
            # c50best
            self.taskid_max = 100
            self.total_samples = 10000
            self.extra = 1/10
            self.mul_ce_loss = 1
            self.mul_etf_loss = 1 * self.extra
            self.mul_diff_loss = 0 * self.extra
            self.mul_cons_loss =  0 * self.extra
            self.mul_load_loss = 0 * self.extra
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

        self.iter = 0

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
        # print("extra:", self.extra, flush=True)

    def etf_transform(self, features):
        return self.alpha_k * features + self.beta_k

    def pre_logits(self, x):
        return F.normalize(x, dim=1)
    
    def RandomFlip(self, x, num=0):
        if not num:
            num=self.flip_num
        x=self.tf_gsa(x)
        X = []

        X.append(x)
        X.append(self.flip_inner(x, 1, 1))

        X.append(self.flip_inner(x, 0, 1))

        X.append(self.flip_inner(x, 1, 0))

        return torch.cat([X[i] for i in range(num)], dim=0)

    def flip_inner(self, x, flip1, flip2):
        num = x.shape[0]
        if x.shape[-1] == 32:
            a = x  # .permute(0,1,3,2)
            a = a.view(num, 3, 2, 16, 32)
            a = a.permute(2, 0, 1, 3, 4)
            s1 = a[0]  # .permute(1,0, 2, 3)#, 4)
            s2 = a[1]  # .permute(1,0, 2, 3)
            if flip1:
                s1 = torch.flip(s1, (3,))  # torch.rot90(s1, 2*rot1, (2, 3))
            if flip2:
                s2 = torch.flip(s2, (3,))  # torch.rot90(s2, 2*rot2, (2, 3))

            s = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2)
            S = s.reshape(num, 3, 32, 32)
        elif x.shape[-1] == 64:
            a = x  # .permute(0,1,3,2)
            a = a.view(num, 3, 2, 32, 64)
            a = a.permute(2, 0, 1, 3, 4)
            s1 = a[0]  # .permute(1,0, 2, 3)#, 4)
            s2 = a[1]  # .permute(1,0, 2, 3)
            if flip1:
                s1 = torch.flip(s1, (3,))
            if flip2:
                s2 = torch.flip(s2, (3,))
            
            s = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2)
            S = s.reshape(num, 3, 64, 64)
        elif x.shape[-1] == 224:
            a = x  # .permute(0,1,3,2)
            a = a.view(num, 3, 2, 112, 224)
            a = a.permute(2, 0, 1, 3, 4)
            s1 = a[0]  # .permute(1,0, 2, 3)#, 4)
            s2 = a[1]  # .permute(1,0, 2, 3)
            if flip1:
                s1 = torch.flip(s1, (3,))
            if flip2:
                s2 = torch.flip(s2, (3,))
            
            s = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2)
            S = s.reshape(num, 3, 224, 224)
        return S
    
    def load_optim(self):
        """Load optimizer for training
        Returns:
            torch.optim: torch optimizer
        """
        if self.params.optim == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        elif self.params.optim == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        elif self.params.optim == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.params.learning_rate,
                momentum=self.params.momentum,
                weight_decay=self.params.weight_decay
                )
        else: 
            raise Warning('Invalid optimizer selected.')
        return optimizer
    
    def load_model(self, **kwargs):
        if self.params.dataset == 'cifar10' or self.params.dataset == 'cifar100' or self.params.dataset == 'tiny':
            model = M_resnet18(nclasses=self.params.n_classes, choose_mamba='mamba modulized', num_experts=self.num_experts, top_k=self.top_k)
            return model.to(device)
        elif self.params.dataset == 'imagenet' or self.params.dataset == 'imagenet100':
            model = ImageNet_GSAResnet(
                head='mlp',
                dim_in=self.params.dim_in,
                dim_int=self.params.dim_int,
                proj_dim=self.params.proj_dim,
                n_classes=self.params.n_classes
            )
        return model.to(device)

    def load_criterion(self):
        return F.cross_entropy
    
    def train(self, dataloader, **kwargs):
        if self.params.training_type == "inc":
            self.train_inc(dataloader, **kwargs)
        elif self.params.training_type == "blurry":
            self.train_blurry(dataloader, **kwargs)

    def train_inc(self, dataloader, **kwargs):
        task_id = kwargs.get('task_id', None)
        task_name = kwargs.get('task_name', None)
        dataloaders = kwargs.get('dataloaders', None)
        new_class_holder = []
        
        if task_id > 0:
            self.Category_sum = torch.cat((self.Category_sum, torch.zeros(self.classes_per_task)))
            self.negative_logits_SUM = torch.cat((self.negative_logits_SUM, torch.zeros(self.classes_per_task).to(device)))
            self.positive_logits_SUM = torch.cat((self.positive_logits_SUM, torch.zeros(self.classes_per_task).to(device)))
            
        negative_logits_sum=None
        positive_logits_sum=None
        sum_num=0
        category_sum = None  
         
        self.model.train()

        num_d = 0
        for batch_idx, batch in enumerate(dataloader):
            # Stream data
            loss = 0
            self.optim.zero_grad()
            num_d += len(batch[0])
            x, y = batch[0].to(device), batch[1].to(device)

            # re-order to adapt GSA code more easily
            y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in y]).to(device)
            
            if not self.buffer.is_empty():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    
                    Y = deepcopy(y)
                    for c in range(len(Y)):
                        if Y[c] not in self.class_holder:
                            self.class_holder.append(Y[c].detach())
                            new_class_holder.append(Y[c].detach())
                    
                    ori_x = x.detach()
                    ori_y = y.detach()
                    x = x.requires_grad_()

                    # cons_loss ##############################
                    cons_loss = 0
                    if self.use_cons_loss:
                        y = y.long()
                        _, outputs = self.model(x, labels=y, is_outputs=True)
                        cons_loss = self.calculate_cons_loss(outputs, y) 
                        cons_loss *= self.mul_cons_loss
                        loss += cons_loss
                    
                    curr_labels = self.params.labels_order[task_id*self.classes_per_task:(task_id+1)*self.classes_per_task]
                    
                    cur_x, cur_y = self.buffer.only_retrieve(n_imgs=22, desired_labels=curr_labels)
                    cur_x = cur_x.to(device)
                    cur_y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in cur_y]).to(device) if len(cur_y) > 0 else cur_y.to(device)
                    
                    x = torch.cat((x, cur_x), dim=0)
                    y = torch.cat((y, cur_y))
                    
                    x = self.RandomFlip(x, num=2)
                    y = y.repeat(2)
                    
                    #!!!
                    # 只保留历史类别的预测结果
                    pred_y = self.model.logits(x)[:, :(task_id+1)*self.classes_per_task]  # Inference 1
                    
                    if task_id>0:
                        # 对于当前任务的预测结果，只取最新任务的类别
                        pred_y_new = pred_y[:, -self.classes_per_task:]
                    else:
                        pred_y_new = pred_y

                    # 标签映射到(0,classes_per_task)的范围
                    y_new = y - self.classes_per_task*task_id
                    rate = len(new_class_holder)/len(self.class_holder)

                    y_new = y_new.long()
                    #ce_loss##############################################
                    ce_loss = F.cross_entropy(pred_y_new, y_new)
                    ce_loss *= self.mul_ce_loss
                    loss += ce_loss

                    # etf_loss ##############################################
                    features, outputs = self.model(x, labels=y, is_outputs=True)
                    features_etf = features
                    features = self.pre_logits(self.etf_transform(features))
                    # 使用裁剪后的 etf_classifier 计算 ETF loss
                    etf_loss = self.etf_criterion(features, self.etf_classifier[:, y].t())
                    etf_loss = etf_loss.mean()  # 或者你可以测试不mean的情况
                    etf_loss *= self.mul_etf_loss
                    # 将 etf_loss 加入总损失
                    loss += etf_loss

                    # load_balancing_loss#############################
                    load_balancing_loss = 0
                    # #load_balancing_loss#############################
                    load_balancing_loss = outputs['load_balancing_loss']
                    load_balancing_loss *= self.mul_load_loss
                    loss += load_balancing_loss

                    # diff_loss ##############################
                    y_pred_linear = self.model.logits(x)
                    y_pred_etf = self.etf_predict(features_etf, self.etf_classifier)
                    diff_loss = F.kl_div(F.log_softmax(y_pred_linear, dim=1), y_pred_etf, reduction='batchmean')
                    diff_loss *= self.mul_diff_loss
                    loss += diff_loss

                    mem_x, mem_y = self.buffer.except_retrieve(int(self.params.mem_batch_size*(1-rate)), undesired_labels=curr_labels)
                    mem_y = torch.stack([torch.tensor(self.params.labels_order.index(int(i))) for i in mem_y]) if len(mem_y) > 0 else mem_y
                    mem_x, mem_y = mem_x.to(device), mem_y.to(device)
                    
                    index_x=ori_x
                    index_y=ori_y
                    if len(cur_x.shape) > 3:
                        index_x = torch.cat((index_x, cur_x), dim=0)
                        index_y = torch.cat((index_y, cur_y))

                    mem_x = torch.cat((mem_x[:int(self.params.mem_batch_size*(1-rate))].to(device),index_x[:int(self.params.mem_batch_size*rate)].to(device)),dim=0)
                    mem_y = torch.cat((mem_y[:int(self.params.mem_batch_size*(1-rate))].to(device),index_y[:int(self.params.mem_batch_size*rate)].to(device)))

                    index = torch.randperm(mem_y.size()[0])
                    mem_x = mem_x[index][:]
                    mem_y = mem_y[index][:]

                    mem_y = mem_y.reshape(-1).long()
                    
                    mem_x = mem_x.requires_grad_()
                    
                    # mem_x = self.transform_train(mem_x)
                    # mem_x = self.tf_gsa(mem_x)
                    mem_x = self.RandomFlip(mem_x, num=2)
                    mem_y = mem_y.repeat(2)
                    
                    #!!!
                    y_pred = self.model.logits(mem_x)[:, :(task_id+1)*self.classes_per_task]  # Inference 2

                    y_pred_new = y_pred

                    exp_new = torch.exp(y_pred_new)
                    exp_new = exp_new
                    exp_new_sum = torch.sum(exp_new, dim=1)
                    logits_new = (exp_new / exp_new_sum.unsqueeze(1))
                    category_matrix_new = torch.zeros(logits_new.shape)
                    for i_v in range(int(logits_new.shape[0])):
                        category_matrix_new[i_v][mem_y[i_v]] = 1
                    positive_prob = torch.zeros(logits_new.shape)
                    false_prob = deepcopy(logits_new.detach())
                    for i_t in range(int(logits_new.shape[0])):
                        false_prob[i_t][mem_y[i_t]] = 0
                        positive_prob[i_t][mem_y[i_t]] = logits_new[i_t][mem_y[i_t]].detach()
                    if negative_logits_sum is None:
                        negative_logits_sum = torch.sum(false_prob, dim=0)
                        positive_logits_sum = torch.sum(positive_prob, dim=0)
                        if task_id == 0:
                            self.Category_sum = torch.sum(category_matrix_new, dim=0)
                        else:
                            self.Category_sum += torch.sum(category_matrix_new, dim=0)

                        category_sum = torch.sum(category_matrix_new, dim=0)
                    else:
                        self.Category_sum += torch.sum(category_matrix_new, dim=0)
                        negative_logits_sum += torch.sum(false_prob, dim=0)
                        positive_logits_sum += torch.sum(positive_prob, dim=0)
                        category_sum += torch.sum(category_matrix_new, dim=0)
                    if self.negative_logits_SUM is None:
                        self.negative_logits_SUM = torch.sum(false_prob, dim=0).to(device)
                        self.positive_logits_SUM = torch.sum(positive_prob, dim=0).to(device)
                    else:
                        self.negative_logits_SUM += torch.sum(false_prob, dim=0).to(device)
                        self.positive_logits_SUM += torch.sum(positive_prob, dim=0).to(device)

                    sum_num += int(logits_new.shape[0])
                    
                    if batch_idx < 5:
                        ANT = torch.ones(len(self.class_holder))
                    else:
                        ANT = (self.Category_sum.to(device) - self.positive_logits_SUM).to(device)/self.negative_logits_SUM.to(device)

                    ttt = torch.zeros(logits_new.shape)
                    for qqq in range(mem_y.shape[0]):
                        if mem_y[qqq]>=len(ANT):
                            ttt[qqq][mem_y[qqq]] = 1
                        else:
                            ttt[qqq][mem_y[qqq]] = 2 / (1+torch.exp(1-(ANT[mem_y[qqq]])))

                    loss_n=-torch.sum(torch.log(logits_new)*ttt.to(device))/mem_y.shape[0]
                    loss_n = loss_n*2
                    loss += loss_n

                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()

                self.iter += 1

                self.loss = loss.item()
                # print(f"Loss {self.loss:.3f}  batch {batch_idx}", end="\r")

                if num_d % self.print_num == 0 or batch_idx == 1:
                    print('==>>> it: {}, loss: ce {:.2f} + ETF {:.4f} + cons {:.4f} + diff {:.4f} + load {:.4f}= {:.6f}, {}%'
                        .format(batch_idx, ce_loss, etf_loss, cons_loss, diff_loss, load_balancing_loss, loss, 100 * (num_d / self.total_samples)),flush=True)
                    self.myprint()

                # Plot to tensorboard
                if (batch_idx == (len(dataloader) - 1)) and (batch_idx > 0):
                    lg.info(
                        f"Phase : {task_name}   batch {batch_idx}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                    )

            # Update buffer
            self.buffer.update(imgs=batch[0].to(device).detach(), labels=batch[1].to(device).detach())
    
    def train_blurry(self, dataloader, **kwargs):
        raise NotImplemented

    # need to rewrite the evaluate function, caz it changes the normal order of the labels    
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
    
    def encode_logits_etf(self, dataloader, nbatches=-1):
        i = 0
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(device)
                feat = self.pre_logits(self.etf_transform(self.model(self.transform_test(inputs))))
                scores = torch.matmul(feat, self.etf_classifier) 
                _, preds = torch.max(scores, dim=1)
                # preds,_  = torch.max(scores, dim=1)
                preds = torch.tensor([self.params.labels_order[i] for i in preds])

                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat= preds.cpu().numpy()
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat= np.hstack([all_feat, preds.cpu().numpy()])
        return all_feat, all_labels