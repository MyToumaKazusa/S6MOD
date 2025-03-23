"""Code adapted from https://github.com/gydpku/OCM
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


from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import transforms
from copy import deepcopy
from sklearn.metrics import accuracy_score
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, RandomGrayscale
from sync_batchnorm import patch_replication_callback


from src.learners.baseline.base import BaseLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.utils import name_match
from src.models.Mamba_onproresnet import M_resnet18
from src.utils.metrics import forgetting_line
from src.utils.utils import get_device
from src.models.etf_classifier import etf_initialize,dot_regression_accuracy, dynamic_etf_initialize, DotRegressionLoss

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import time


device = get_device()
scaler = amp.GradScaler()

class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        if name == 'module':
            return self._modules['module']
        else:
            return getattr(self.module, name)

class OCMLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = name_match.buffers[self.params.buffer](
                max_size=self.params.mem_size,
                img_size=self.params.img_size,
                nb_ch=self.params.nb_channels,
                n_classes=self.params.n_classes,
                drop_method=self.params.drop_method,
            )
        # When task id need to be infered
        self.n_classes_num = self.params.n_classes
        self.old_classes = torch.LongTensor(size=(0,)).to(device)
        self.lag_task_change = 100
        

        if self.n_classes_num == 10:
            self.taskid_max = 5
            self.total_samples = 10000
            self.mul_ce_loss = 1
            # self.mul_etf_loss = 1
            # self.mul_diff_loss = 1
            # self.mul_cons_loss = 5
            # self.mul_load_loss = 1
            self.mul_etf_loss = 0
            self.mul_diff_loss = 0
            self.mul_cons_loss = 0
            self.mul_load_loss = 0
            self.num_experts = 10
            self.top_k = 1
        elif self.n_classes_num == 100:
            self.taskid_max = 10
            self.total_samples = 5000
            self.mul_ce_loss = 1
            self.extra = 0 # 1
            self.mul_etf_loss = 10 * self.extra
            self.mul_diff_loss = 10 * self.extra  #10
            self.mul_cons_loss = 50 * self.extra #这个参数万万不能动
            self.mul_load_loss = 1 * self.extra
            self.num_experts = 8
            # self.mul_etf_loss = 0
            # self.mul_diff_loss = 0 #10
            # self.mul_cons_loss = 0 #这个参数万万不能动
            # self.mul_load_loss = 0
            self.top_k = 1
        elif self.n_classes_num == 200:
            # e1d1best e1
            # c50best
            # self.taskid_max = 100
            # self.total_samples = 10000
            # self.mul_ce_loss = 1
            # self.mul_etf_loss = 1
            # self.mul_diff_loss = 1
            # self.mul_cons_loss = 50 
            # self.mul_load_loss = 1
            # self.num_experts = 10
            self.taskid_max = 100
            self.total_samples = 10000
            # 1/2 19.8
            # 1/5 18.96
            # 3 18.07
            self.extra = 1/2 # 1/2
            self.mul_ce_loss = 1
            self.mul_etf_loss = 1 * self.extra
            self.mul_diff_loss = 1 * self.extra
            self.mul_cons_loss = 10 * self.extra
            self.mul_load_loss = 1 * self.extra
            self.num_experts = 10
            if self.params.mem_size == 5000:
                self.extra = 1/5
                self.mul_ce_loss = 1
                self.mul_etf_loss = 1 * self.extra
                self.mul_diff_loss = 1 * self.extra
                self.mul_cons_loss = 50 * self.extra
                self.mul_load_loss = 1 * self.extra
                self.num_experts = 10
            self.top_k = 10

        # need different parameters for initialization
        self.model = self.load_model()
        self.optim = self.load_optim()

        self.print_num = self.total_samples // 10

        self.oop = 16
        self.iter = 0
        self.dim_neckout = 512
        self.use_cons_loss = True
        # etf 
        self.alpha_k = torch.ones(1).cuda()
        self.beta_k = torch.zeros(self.dim_neckout).cuda()
        self.etf_criterion = DotRegressionLoss(reduction="none")
        self.register_buffer("etf_classifier", etf_initialize(self.dim_neckout, self.n_classes_num).cuda()) # 512 10

        # self.task_classes = {}

        self.myprint()

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

    def etf_transform(self, features):
        return self.alpha_k * features + self.beta_k
    # 归一化特征
    def pre_logits(self, x):
        return F.normalize(x, dim=1)
        
    def rotation(self, x):
        X = self.rot_inner_all(x)#, 1, 0)
        return torch.cat((X,torch.rot90(X,2,(2,3)),torch.rot90(X,1,(2,3)),torch.rot90(X,3,(2,3))),dim=0)

    def rot_inner_all(self, x):
        num=x.shape[0]
        R=x.repeat(4,1,1,1)
        a=x.permute(0,1,3,2)
        a = a.view(num,3, 2, self.params.img_size // 2 , self.params.img_size)
        a = a.permute(2,0, 1, 3, 4)
        s1=a[0]#.permute(1,0, 2, 3)#, 4)
        s2=a[1]#.permute(1,0, 2, 3)
        a= torch.rot90(a, 2, (3, 4))
        s1_1=a[0]#.permute(1,0, 2, 3)#, 4)
        s2_2=a[1]#.permute(1,0, 2, 3)R[3*num:]

        R[num:2*num] = torch.cat((s1_1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num,3, self.params.img_size, self.params.img_size).permute(0,1,3,2)
        R[3*num:] = torch.cat((s1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num,3, self.params.img_size, self.params.img_size).permute(0,1,3,2)
        R[2 * num:3 * num] = torch.cat((s1_1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num,3, self.params.img_size, self.params.img_size).permute(0,1,3,2)
        return R

    def load_optim(self):
        """Load optimizer for training
        Returns:
            torch.optim: torch optimizer
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            betas=(0.9, 0.99),
            weight_decay=1e-4
            )
        return optimizer

    def load_model(self, **kwargs):
        if self.params.dataset == 'cifar10' or self.params.dataset == 'cifar100' or self.params.dataset == 'tiny':
            model = M_resnet18(nclasses=self.params.n_classes, choose_mamba='mamba modulized', num_experts=self.num_experts, top_k=self.top_k)
            return model.to(device)
        elif self.params.dataset == 'imagenet' or self.params.dataset == 'imagenet100':
            # for imagenet experiments, the 80 gig memory is not enough, so do it in a data parallel way
            model = MyDataParallel(ImageNet_OCMResnet(
                head='mlp',
                dim_in=self.params.dim_in,
                dim_int=self.params.dim_int,
                proj_dim=self.params.proj_dim,
                n_classes=self.params.n_classes
            ))
            patch_replication_callback(model)
            return model.to(device)

    def load_criterion(self):
        return SupConLoss(self.params.temperature) 
    
    def normalize(self, x, dim=1, eps=1e-8):
        return x / (x.norm(dim=dim, keepdim=True) + eps)
    
    def get_similarity_matrix(self, outputs, chunk=2, multi_gpu=False):
        '''
            Compute similarity matrix
            - outputs: (B', d) tensor for B' = B * chunk
            - sim_matrix: (B', B') tensor
        '''
        sim_matrix = torch.mm(outputs, outputs.t())  # (B', d), (d, B') -> (B', B')#这里是sim(z(x),z(x'))
        return sim_matrix
    
    def Supervised_NT_xent_n(self, sim_matrix, labels, embedding=None,temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
        '''
            Compute NT_xent loss
            - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
        '''
        labels1 = labels.repeat(2)
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()
        B = sim_matrix.size(0) // chunk  # B = B' / chunk
        eye = torch.eye(B * chunk).to(device)  # (B', B')
        sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal
        denom = torch.sum(sim_matrix, dim=1, keepdim=True)
        sim_matrix = -torch.log(sim_matrix/(denom+eps)+eps)  # loss matrix
        labels1 = labels1.contiguous().view(-1, 1)
        Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
        Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
        loss1 = 2*torch.sum(Mask1 * sim_matrix) / (2 * B)
        return (torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)) +  loss1#+1*loss2
    
    def Supervised_NT_xent_uni(self, sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
        '''
            Compute NT_xent loss
            - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
        '''
        labels1 = labels.repeat(2)
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()
        B = sim_matrix.size(0) // chunk  # B = B' / chunk
        sim_matrix = torch.exp(sim_matrix / temperature)# * (1 - eye)  # remove diagonal
        denom = torch.sum(sim_matrix, dim=1, keepdim=True)
        sim_matrix = -torch.log(sim_matrix/(denom+eps)+eps)  # loss matrix
        labels1 = labels1.contiguous().view(-1, 1)
        Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
        Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
        return torch.sum(Mask1 * sim_matrix) / (2 * B)

    def Supervised_NT_xent_pre(self, sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
        '''
            Compute NT_xent loss
            - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
        '''
        labels1 = labels#.repeat(2)
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()
        B = sim_matrix.size(0) // chunk  # B = B' / chunk
        sim_matrix = torch.exp(sim_matrix / temperature) #* (1 - eye)  # remove diagonal
        denom = torch.sum(sim_matrix, dim=1, keepdim=True)
        sim_matrix = -torch.log(sim_matrix/(denom+eps)+eps)  # loss matrix
        labels1 = labels1.contiguous().view(-1, 1)
        Mask1 = torch.eq(labels1, labels1.t()).float().to(device)
        Mask1 = Mask1 / (Mask1.sum(dim=1, keepdim=True) + eps)
        return torch.sum(Mask1 * sim_matrix) / (2 * B)

    def train(self, dataloader, **kwargs):
        if self.params.training_type == "inc":
            self.train_inc(dataloader, **kwargs)
        elif self.params.training_type == "blurry":
            self.train_blurry(dataloader, **kwargs)

    def generate_task_mask(batch_y, task_labels, device):
        # 创建一个全零的mask，形状为 (batch_size, num_tasks)
        task_mask = torch.zeros((batch_y.size(0), 10), dtype=torch.float).to(device)
        
        # 将属于当前任务的类别位置设为1
        for label in task_labels:
            task_mask[batch_y == label] = 1.0
        
        return task_mask


    def train_inc(self, dataloader, **kwargs):
        start_time = time.time()
        task_id = kwargs.get('task_id', None)
        task_name = kwargs.get('task_name', None)
        dataloaders = kwargs.get('dataloaders', None)
        present = torch.LongTensor(size=(0,)).to(device)

        if task_id == 0:
            num_d = 0
            for batch_idx, batch in enumerate(dataloader):
                num_d += len(batch[0])
                # print("batch_idx", batch_idx)
                # Stream data
                self.model.train()
                self.optim.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss = 0
                    batch_x, batch_y = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)

                    #ins_loss############################# 
                    # update classes seen
                    present = torch.cat([batch_y, present]).unique().to(device)

                    # 根据当前task的类别生成mask

                    # Augment
                    aug1 = self.rotation(batch_x)
                    aug2 = self.transform_train(aug1)
                    images_pair = torch.cat([aug1, aug2], dim=0)
                    # labels rotations or something
                    rot_sim_labels = torch.cat([batch_y.to(device) + 1000 * i for i in range(self.oop)], dim=0)
                    # Inference
                    feature_map, output_aux = self.model(images_pair, is_simclr=True)
                    simclr = self.normalize(output_aux)
                    feature_map_out = self.normalize(feature_map[:images_pair.shape[0]])

                    num1 = feature_map_out.shape[1] - simclr.shape[1]
                    id1 = torch.randperm(num1)[0]
                    size = simclr.shape[1]
                    sim_matrix = 1*torch.matmul(simclr, feature_map_out[:, id1 :id1+ 1 * size].t())
                    sim_matrix += 1 * self.get_similarity_matrix(simclr)  # *(1-torch.eye(simclr.shape[0]).to(device))#+0.5*get_similarity_matrix(feature_map_out)
                    ins_loss = self.Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels,
                                                    temperature=0.07)                             
                    loss += ins_loss

                    batch_y = batch_y.long()
                    #ce_loss#############################
                    y_pred_linear = self.model.logits(self.transform_train(batch_x))
                    ce_loss = F.cross_entropy(y_pred_linear, batch_y)
                    wandb.log({"ce_loss": ce_loss.item()})
                    ce_loss *= self.mul_ce_loss
                    loss += ce_loss

                    #etf_loss#############################
                    features, outputs = self.model(self.transform_train(batch_x), labels=batch_y, is_outputs=True)
                    features = self.pre_logits(self.etf_transform(features))
                    etf_loss = self.etf_criterion(features, self.etf_classifier[:, batch_y].t())
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
                        cons_loss = self.calculate_cons_loss(outputs, batch_y)
                        cons_loss *= self.mul_cons_loss
                        loss += cons_loss
                    else:
                        cons_loss = 0

                scaler.scale(loss).backward()
                # scaler.scale(load_balancing_loss).backward()
                scaler.step(self.optim)
                scaler.update()

                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name} has gradient")
                #     else:
                #         # print(f"{name} has no gradient")
                #         continue

                # check_gradient_explosion(self.model)
                self.iter += 1
                self.buffer.update(imgs=batch_x.detach(), labels=batch_y.detach())

                if num_d % self.print_num == 0 or batch_idx == 1:
                    print('==>>> it: {}, loss: ce {:.2f} + ins {:.4f} + ETF {:.4f} + cons {:.4f} + diff {:.4f} + load {:.4f}= {:.6f}, {}%'
                        .format(batch_idx, ce_loss, ins_loss, etf_loss, cons_loss, diff_loss, load_balancing_loss, loss, 100 * (num_d / self.total_samples)),flush=True)
                    self.myprint()

                # Plot to tensorboard
                if (batch_idx == (len(dataloader) - 1)) and (batch_idx > 0):
                    lg.info(
                        f"Phase : {task_name}   batch {batch_idx}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                    )

        else:
            num_d = 0   
            for batch_idx, batch in enumerate(dataloader):
                # print("batch_idx", batch_idx)
                num_d += len(batch[0])
                self.model.train()
                self.optim.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss = 0
                    # Stream data
                    batch_x, batch_y = batch[0].to(device), batch[1].to(device)
                    # update classes seen
                    present = torch.cat([batch_y, present]).unique().to(device)

                    mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size, present=present)
                    mem_x, mem_y = mem_x.to(device), mem_y.to(device)
                    
                    # Augment
                    aug1_batch = self.rotation(batch_x)
                    aug2_batch = self.transform_train(aug1_batch)
                    aug1_mem = self.rotation(mem_x)
                    aug2_mem = self.transform_train(aug1_mem)

                    images_pair_batch = torch.cat((aug1_batch, aug2_batch), dim=0)
                    images_pair_mem = torch.cat([aug1_mem, aug2_mem], dim=0)

                    #ins_loss#############################
                    t = torch.cat((images_pair_batch, images_pair_mem),dim=0)
                    feature_map, u = self.model(t, is_simclr=True)
                    pre_u = self.previous_model(aug1_mem, is_simclr=True)[1] #
                    feature_map_out_batch = self.normalize(feature_map[:images_pair_batch.shape[0]])
                    feature_map_out_mem = self.normalize(feature_map[images_pair_batch.shape[0]:])
                    images_out = u[:images_pair_batch.shape[0]]
                    images_out_r = u[images_pair_batch.shape[0]:]
                    pre_u = self.normalize(pre_u) #
                    simclr = self.normalize(images_out)
                    simclr_r = self.normalize(images_out_r)
                    rot_sim_labels = torch.cat([batch_y.to(device)+ 1000 * i for i in range(self.oop)],dim=0)
                    rot_sim_labels_r = torch.cat([mem_y.to(device)+ 1000 * i for i in range(self.oop)],dim=0)
                    num1 = feature_map_out_batch.shape[1] - simclr.shape[1]
                    id1 = torch.randperm(num1)[0]
                    id2=torch.randperm(num1)[0]
                    size = simclr.shape[1]
                    sim_matrix = 0.5*torch.matmul(simclr, feature_map_out_batch[:, id1:id1 + size].t())
                    sim_matrix_r = 0.5*torch.matmul(simclr_r,
                                                    feature_map_out_mem[:, id2:id2 + size].t())
                    sim_matrix += 0.5 * self.get_similarity_matrix(simclr)  # *(1-torch.eye(simclr.shape[0]).to(device))#+0.5*get_similarity_matrix(feature_map_out)
                    sim_matrix_r += 0.5 * self.get_similarity_matrix(simclr_r)
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):    
                        sim_matrix_r_pre = torch.matmul(simclr_r[:aug1_mem.shape[0]],pre_u.t())

                    loss_sim_r = self.Supervised_NT_xent_uni(sim_matrix_r,labels=rot_sim_labels_r,temperature=0.07)
                    loss_sim_pre = self.Supervised_NT_xent_pre(sim_matrix_r_pre, labels=rot_sim_labels_r, temperature=0.07)
                    loss_sim = self.Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=0.07)

                    ins_loss =1* loss_sim_r+1*loss_sim+loss_sim_pre
                    loss += ins_loss

                    #mse_loss#############################
                    y_label = self.model.logits(self.transform_train(mem_x))
                    y_label_pre = self.previous_model.logits(self.transform_train(mem_x))
                    loss += F.mse_loss(y_label_pre[:, self.old_classes.long()], y_label[:, self.old_classes.long()])

                    batch_y = batch_y.long()
                    mem_y = mem_y.long()
                    
                    #ce_loss#############################
                    y_pred_linear = self.model.logits(self.transform_train(mem_x))
                    ce_loss = F.cross_entropy(y_pred_linear, mem_y)
                    wandb.log({"ce_loss": ce_loss.item()})
                    ce_loss *= self.mul_ce_loss
                    loss += ce_loss

                    #etf_loss#############################
                    features = self.model(self.transform_train(mem_x), labels=mem_y)
                    features = self.pre_logits(self.etf_transform(features))
                    etf_loss = self.etf_criterion(features, self.etf_classifier[:, mem_y].t())
                    etf_loss = etf_loss.mean() #或许可以测试一个不mean的结果
                    etf_loss *= self.mul_etf_loss
                    loss += etf_loss

                    #diff_loss#############################
                    y_pred_etf = self.etf_predict(features, self.etf_classifier)
                    diff_loss = F.kl_div(F.log_softmax(y_pred_linear, dim=1), y_pred_etf, reduction='batchmean')
                    diff_loss *= self.mul_diff_loss
                    loss += diff_loss

                    #cons_loss#############################
                    #优化！太慢了
                    cons_loss = 0
                    load_balancing_loss = 0
                    load_balancing_loss2 = 0
                    if self.use_cons_loss:
                        _, outputs = self.model(self.transform_train(batch_x), labels=batch_y, is_outputs=True)
                        cons_loss1 = self.calculate_cons_loss(outputs, batch_y) # 可做消融
                        cons_loss += cons_loss1
                        _, mem_outputs = self.model(self.transform_train(mem_x), labels=mem_y, is_outputs=True)
                        cons_loss *= self.mul_cons_loss
                        loss += cons_loss

                        #load_balancing_loss#############################
                        load_balancing_loss = mem_outputs['load_balancing_loss']
                        load_balancing_loss *= self.mul_load_loss
                        loss += load_balancing_loss

                        load_balancing_loss2 = outputs['load_balancing_loss']
                        load_balancing_loss2 *= self.mul_load_loss
                        loss += load_balancing_loss2

                    

                # Loss
                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()
                # check_gradient_explosion(self.model)
                self.iter += 1

                self.buffer.update(imgs=batch_x.detach(), labels=batch_y.detach())

                if num_d % self.print_num == 0 or batch_idx == 1:
                    print('==>>> it: {}, loss: ce {:.2f} + ins {:.4f} + ETF {:.4f} + cons {:.4f} + diff {:.4f} + load1 {:.4f} + load2 {:.4f}= {:.6f}, {}%'
                        .format(batch_idx, ce_loss, ins_loss, etf_loss, cons_loss, diff_loss, load_balancing_loss, load_balancing_loss2, loss, 100 * (num_d / self.total_samples)),flush=True)
                    self.myprint()

                if (batch_idx == (len(dataloader) - 1)) and (batch_idx > 0):
                    lg.info(
                        f"Phase : {task_name}   batch {batch_idx}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                    )
        
        self.previous_model = deepcopy(self.model)
        self.old_classes = torch.cat([self.old_classes, present]).unique()

        # if task_id == self.taskid_max-1:
        # if task_id != -1:
        #     x, y = self.buffer.get_all()
        #     # print("x", x.shape)
        #     x = x[:999].cuda()
        #     y = y[:999].cuda()
        #     x = self.model.after_backbone(self.transform_train(x))
        #     # print("x", x.shape)
        #     self.plot_embedding_2d(x, y, save_path=f"embedding_plot_ocm_mod_c100.pdf")

        end_time = time.time()
        print(f"Training ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Total training time: {end_time - start_time:.2f} seconds")
    
    def train_blurry(self, dataloader, **kwargs):
        raise NotImplementedError



    