import torch
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
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

from torch.utils.data import DataLoader
from torchvision import transforms
from copy import deepcopy
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, RandomGrayscale
from sync_batchnorm import patch_replication_callback

from src.learners.ccldc.baseccl import BaseCCLLearner
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

device = get_device()
scaler = amp.GradScaler()
scaler2 = amp.GradScaler()

class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        if name == 'module':
            return self._modules['module']
        else:
            return getattr(self.module, name)

class OCMCCLLearner(BaseCCLLearner):
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
        self.classes_seen_so_far = torch.LongTensor(size=(0,)).to(device)
        self.old_classes = torch.LongTensor(size=(0,)).to(device)
        self.lag_task_change = 100

        # 是否真的需要extra？
        if self.n_classes_num == 10:
            self.taskid_max = 5
            self.total_samples = 10000
            self.extra = 1/2
            self.mul_ce_loss = 1
            self.mul_etf_loss = 0.01 * self.extra
            self.mul_diff_loss = 1 * self.extra
            self.mul_cons_loss = 5 * self.extra
            self.mul_load_loss = 1 * self.extra
            self.num_experts = 10
            self.top_k = 1
        elif self.n_classes_num == 100:

            self.taskid_max = 10
            self.total_samples = 5000
            self.extra = 1/2
            self.mul_ce_loss = 1
            self.mul_etf_loss = 10 * self.extra
            self.mul_diff_loss = 10 * self.extra#10
            self.mul_cons_loss = 50 * self.extra#这个参数万万不能动
            self.mul_load_loss = 1 * self.extra
            if self.params.mem_size == 5000:
                self.extra = 1/2
                self.mul_ce_loss = 1
                self.mul_etf_loss = 10 * self.extra
                self.mul_diff_loss = 1 * self.extra#10
                self.mul_cons_loss = 50 * self.extra#这个参数万万不能动
                self.mul_load_loss = 1 * self.extra
            self.num_experts = 8
            self.top_k = 1
        elif self.n_classes_num == 200:
            self.taskid_max = 100
            self.total_samples = 10000
            if self.params.mem_size == 10000:
                self.extra = 1/2
                self.mul_ce_loss = 1
                self.mul_etf_loss = 25 * self.extra
                self.mul_diff_loss = 1 * self.extra
                self.mul_cons_loss = 5 * self.extra
                self.mul_load_loss = 30 * self.extra
                self.num_experts = 15
                self.top_k = 10
            else:
                self.extra = 1/18
                self.mul_ce_loss = 1
                self.mul_etf_loss = 10 * self.extra
                self.mul_diff_loss = 10 * self.extra
                self.mul_cons_loss = 50 * self.extra
                self.mul_load_loss = 1 * self.extra
                self.num_experts = 15
                self.top_k = 10
        self.model1 = self.load_model()
        self.model2 = self.load_model()
        self.optim1 = self.load_optim1()
        self.optim2 = self.load_optim2()
        self.dim_neckout = 512
        self.print_num = self.total_samples // 10
        self.use_cons_loss = True
        # etf 
        self.alpha_k = torch.ones(1).cuda()
        self.beta_k = torch.zeros(self.dim_neckout).cuda()
        self.etf_criterion = DotRegressionLoss(reduction="none")
        self.register_buffer("etf_classifier", etf_initialize(self.dim_neckout, self.n_classes_num).cuda()) # 512 10
        self.myprint()
  
        self.oop = 16
        self.iter = 0

        self.kd_lambda = self.params.kd_lambda
        self.results = []
        self.results_forgetting = []
        self.results_1 = []
        self.results_forgetting_1 = []
        self.results_2 = []
        self.results_forgetting_2 = []

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

    def train_inc(self, dataloader, **kwargs):
        task_id = kwargs.get('task_id', None)
        task_name = kwargs.get('task_name', None)
        dataloaders = kwargs.get('dataloaders', None)
        present = torch.LongTensor(size=(0,)).to(device)

        if task_id == 0:
            num_d = 0
            for batch_idx, batch in enumerate(dataloader):
                num_d += len(batch[0])
                # Stream data
                self.model1.train()
                self.model2.train()
                self.optim1.zero_grad()
                self.optim2.zero_grad()

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss = 0
                    loss2 = 0
                    batch_x, batch_y = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)

                    #model1################################
                    #ins_loss1#############################
                    # update classes seen
                    present = torch.cat([batch_y, present]).unique().to(device)
                    # Augment
                    aug1 = self.rotation(batch_x)
                    aug2 = self.transform_train(aug1)
                    images_pair = torch.cat([aug1, aug2], dim=0)
                    # labels rotations or something
                    rot_sim_labels = torch.cat([batch_y.to(device) + 1000 * i for i in range(self.oop)], dim=0)
                    # Inference - model1
                    feature_map, output_aux = self.model1(images_pair, is_simclr=True)
                    simclr = self.normalize(output_aux)
                    feature_map_out = self.normalize(feature_map[:images_pair.shape[0]])
                    num1 = feature_map_out.shape[1] - simclr.shape[1]
                    id1 = torch.randperm(num1)[0]
                    size = simclr.shape[1]
                    sim_matrix = 1*torch.matmul(simclr, feature_map_out[:, id1 :id1+ 1 * size].t())
                    sim_matrix += 1 * self.get_similarity_matrix(simclr)  # *(1-torch.eye(simclr.shape[0]).to(device))#+0.5*get_similarity_matrix(feature_map_out)
                    ins_loss1 = self.Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels,
                                                    temperature=0.07)
                    loss += ins_loss1

                    batch_y = batch_y.long()

                    #ce_loss1#############################
                    y_pred_linear = self.model1.logits(self.transform_train(batch_x))
                    ce_loss1 = F.cross_entropy(y_pred_linear, batch_y)
                    wandb.log({"ce_loss": ce_loss1.item()})
                    ce_loss1 *= self.mul_ce_loss
                    loss += ce_loss1

                    #etf_loss1#############################
                    features, outputs = self.model1(self.transform_train(batch_x), labels=batch_y, is_outputs=True)
                    features = self.pre_logits(self.etf_transform(features))
                    etf_loss1 = self.etf_criterion(features, self.etf_classifier[:, batch_y].t())
                    etf_loss1 = etf_loss1.mean() #或许可以测试一个不mean的结果
                    etf_loss1 *= self.mul_etf_loss
                    loss += etf_loss1

                    load_balancing_loss1 = 0
                    # #load_balancing_loss#############################
                    load_balancing_loss1 = outputs['load_balancing_loss']
                    load_balancing_loss1 *= self.mul_load_loss
                    loss += load_balancing_loss1

                    #diff_loss#############################
                    y_pred_etf = self.etf_predict(features, self.etf_classifier)   
                    diff_loss1 = F.kl_div(F.log_softmax(y_pred_linear, dim=1), y_pred_etf, reduction='batchmean')
                    diff_loss1 *= self.mul_diff_loss
                    loss += diff_loss1

                    #cons_loss#############################
                    if self.use_cons_loss:
                        cons_loss1 = self.calculate_cons_loss(outputs, batch_y)
                        cons_loss1 *= self.mul_cons_loss
                        loss += cons_loss1
                    else:
                        cons_loss1 = 0


                    #model2################################
                    #ins_loss2#############################
                    # Inference - model2
                    feature_map2, output_aux2 = self.model2(images_pair, is_simclr=True)
                    simclr2 = self.normalize(output_aux2)
                    feature_map_out2 = self.normalize(feature_map2[:images_pair.shape[0]])
                    num2 = feature_map_out2.shape[1] - simclr2.shape[1]
                    id2 = torch.randperm(num2)[0]
                    size = simclr2.shape[1]
                    sim_matrix2 = 1*torch.matmul(simclr2, feature_map_out2[:, id2 :id2+ 1 * size].t())
                    sim_matrix2 += 1 * self.get_similarity_matrix(simclr2)  # *(1-torch.eye(simclr.shape[0]).to(device))#+0.5*get_similarity_matrix(feature_map_out)
                    ins_loss2 = self.Supervised_NT_xent_n(sim_matrix2, labels=rot_sim_labels,
                                                    temperature=0.07)
                    loss2 += ins_loss2

                    #ce_loss2#############################
                    y_pred_linear2 = self.model2.logits(self.transform_train(batch_x))
                    ce_loss2 = F.cross_entropy(y_pred_linear2, batch_y)
                    ce_loss2 *= self.mul_ce_loss
                    loss2 += ce_loss2

                    #etf_loss2#############################
                    features2, outputs2 = self.model2(self.transform_train(batch_x), labels=batch_y, is_outputs=True)
                    features2 = self.pre_logits(self.etf_transform(features2))
                    etf_loss2 = self.etf_criterion(features2, self.etf_classifier[:, batch_y].t())
                    etf_loss2 = etf_loss2.mean() #或许可以测试一个不mean的结果
                    etf_loss2 *= self.mul_etf_loss
                    loss2 += etf_loss2

                    load_balancing_loss2 = 0
                    # #load_balancing_loss#############################
                    load_balancing_loss2 = outputs2['load_balancing_loss']
                    load_balancing_loss2 *= self.mul_load_loss
                    loss2 += load_balancing_loss2

                    #diff_loss#############################
                    y_pred_etf2 = self.etf_predict(features2, self.etf_classifier)   
                    diff_loss2 = F.kl_div(F.log_softmax(y_pred_linear2, dim=1), y_pred_etf2, reduction='batchmean')
                    diff_loss2 *= self.mul_diff_loss
                    loss2 += diff_loss2

                    #cons_loss#############################
                    if self.use_cons_loss:
                        cons_loss2 = self.calculate_cons_loss(outputs2, batch_y)
                        cons_loss2 *= self.mul_cons_loss #需要调
                        loss2 += cons_loss2
                    else:
                        cons_loss2 = 0


                    # Distillation loss
                    # Combined batch
                    combined_x = torch.cat([batch_x.to(device)], dim=0)
                    combined_y = torch.cat([batch_y.to(device)], dim=0)
                    # Augment
                    combined_aug1 = self.transform_1(combined_x)
                    combined_aug2 = self.transform_2(combined_aug1)
                    combined_aug = self.transform_3(combined_aug2)
                    # Inference
                    logits1 = self.model1.logits(combined_aug)
                    logits2 = self.model2.logits(combined_aug)
                    logits1_vanilla = self.model1.logits(combined_x)
                    logits2_vanilla = self.model2.logits(combined_x)
                    logits1_step1 = self.model1.logits(combined_aug1)
                    logits2_step1 = self.model2.logits(combined_aug1)
                    logits1_step2 = self.model1.logits(combined_aug2)
                    logits2_step2 = self.model2.logits(combined_aug2)
                    # Cls Loss
                    loss_ce = nn.CrossEntropyLoss()(logits1, combined_y.long()) + nn.CrossEntropyLoss()(logits1_vanilla, combined_y.long()) + nn.CrossEntropyLoss()(logits1_step1, combined_y.long()) + nn.CrossEntropyLoss()(logits1_step2, combined_y.long())
                    loss_ce2 = nn.CrossEntropyLoss()(logits2, combined_y.long()) + nn.CrossEntropyLoss()(logits2_vanilla, combined_y.long()) + nn.CrossEntropyLoss()(logits2_step1, combined_y.long()) + nn.CrossEntropyLoss()(logits2_step2, combined_y.long())
                    loss_dist = self.kl_loss(logits1, logits2.detach()) + self.kl_loss(logits1_vanilla, logits2_step1.detach()) + self.kl_loss(logits1_step1, logits2_step2.detach()) + self.kl_loss(logits1_step2, logits2.detach()) 
                    loss_dist2 = self.kl_loss(logits2, logits1.detach()) + self.kl_loss(logits2_vanilla, logits1_step1.detach()) + self.kl_loss(logits2_step1, logits1_step2.detach()) + self.kl_loss(logits2_step2, logits1.detach())

                    # Total Loss
                    loss_sum = 0.25 * loss_ce + self.kd_lambda * loss_dist  + loss  
                    loss2_sum = 0.25 * loss_ce2 + self.kd_lambda * loss_dist2  + loss2 

                scaler.scale(loss_sum).backward()
                scaler.step(self.optim1)
                scaler.update()

                scaler2.scale(loss2_sum).backward()
                scaler2.step(self.optim2)
                scaler2.update()

                self.iter += 1

                self.loss = loss_sum.item()
                # print(f"Loss (Peer1) : {loss_sum.item():.4f}  Loss (Peer2) : {loss2_sum.item():.4f}  batch {batch_idx}", end="\r")
                self.buffer.update(imgs=batch_x.detach().cpu(), labels=batch_y.detach().cpu())

                if num_d % self.print_num == 0 or batch_idx == 1:
                    print('==>>> it: {}, loss: ce {:.2f} + ins {:.4f} + ETF {:.4f} + cons {:.4f} + diff {:.4f}= {:.6f}, {}%'
                        .format(batch_idx, ce_loss1, ins_loss1, etf_loss1, cons_loss1, diff_loss1, loss, 100 * (num_d / self.total_samples)),flush=True)

                if num_d % self.print_num == 0 or batch_idx == 1:
                    print('==>>> it: {}, loss: ce {:.2f} + ins {:.4f} + ETF {:.4f} + cons {:.4f} + diff {:.4f}= {:.6f}, {}%'
                        .format(batch_idx, ce_loss2, ins_loss2, etf_loss2, cons_loss2, diff_loss2, loss2, 100 * (num_d / self.total_samples)),flush=True)
                    self.myprint()

                # if (batch_idx == (len(dataloader) - 1)) and (batch_idx > 0):
                #     lg.info(
                #         f"Phase : {task_name}   batch {batch_idx}/{len(dataloader)}  Loss (Peer1) : {loss_sum.item():.4f}  Loss (Peer2) : {loss2_sum.item():.4f}  time : {time.time() - self.start:.4f}s"
                #     )
        else:
            num_d = 0
            for batch_idx, batch in enumerate(dataloader):
                num_d += len(batch[0])
                self.model1.train()
                self.model2.train()
                self.optim1.zero_grad()
                self.optim2.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss = 0
                    loss2 = 0
                    # Stream data
                    batch_x, batch_y = batch[0].to(device), batch[1].to(device)
                    # update classes seen
                    present = torch.cat([batch_y, present]).unique().to(device)

                    mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    mem_x, mem_y = mem_x.to(device), mem_y.to(device)
                    
                    # Augment
                    aug1_batch = self.rotation(batch_x)
                    aug2_batch = self.transform_train(aug1_batch)
                    aug1_mem = self.rotation(mem_x)
                    aug2_mem = self.transform_train(aug1_mem)

                    images_pair_batch = torch.cat((aug1_batch, aug2_batch), dim=0)
                    images_pair_mem = torch.cat([aug1_mem, aug2_mem], dim=0)

                    #model1################################
                    #ins_loss1#############################
                    # Inference- model1
                    t = torch.cat((images_pair_batch, images_pair_mem),dim=0)
                    feature_map, u = self.model1(t, is_simclr=True)
                    pre_u = self.previous_model(aug1_mem, is_simclr=True)[1]
                    feature_map_out_batch = self.normalize(feature_map[:images_pair_batch.shape[0]])
                    feature_map_out_mem = self.normalize(feature_map[images_pair_batch.shape[0]:])
                    images_out = u[:images_pair_batch.shape[0]]
                    images_out_r = u[images_pair_batch.shape[0]:]
                    pre_u = self.normalize(pre_u)
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

                    ins_loss1 =1* loss_sim_r+1*loss_sim+loss_sim_pre#+loss_sup1#+0*loss_sim_r1+0*loss_sim1#+0*loss_sim_mix1+0*loss_sim_mix2#+ 1 * loss_sup1#+loss_sim_kd
                    loss += ins_loss1

                    #mse_loss1#############################
                    y_label = self.model1.logits(self.transform_train(mem_x))
                    y_label_pre = self.previous_model(self.transform_train(mem_x))
                    loss += F.mse_loss(y_label_pre[:, self.old_classes.long()], y_label[:, self.old_classes.long()])

                    batch_y = batch_y.long()
                    mem_y = mem_y.long()
                    #ce_loss1#############################
                    y_pred_linear1 = self.model1.logits(self.transform_train(mem_x))
                    ce_loss1 = F.cross_entropy(y_pred_linear1, mem_y)
                    wandb.log({"ce_loss": ce_loss1.item()})
                    ce_loss1 *= self.mul_ce_loss
                    loss += ce_loss1
                
                    #etf_loss1#############################
                    features1 = self.model1(self.transform_train(mem_x), labels=mem_y)
                    features1 = self.pre_logits(self.etf_transform(features1))
                    etf_loss1 = self.etf_criterion(features1, self.etf_classifier[:, mem_y].t())
                    etf_loss1 = etf_loss1.mean() #或许可以测试一个不mean的结果
                    etf_loss1 *= self.mul_etf_loss
                    loss += etf_loss1

                    #diff_loss1#############################
                    y_pred_etf1 = self.etf_predict(features1, self.etf_classifier)
                    diff_loss1 = F.kl_div(F.log_softmax(y_pred_linear1, dim=1), y_pred_etf1, reduction='batchmean')
                    diff_loss1 *= self.mul_diff_loss
                    loss += diff_loss1

                    cons_loss1 = 0
                    load_balancing_loss = 0
                    load_balancing_loss2 = 0
                    #cons_loss1#############################
                    if self.use_cons_loss:
                        _, outputs1 = self.model1(self.transform_train(batch_x), labels=batch_y, is_outputs=True)
                        cons_loss1_1 = self.calculate_cons_loss(outputs1, batch_y) # 可做消融
                        cons_loss1 += cons_loss1_1
                        _, mem_outputs1 = self.model1(self.transform_train(mem_x), labels=mem_y, is_outputs=True)
                        cons_loss1 *= self.mul_cons_loss
                        loss += cons_loss1
                    
                        #load_balancing_loss#############################
                        load_balancing_loss1_1 = mem_outputs1['load_balancing_loss']
                        load_balancing_loss1_1 *= self.mul_load_loss
                        loss += load_balancing_loss1_1

                        load_balancing_loss1_2 = outputs1['load_balancing_loss']
                        load_balancing_loss1_2 *= self.mul_load_loss
                        loss += load_balancing_loss1_2


                    #model2################################
                    #ins_loss2#############################
                    #Inference - model2
                    feature_map2, u2 = self.model2(t, is_simclr=True)
                    pre_u2 = self.previous_model2(aug1_mem, is_simclr=True)[1]
                    feature_map_out_batch2 = self.normalize(feature_map2[:images_pair_batch.shape[0]])
                    feature_map_out_mem2 = self.normalize(feature_map2[images_pair_batch.shape[0]:])
                    images_out2 = u2[:images_pair_batch.shape[0]]
                    images_out_r2 = u2[images_pair_batch.shape[0]:]
                    pre_u2 = self.normalize(pre_u2)
                    simclr2 = self.normalize(images_out2)
                    simclr_r2 = self.normalize(images_out_r2)
                    rot_sim_labels2 = torch.cat([batch_y.to(device)+ 1000 * i for i in range(self.oop)],dim=0)
                    rot_sim_labels_r2 = torch.cat([mem_y.to(device)+ 1000 * i for i in range(self.oop)],dim=0)
                    num12 = feature_map_out_batch2.shape[1] - simclr2.shape[1]
                    id12 = torch.randperm(num12)[0]
                    id22=torch.randperm(num12)[0]
                    size2 = simclr2.shape[1]
                    sim_matrix2 = 0.5*torch.matmul(simclr2, feature_map_out_batch2[:, id12:id12 + size2].t())
                    sim_matrix_r2 = 0.5*torch.matmul(simclr_r2,
                                                    feature_map_out_mem2[:, id22:id22 + size2].t())
                    sim_matrix2 += 0.5 * self.get_similarity_matrix(simclr2)  # *(1-torch.eye(simclr.shape[0]).to(device))#+0.5*get_similarity_matrix(feature_map_out)
                    sim_matrix_r2 += 0.5 * self.get_similarity_matrix(simclr_r2)
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                        sim_matrix_r_pre2 = torch.matmul(simclr_r2[:aug1_mem.shape[0]],pre_u2.t())
                    loss_sim_r2 = self.Supervised_NT_xent_uni(sim_matrix_r2,labels=rot_sim_labels_r2,temperature=0.07)
                    loss_sim_pre2 = self.Supervised_NT_xent_pre(sim_matrix_r_pre2, labels=rot_sim_labels_r2, temperature=0.07)
                    loss_sim2 = self.Supervised_NT_xent_n(sim_matrix2, labels=rot_sim_labels2, temperature=0.07)

                    ins_loss2 =1* loss_sim_r2+1*loss_sim2+loss_sim_pre2#+loss_sup1#+0*loss_sim_r1+0*loss_sim1#+0*loss_sim_mix1+0*loss_sim_mix2#+ 1 * loss_sup1#+loss_sim_kd
                    loss2 += ins_loss2

                    #mse_loss2#############################
                    y_label2 = self.model2.logits(self.transform_train(mem_x))
                    y_label_pre2 = self.previous_model2(self.transform_train(mem_x))
                    loss2 += F.mse_loss(y_label_pre2[:, self.old_classes.long()], y_label2[:, self.old_classes.long()])

                    #ce_loss2#############################
                    y_pred_linear2 = self.model2.logits(self.transform_train(mem_x))
                    ce_loss2 = F.cross_entropy(y_pred_linear2, mem_y)
                    ce_loss2 *= self.mul_ce_loss
                    loss2 += ce_loss2
                
                    #etf_loss2#############################
                    features2 = self.model2(self.transform_train(mem_x), labels=mem_y)
                    features2 = self.pre_logits(self.etf_transform(features2))
                    etf_loss2 = self.etf_criterion(features2, self.etf_classifier[:, mem_y].t())
                    etf_loss2 = etf_loss2.mean() #或许可以测试一个不mean的结果
                    etf_loss2 *= self.mul_etf_loss
                    loss2 += etf_loss2

                    #diff_loss2#############################
                    y_pred_etf2 = self.etf_predict(features2, self.etf_classifier)
                    diff_loss2 = F.kl_div(F.log_softmax(y_pred_linear2, dim=1), y_pred_etf2, reduction='batchmean')
                    diff_loss2 *= self.mul_diff_loss
                    loss2 += diff_loss2

                    #cons_loss2#############################
                    cons_loss2 = 0
                    load_balancing_loss2_1 = 0
                    load_balancing_loss2_2 = 0
                    if self.use_cons_loss:
                        _, outputs2 = self.model2(self.transform_train(batch_x), labels=batch_y, is_outputs=True)
                        cons_loss2_1 = self.calculate_cons_loss(outputs2, batch_y) # 可做消融
                        cons_loss2 += cons_loss2_1
                        _, mem_outputs2 = self.model2(self.transform_train(mem_x), labels=mem_y, is_outputs=True)
                        cons_loss2 *= self.mul_cons_loss
                        loss2 += cons_loss2

                        #load_balancing_loss#############################
                        load_balancing_loss2_1 = mem_outputs2['load_balancing_loss']
                        load_balancing_loss2_1 *= self.mul_load_loss
                        loss2 += load_balancing_loss2_1

                        load_balancing_loss2_2 = outputs2['load_balancing_loss']
                        load_balancing_loss2_2 *= self.mul_load_loss
                        loss2 += load_balancing_loss2_2

                    

                    # Distillation loss
                    # Combined batch
                    combined_x = torch.cat([batch_x.to(device), mem_x.to(device)], dim=0)
                    combined_y = torch.cat([batch_y.to(device), mem_y.to(device)], dim=0)
                    
                    # Augment
                    combined_aug1 = self.transform_1(combined_x)
                    combined_aug2 = self.transform_2(combined_aug1)
                    combined_aug = self.transform_3(combined_aug2)
                    # Inference

                    logits1 = self.model1.logits(combined_aug)
                    logits2 = self.model2.logits(combined_aug)

                    logits1_vanilla = self.model1.logits(combined_x)
                    logits2_vanilla = self.model2.logits(combined_x)

                    logits1_step1 = self.model1.logits(combined_aug1)
                    logits2_step1 = self.model2.logits(combined_aug1)

                    logits1_step2 = self.model1.logits(combined_aug2)
                    logits2_step2 = self.model2.logits(combined_aug2)
                    # Cls Loss
                    loss_ce = nn.CrossEntropyLoss()(logits1, combined_y.long()) + nn.CrossEntropyLoss()(logits1_vanilla, combined_y.long()) + nn.CrossEntropyLoss()(logits1_step1, combined_y.long()) + nn.CrossEntropyLoss()(logits1_step2, combined_y.long())
                    loss_ce2 = nn.CrossEntropyLoss()(logits2, combined_y.long()) + nn.CrossEntropyLoss()(logits2_vanilla, combined_y.long()) + nn.CrossEntropyLoss()(logits2_step1, combined_y.long()) + nn.CrossEntropyLoss()(logits2_step2, combined_y.long())

                    loss_dist = self.kl_loss(logits1, logits2.detach()) + self.kl_loss(logits1_vanilla, logits2_step1.detach()) + self.kl_loss(logits1_step1, logits2_step2.detach()) + self.kl_loss(logits1_step2, logits2.detach()) 
                    loss_dist2 = self.kl_loss(logits2, logits1.detach()) + self.kl_loss(logits2_vanilla, logits1_step1.detach()) + self.kl_loss(logits2_step1, logits1_step2.detach()) + self.kl_loss(logits2_step2, logits1.detach())

                    # Total Loss
                    loss_sum = 0.25 * loss_ce + self.kd_lambda * loss_dist  + loss  
                    loss2_sum = 0.25 * loss_ce2 + self.kd_lambda * loss_dist2  + loss2 

                # Loss
                scaler.scale(loss_sum).backward()
                scaler.step(self.optim1)
                scaler.update()

                scaler2.scale(loss2_sum).backward()
                scaler2.step(self.optim2)
                scaler2.update()

                self.iter += 1
                
                self.loss = loss_sum.item()
                # print(f"Loss (Peer1) : {loss_sum.item():.4f}  Loss (Peer2) : {loss2_sum.item():.4f}  batch {batch_idx}", end="\r")
                self.buffer.update(imgs=batch_x.detach(), labels=batch_y.detach())

                if num_d % self.print_num == 0 or batch_idx == 1:
                    print('==>>> it: {}, loss: ce {:.2f} + ins {:.4f} + ETF {:.4f} + cons {:.4f} + diff {:.4f} + load1 {:.4f} + load2 {:.4f}= {:.6f}, {}%'
                        .format(batch_idx, ce_loss1, ins_loss1, etf_loss1, cons_loss1, diff_loss1, load_balancing_loss1_1, load_balancing_loss1_2, loss, 100 * (num_d / self.total_samples)),flush=True)

                if num_d % self.print_num == 0 or batch_idx == 1:
                    print('==>>> it: {}, loss: ce {:.2f} + ins {:.4f} + ETF {:.4f} + cons {:.4f} + diff {:.4f} + load1 {:.4f} + load2 {:.4f}= {:.6f}, {}%'
                        .format(batch_idx, ce_loss2, ins_loss2, etf_loss2, cons_loss2, diff_loss2, load_balancing_loss2_1, load_balancing_loss2_2, loss2, 100 * (num_d / self.total_samples)),flush=True)
                    self.myprint()

                # if (batch_idx == (len(dataloader) - 1)) and (batch_idx > 0):
                #     lg.info(
                #         f"Phase : {task_name}   batch {batch_idx}/{len(dataloader)}  Loss (Peer1) : {loss_sum.item():.4f}  Loss (Peer2) : {loss2_sum.item():.4f}  time : {time.time() - self.start:.4f}s"
                #     )

        self.previous_model = deepcopy(self.model1)
        self.previous_model2 = deepcopy(self.model2)
        self.old_classes = torch.cat([self.old_classes, present]).unique()

        if task_id != -1:
            x, y = self.buffer.get_all()
            # print("x", x.shape)
            x = x[:999].cuda()
            y = y[:999].cuda()
            x = self.model1.after_backbone(self.transform_train(x))
            # print("x", x.shape)
            self.plot_embedding_2d(x, y, save_path=f"embedding_plot_ocmccldc_mod_c100.pdf")

    
    def train_blurry(self, dataloader, **kwargs):
        raise NotImplementedError

    def print_results(self, task_id):
        n_dashes = 20
        pad_size = 8
        print('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
        
        print('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
        for line in self.results:
            print('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line), f"{np.nanmean(line):.4f}")

    