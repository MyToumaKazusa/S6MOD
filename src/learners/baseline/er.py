import torch
import time
import torch.nn as nn
import random as r
import numpy as np
import os
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix

from src.learners.baseline.base import BaseLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
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
all_time = 0

class ERLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = Reservoir(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method,
        )
        self.iter = 0
        self.n_classes_num = self.params.n_classes
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
            # self.mul_etf_loss = 1
            # self.mul_diff_loss = 1 #10
            # self.mul_cons_loss = 5 #这个参数万万不能动
            # self.mul_load_loss = 1
            # self.num_experts = 8
            self.mul_etf_loss = 0
            self.mul_diff_loss = 0 #10
            self.mul_cons_loss = 0 #这个参数万万不能动
            self.mul_load_loss = 0
            self.num_experts = 8
            self.top_k = 1
        elif self.n_classes_num == 200:
            # e1d1best e1
            # c50best
            self.taskid_max = 100
            self.total_samples = 10000
            self.extra = 1
            self.mul_ce_loss = 1
            self.mul_etf_loss = 1 * self.extra
            self.mul_diff_loss = 1 * self.extra
            self.mul_cons_loss = 50 * self.extra
            self.mul_load_loss = 1 * self.extra
            if self.params.mem_size == 10000:
                self.extra = 1/2
                self.mul_ce_loss = 1
                self.mul_etf_loss = 1 * self.extra
                self.mul_diff_loss = 1 * self.extra
                self.mul_cons_loss = 5 * self.extra
                self.mul_load_loss = 1 * self.extra
            self.num_experts = 10
            self.top_k = 10
        self.model = self.load_model()
        self.optim = self.load_optim()
        self.dim_neckout = 128
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

    def load_criterion(self):
        return nn.CrossEntropyLoss()
    
    def load_model(self, **kwargs):
        if self.params.dataset == 'cifar10' or self.params.dataset == 'cifar100' or self.params.dataset == 'tiny':
            model = M_resnet18(nclasses=self.params.n_classes, choose_mamba='mamba modulized', num_experts=self.num_experts, top_k=self.top_k)
            return model.to(device)
        elif self.params.dataset == 'imagenet' or self.params.dataset == 'imagenet100':
            return ImageNet_ResNet18(
                dim_in=self.params.dim_in,
                nclasses=self.params.n_classes,
                nf=self.params.nf
            ).to(device)

    
    def train(self, dataloader, **kwargs):
        start_time = time.time()
        task_name  = kwargs.get('task_name', 'unknown task')
        task_id    = kwargs.get('task_id', 0)
        dataloaders = kwargs.get('dataloaders', None)
        self.model = self.model.train()
        num_d = 0
        for batch_idx, batch in enumerate(dataloader):
            # Stream data
            num_d += len(batch[0])
            batch_x, batch_y = batch[0], batch[1]
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.stream_idx += len(batch_x)
            
            for _ in range(self.params.mem_iters):
                mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                loss = 0

                if mem_x.size(0) > 0:
                    combined_x, combined_y = self.combine(batch_x, batch_y, mem_x, mem_y)  # (batch_size, nb_channel, img_size, img_size)
                    combined_x = self.transform_train(combined_x)
                    batch_y = batch_y.long()
                    combined_y = combined_y.long()
                    # ce_loss
                    y_pred_linear = self.model.logits(combined_x)
                    ce_loss = self.criterion(y_pred_linear, combined_y)
                    wandb.log({"ce_loss": ce_loss.item()})
                    ce_loss *= self.mul_ce_loss
                    loss += ce_loss
                    # etf_loss
                    features = self.model(self.transform_train(combined_x), labels=combined_y)
                    features = self.pre_logits(self.etf_transform(features))
                    etf_loss = self.etf_criterion(features, self.etf_classifier[:, combined_y].t())
                    etf_loss = etf_loss.mean()
                    etf_loss *= self.mul_etf_loss
                    loss += etf_loss
                    # diff_loss
                    y_pred_etf = self.etf_predict(features, self.etf_classifier)
                    diff_loss = F.kl_div(F.log_softmax(y_pred_linear, dim=1), y_pred_etf, reduction='batchmean')
                    diff_loss *= self.mul_diff_loss
                    loss += diff_loss
                    # cons_loss
                    cons_loss = 0
                    load_balancing_loss = 0
                    if self.use_cons_loss:
                        _, outputs = self.model(self.transform_train(batch_x), labels=batch_y, is_outputs=True)
                        cons_loss = self.calculate_cons_loss(outputs, batch_y) # 可做消融
                        cons_loss *= self.mul_cons_loss
                        loss += cons_loss

                        #load_balancing_loss#############################
                        load_balancing_loss = outputs['load_balancing_loss']
                        load_balancing_loss *= self.mul_load_loss
                        loss += load_balancing_loss

                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    self.iter += 1
               
            # Update reservoir buffer
            self.buffer.update(imgs=batch_x, labels=batch_y, model=self.model)

            if num_d % self.print_num == 0 or batch_idx == 1:
                    print('==>>> it: {}, loss: ce {:.2f} + ETF {:.4f} + cons {:.4f} + diff {:.4f} + load1 {:.4f} = {:.6f}, {}%'
                        .format(batch_idx, ce_loss, etf_loss, cons_loss, diff_loss, load_balancing_loss, loss, 100 * (num_d / self.total_samples)),flush=True)
                    self.myprint()
        end_time = time.time()
        print(f"Training ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Total training time: {end_time - start_time:.2f} seconds")
        all_time+= (end_time - start_time)
        print("all_time", all_time)

    def print_results(self, task_id):
        n_dashes = 20
        pad_size = 8
        print('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
        
        print('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
        for line in self.results:
            print('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line), f"{np.nanmean(line):.4f}")
    
    def combine(self, batch_x, batch_y, mem_x, mem_y):
        mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        combined_x = torch.cat([mem_x, batch_x])
        combined_y = torch.cat([mem_y, batch_y])
        if self.params.memory_only:
            return mem_x, mem_y
        else:
            return combined_x, combined_y
        