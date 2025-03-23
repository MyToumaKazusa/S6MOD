# INPUT (B, D, H, W)
# OUTPUT (B, D, H, W)

from collections import OrderedDict

import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from rope import *
from timm.models.layers import trunc_normal_

from mmcls.models.builder import NECKS
from mmcls.utils import get_root_logger

from mamba_ssm.modules.mamba_simple import Mamba
from src.models.ss2d import SS2D
import math

import torch.nn.functional as F

# import numpy as np
# torch.set_printoptions(threshold=np.inf)



@NECKS.register_module(force=True)
class MambaNeck_1(BaseModule):
    """Dual selective SSM branch in Mamba-FSCIL framework.

        This module integrates our dual selective SSM branch for dynamic adaptation in few-shot
        class-incremental learning tasks.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int, optional): Number of intermediate channels in MLP projections, defaults to twice the in_channels if not specified.
            version (str): Specifies the version of the state space model; 'ssm' or 'ss2d'.
            use_residual_proj (bool): If True, adds a residual projection.
            d_state (int): Dimension of the hidden state in the SSM.
            d_rank (int, optional): Dimension rank in the SSM, if not provided, defaults to d_state.
            ssm_expand_ratio (float): Expansion ratio for the SSM block.
            num_layers (int): Number of layers in the MLP projections.
            num_layers_new (int, optional): Number of layers in the new branch MLP projections, defaults to num_layers if not specified.
            feat_size (int): Size of the input feature map.
            use_new_branch (bool): ALL time true
            loss_weight_supp (float): Loss weight for suppression term for base classes.
            loss_weight_supp_novel (float): Loss weight for suppression term for novel classes.
            loss_weight_sep (float): Loss weight for separation term during the base session.
            loss_weight_sep_new (float): Loss weight for separation term during the incremental session.
            param_avg_dim (str): Dimensions to average for computing averaged input-dependment parameter features; '0-1' or '0-3' or '0-1-3'.
            detach_residual (bool): If True, detaches the residual connections during the output computation.
            length: It represents the length of the sequence after expansion. It must be a perfect square, as square root operations are involved later.
    """
    def __init__(self,
                 in_channels=128,
                 out_channels=128, # 可以修改SS2D的输出维度
                 mid_channels=None,
                 version='ss2d',
                 use_residual_proj=False,
                 d_state=16, # 决定B和C的特征维度 # ???官方说这里应该用16???
                 d_rank=None, # 决定dts的特征维度
                 ssm_expand_ratio=1,
                 num_layers=2,
                 num_layers_new=None,
                 feat_size=8, # 2
                 use_new_branch=True,
                 loss_weight_supp=0.0,
                 loss_weight_supp_novel=0.0,
                 loss_weight_sep=0.0,
                 loss_weight_sep_new=0.0,
                 param_avg_dim='0-1-3',
                 a=0, # 目前没有，暂时取消了。# 改成可学习的参数
                 detach_residual=False,
                 length=16, # 先删去了这个功能
                 change = False,
                 num_experts=8,
                 top_k=2,
                 num_classes=10): #todo: 调整
        super(MambaNeck_1, self).__init__(init_cfg=None)
        self.version = version
        assert self.version in ['ssm', 'ss2d'], f'Invalid branch version.'
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.use_residual_proj = use_residual_proj
        self.mid_channels = in_channels * 2 if mid_channels is None else mid_channels
        self.feat_size = feat_size
        self.d_state = d_state
        self.d_rank = d_rank if d_rank is not None else d_state
        self.use_new_branch = use_new_branch
        self.num_layers = num_layers
        self.num_layers_new = self.num_layers if num_layers_new is None else num_layers_new
        self.detach_residual = detach_residual
        self.loss_weight_supp = loss_weight_supp
        self.loss_weight_supp_novel = loss_weight_supp_novel
        self.loss_weight_sep = loss_weight_sep
        self.loss_weight_sep_new = loss_weight_sep_new
        self.param_avg_dim = [int(item) for item in param_avg_dim.split('-')]
        self.logger = get_root_logger()
        self.a = a
        self.ln = nn.LayerNorm(in_channels)
        # self.a = nn.Parameter(torch.tensor(0.5))
        directions = ('h', 'h_flip', 'v', 'v_flip')
        self.length = length

        self.num_classes = num_classes
        if self.num_classes != 200:
            H = W = 4
        else:
            H = W = 8
        D = 128
        self.mean = torch.full((self.num_classes, H, W, D), float('inf')).cuda()
        self.generate_topk = nn.Linear(1, num_experts).cuda()
        self.num_experts = num_experts
        if self.num_classes != 100:
            self.mul_pro1 = 12
        else:
            self.mul_pro1 = 24

        # self.mean = torch.zeros((self.num_classes, H, W, D)).cuda()

        # Positional embeddings for features
        # self.pos_embed = nn.Parameter(torch.zeros(1, feat_size*feat_size, out_channels))
        # trunc_normal_(self.pos_embed, std=.02)

        self.mlp_proj = self.build_mlp(in_channels, out_channels, self.mid_channels, 
                                    num_layers=self.num_layers) 
        
        # out_channels会传递给d_model
        self.block = SS2D(out_channels,
                            ssm_ratio=ssm_expand_ratio,
                            d_state=d_state,
                            dt_rank=self.d_rank,
                            directions=directions,
                            use_out_proj=False, # 用True还是False？？？ 维度一样
                            use_out_norm=True,
                            num_experts=num_experts,
                            top_k=top_k)
        print("SS2D", out_channels)

        self.learnable_padding = nn.Parameter(torch.randn(1))

        self.outlinear = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)


    def build_mlp(self, in_channels, out_channels, mid_channels, num_layers):
        """Builds the MLP projection part of the neck.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int): Number of mid-level channels.
            num_layers (int): Number of linear layers in the MLP.

        Returns:
            nn.Sequential: The MLP layers as a sequential module.
        """
        layers = []
        layers.append(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=1,
                      padding=0,
                      bias=True))
        layers.append(
            build_norm_layer(
                dict(type='LN'),
                [mid_channels, self.feat_size, self.feat_size])[1])
        layers.append(nn.LeakyReLU(0.1))

        if num_layers == 3:
            layers.append(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1,
                          bias=True))
            layers.append(
                build_norm_layer(
                    dict(type='LN'),
                    [mid_channels, self.feat_size, self.feat_size])[1])
            layers.append(nn.LeakyReLU(0.1))

        layers.append(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False))
        return nn.Sequential(*layers)

    def broaden_length(self, x, length: int):
        """
        Expand the sequence length of the feature output from the backbone.
        :param sequence length: the second dimention of the tensor
        :input: feature with (B, L, D) shape
        :output: feature with (B, length, D) shape
        """

        B, L, D = x.shape
        if L < length:
            L1 = length - L
            padding = self.learnable_padding.expand(B, L1, D)
            x = torch.cat([x, padding], dim=1)
        return x

    def forward(self, x, labels=None):
        """Forward pass for MambaNeck, integrating both the main and an optional new branch for processing.

            Args:
                x (Tensor): Input tensor, potentially as part of a tuple from previous layers.

            Returns:
                dict: A dictionary of outputs including processed features from main and new branches,
                      along with the combined final output.
            """
        # input (B, D, H, W)
        # "Selective SSM Branch"
        "no_input_mlp+no_pos+layernorm"
        # Extract the last element if input is a tuple (from previous layers).
        if isinstance(x, tuple):
            x = x[-1]

        B, D, H, W = x.shape
        outputs = {}
        C, dts, Bs, Cs = None, None, None, None

        x = x.view(B, H, W, -1)

        x = self.ln(x)
        num = 15

        if labels is not None:
            # Trainging
            self.update_mean(x, labels) # 是否在ln后面
            self.calculate_mis_prob()

            mis_mask = self.generate_mis_mask_training(labels)
            # mis_mask = torch.full(labels.shape, num)

            # print("mis_mask", mis_mask)
            
        else:
            # Inference
            mis_mask = self.generate_mis_mask_inference(x)
            # mis_mask = torch.full((x.size(0),), num)

        
        x, C = self.block(x, return_param=True, mis_mask=mis_mask)    # x (74, 4, 4, 512)print("x", x)
        if isinstance(C, list):
            C, dts, Bs, Cs, z_loss = C  
            outputs.update({'dts': dts, 'Bs': Bs, 'Cs': Cs , 'load_balancing_loss': z_loss})
        """Combines outputs from the main and new branches with the identity projection."""

        # 如果输入和输出维度不一致，这里就不能用F直接相加
        # final_result = self.a * F + (1-self.a) * x.view(B, H * W, -1) #(B, L ,D)
        final_result = x.view(B, H * W, -1)
        # 这里不能用D，因为D是输入维度，输出维度是out_channels
        final_result = (final_result.view(B, H, W, -1)).permute(0, 3, 1, 2) #(B, D, H, W)

        return final_result, outputs

    def update_mean(self, x, labels):
        B, H, W, D = x.size()
        unique_labels = labels.unique()
        
        # 更新均值
        for label in range(self.num_classes):
            if label in unique_labels:
                # 计算当前类别的均值
                class_samples = x[labels == label]
                class_mean = class_samples.mean(dim=0).cuda().detach()
                
                # 更新均值，避免与inf比较
                if torch.isinf(self.mean[label]).any():
                    self.mean[label] = class_mean.cuda().detach()
                else:
                    self.mean[label] = 0.8*self.mean[label].detach() + 0.2*class_mean.detach()  # 或者其他合适的更新方法
                # if self.mean[label] != self.mean[label]:
                #     print("label", label)
                #     print("mean", self.mean[label])
                #     print("class_mean", class_mean)

    def calculate_mis_prob(self):
        mean = self.mean.view(self.mean.size(0), -1)
        dis_matrix = torch.cdist(mean, mean).cuda()
        # print("dis_matrix1", dis_matrix)

        inf_mask = torch.isinf(mean).any(dim=1)  # 返回一个布尔向量，表示哪些行包含 inf
        dis_matrix[inf_mask] = float('inf')  # 设置含 inf 的行
        dis_matrix[:, inf_mask] = float('inf')  # 设置含 inf 的列
        dis_matrix[torch.eye(dis_matrix.size(0), dtype=torch.bool)] = float('inf')

        # print("dis_matrix2", dis_matrix)

        # 计算 self.misclassification_probs
        neg_dist = -dis_matrix
        exp_neg_dist = torch.exp(neg_dist / self.mul_pro1)
        exp_neg_dist = exp_neg_dist.max(dim=1)[0]
        self.misclassification_probs = exp_neg_dist.detach()

    def generate_mis_mask_training(self, labels):
        B = labels.size(0)
        mis_mask = torch.zeros(B)
        for i in range(B):
            mis_pro = self.misclassification_probs[labels[i]]
            mis_mask[i] = torch.round(self.num_experts * mis_pro)
            if mis_mask[i] == 0:
                mis_mask[i] = 1
            # if mis_mask[i] != 1 and mis_mask[i] != 2 and mis_mask[i] != 3 and mis_mask[i] != 4 and mis_mask[i] != 5 and mis_mask[i] != 6 and mis_mask[i] != 7 and mis_mask[i] != 8 and mis_mask[i] != 9 and mis_mask[i] != 10:
            # if mis_mask[i] == 0:    
            #     print("label", labels[i])
            #     # print("mean_label", self.mean[labels[i]])
            #     # print("self.mis", self.misclassification_probs)
            #     tensor_str = self.mean[labels[i]].detach().cpu().numpy().astype(str)  # 将tensor转换为numpy数组并再转为字符串
            #     tensor_str = "\n".join(["\t".join(row) for row in tensor_str])  # 以制表符分隔每个元素，并行连接为字符串
            #     file_path = 'tensor_output.txt'  # 定义文件路径
            #     with open(file_path, 'w') as file:  # 以写入模式打开文件
            #         file.write(tensor_str)  # 将字符串写入文件

        return mis_mask

    def generate_mis_mask_inference(self, x):
        # calculate uncertainty
        mean = self.mean.view(self.mean.size(0), -1)
        x_cal = x.view(x.size(0), -1).detach()
        dis_matrix = torch.cdist(x_cal, mean).cuda()

        inf_mask = torch.isinf(mean).any(dim=1)  # 返回一个布尔向量，表示哪些行包含 inf
        dis_matrix[:, inf_mask] = float('inf')  # 设置含 inf 的列

        # 计算 self.misclassification_probs
        neg_dist = -dis_matrix
        exp_neg_dist = torch.exp(neg_dist / self.mul_pro1)
        exp_neg_dist = exp_neg_dist.max(dim=1)[0]
        
        B = x.size(0)
        mis_mask = torch.zeros(B)
        for i in range(B):
            mis_pro = exp_neg_dist[i]
            mis_mask[i] = torch.round(self.num_experts * mis_pro)
            if mis_mask[i] == 0:
                mis_mask[i] = 1
        return mis_mask


    def calculate_sim(self, x):        
        # 计算每个样本与每个类别的余弦相似度
        x = x.view(x.size(0), -1)  # 将特征图展平
        mean_flat = self.mean.view(self.num_classes, -1)
        similarity_matrix = torch.mm(x, mean_flat.t())  # 形状为 (batch_size, num_classes)
        
        # 找到每个样本的最大相似度对应的类别索引，作为伪标签
        pseudo_labels = torch.argmax(similarity_matrix, dim=1)
        
        return pseudo_labels