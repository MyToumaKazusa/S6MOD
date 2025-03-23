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



'''
@NECKS.register_module(force=True)
class MambaNeck(BaseModule):
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
    """

    def __init__(self,
                 in_channels=512,
                 out_channels=512,
                 mid_channels=None,
                 version='ss2d',
                 use_residual_proj=False,
                 d_state=256,
                 d_rank=None,
                 ssm_expand_ratio=1,
                 num_layers=2,
                 num_layers_new=None,
                 feat_size=4, # 2
                 use_new_branch=True,
                 loss_weight_supp=0.0,
                 loss_weight_supp_novel=0.0,
                 loss_weight_sep=0.0,
                 loss_weight_sep_new=0.0,
                 param_avg_dim='0-1-3',
                 a=0.5,
                 detach_residual=False):
        super(MambaNeck, self).__init__(init_cfg=None)
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
        self.length = 64
        directions = ('h', 'h_flip', 'v', 'v_flip')

        # Positional embeddings for features
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.feat_size * self.feat_size, out_channels))
        trunc_normal_(self.pos_embed, std=.02)

        self.pos_embed_new = nn.Parameter(
            torch.zeros(1, self.feat_size * self.feat_size, out_channels))
        trunc_normal_(self.pos_embed_new, std=.02)


        self.mlp_proj = self.build_mlp(in_channels, out_channels, self.mid_channels, 
                                    num_layers=self.num_layers) 


        
        self.block = SS2D(out_channels,
                            ssm_ratio=ssm_expand_ratio,
                            d_state=d_state,
                            dt_rank=self.d_rank,
                            directions=directions,
                            use_out_proj=False, # 用True还是False？？？ 维度一样
                            use_out_norm=True)


        self.mlp_proj_new = self.build_mlp(in_channels, out_channels, self.mid_channels,
                                        num_layers=self.num_layers_new)

        
        self.block_new = SS2D(out_channels,
                                ssm_ratio=ssm_expand_ratio,
                                d_state=d_state,
                                dt_rank=self.d_rank,
                                directions=directions,
                                use_out_proj=False,
                                use_out_norm=True)

        self.learnable_padding = nn.Parameter(torch.randn(1))

        self.init_weights()

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

    def init_weights(self):
        # why only initial one branch?
        """Zero initialization for the newly attached residual branche."""
        with torch.no_grad():
            dim_proj = int(self.block_new.in_proj.weight.shape[0] / 2)
            self.block_new.in_proj.weight.data[-dim_proj:, :].zero_()
            self.block.in_proj.weight.data[-dim_proj:, :].zero_()
        self.logger.info(
            f'--MambaNeck zero_init_residual z: '
            f'(self.block_new.in_proj.weight{self.block_new.in_proj.weight.shape}), '
            f'{torch.norm(self.block_new.in_proj.weight.data[-dim_proj:, :])}'
            )

    def _BroadenLength(self, x, length: int):
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


    def forward(self, x):
        """Forward pass for MambaNeck, integrating both the main and an optional new branch for processing.

            Args:
                x (Tensor): Input tensor, potentially as part of a tuple from previous layers.

            Returns:
                dict: A dictionary of outputs including processed features from main and new branches,
                      along with the combined final output.
            """
        # Extract the last element if input is a tuple (from previous layers).
        if isinstance(x, tuple):
            # print("x0:",x)
            x = x[-1]
        # print("x:",x.shape)
        # print("x[-1]:",x.shape)
        B, D, H, W = x.shape
        identity = x
        outputs = {}

        C, dts, Bs, Cs, C_new, dts_new, Bs_new, Cs_new = None, None, None, None, None, None, None, None

        # if self.detach_residual:
        #     self.block.eval()
        #     self.mlp_proj.eval()

        # Prepare the identity projection for the residual connection
            
        # 原维度为（B，C，H，W）变成（B，H * W，C）
        x = self.mlp_proj(identity).permute(0, 2, 3, 1).view(B, H * W, -1)

        # 提升序列长度
        x = self._BroadenLength(x, self.length)

        # Process the input tensor through MLP projection and add positional embeddings
        F = x.view(B, H * W, -1) # 是否不应该加上pos_embed或者应该两个都加上
        x = x.view(B, H * W, -1) + self.pos_embed # 广播机制使pos_embed变成（B，L，D）
        F = F + self.pos_embed + self.pos_embed_new


        # SS2D processing
        x = x.view(B, H, W, -1)
        x, C = self.block(x, return_param=True)  # x (74, 4, 4, 512)
        # print("x.shape:", x.shape)

        if isinstance(C, list):
            C, dts, Bs, Cs = C
            outputs.update({'dts': dts, 'Bs': Bs, 'Cs': Cs})
        # x = self.avg(x.permute(0, 3, 1, 2)).view(B, -1)


        # New branch processing for incremental learning sessions, if enabled.
        x_new = self.mlp_proj_new(identity.detach()).permute(
            0, 2, 3, 1).view(B, H * W, -1)
        x_new += self.pos_embed_new

        
        x_new = x_new.view(B, H, W, -1)
        x_new, C_new = self.block_new(x_new, return_param=True)
        if isinstance(C_new, list):
            C_new, dts_new, Bs_new, Cs_new = C_new
            outputs.update({
                'dts_new': dts_new,
                'Bs_new': Bs_new,
                'Cs_new': Cs_new
            })
        # x_new = self.avg(x_new.permute(0, 3, 1, 2)).view(B, -1)
        # x_new (74, 512)
        # C_new (74, 4, 4, 512)

        
        """Combines outputs from the main and new branches with the identity projection."""

        x_flatten = self.avg(x.permute(0, 3, 1, 2)).reshape(B, D, -1)
        x_new_flatten = self.avg(x_new.permute(0, 3, 1, 2)).reshape(B, D, -1)
        W1 = torch.bmm(x_flatten, x_new_flatten.transpose(1, 2))
        result = torch.bmm(F, W1) # (B, L, D) (74, 16, 512)
        
        # return outputs
        final_result = self.a * F + (1-self.a) * result #(B, L ,D)
        final_result = (final_result.view(B, H, W, D)).permute(0, 3, 1, 2) #(B, H, W, D)
        # return F.mean(dim=1)
        return final_result
'''


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
                 in_channels=512,
                 out_channels=512, # 可以修改SS2D的输出维度
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
        self.change = change
        self.ln = nn.LayerNorm(in_channels)
        # self.a = nn.Parameter(torch.tensor(0.5))
        directions = ('h', 'h_flip', 'v', 'v_flip')
        self.length = length

        self.num_classes = num_classes
        if self.num_classes != 200:
            H = W = 4
        else:
            H = W = 8
        D = 512
        self.mean = torch.full((self.num_classes, H, W, D), float('inf')).cuda()
        self.generate_topk = nn.Linear(1, num_experts).cuda()
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
        if self.change == True:
            "VSS Block"
            # Extract the last element if input is a tuple (from previous layers).
            if isinstance(x, tuple):
                x = x[-1]

            B, D, H, W = x.shape
            identity = x
            outputs = {}

            C, dts, Bs, Cs = None, None, None, None
                
            # 原维度为（B，C，H，W）变成（B，H * W，C）
            # x = self.mlp_proj(identity).permute(0, 2, 3, 1).view(B, H * W, -1)
            F = x
            x = x.permute(0, 2, 3, 1).view(B, H * W, -1)

            # H = W = int(math.sqrt(self.length))
            H = W = self.feat_size
            # SS2D processing
            x = x.view(B, H, W, -1)

            x = self.ln(x)
            x, C = self.block(x, return_param=True, input_params=input_params, dts_input=dts_input, Bs_input=bs_input, Cs_input=cs_input)  # x (74, 4, 4, 512)print("x", x)
            if isinstance(C, list):
                C, dts, Bs, Cs = C  
                outputs.update({'dts': dts, 'Bs': Bs, 'Cs': Cs})
            """Combines outputs from the main and new branches with the identity projection."""

            # 如果输入和输出维度不一致，这里就不能用F直接相加
            # final_result = self.a * F + (1-self.a) * x.view(B, H * W, -1) #(B, L ,D)
            final_result = x.view(B, H * W, -1)
            # 这里不能用D，因为D是输入维度，输出维度是out_channels
            final_result = (final_result.view(B, H, W, -1)).permute(0, 3, 1, 2) #(B, D, H, W)
            # test
            final_result = self.outlinear(final_result)
            # ? directely plus ？
            final_result = 0.5*F + 0.5*final_result

            return final_result, outputs

        else:
            # input (B, D, H, W)
            # "Selective SSM Branch"
            "no_input_mlp+no_pos+layernorm"
            # Extract the last element if input is a tuple (from previous layers).
            if isinstance(x, tuple):
                x = x[-1]

            B, D, H, W = x.shape
            outputs = {}
            C, dts, Bs, Cs = None, None, None, None
                
            # 原维度为（B，C，H，W）变成（B，H * W，C）
            # x = self.mlp_proj(identity).permute(0, 2, 3, 1).view(B, H * W, -1)
            # x = x.permute(0, 2, 3, 1).view(B, H * W, -1)

            # Process the input tensor through MLP projection and add positional embeddings
            # F = x # 是否不应该加上pos_embed或者应该两个都加上
            # x = x + self.pos_embed # 广播机制使pos_embed变成（B，L，D）

            # H = W = int(math.sqrt(self.length))
            # H = W = self.feat_size
            # SS2D processing
            x = x.view(B, H, W, -1)

            x = self.ln(x)

            if labels is not None:
                self.update_mean(x, labels) # 是否在ln后面
                self.calculate_mis_prob()
                mis_mask = self.generate_mis_mask(labels)
                # mis_mask = self.auto_generate_mis_mask(labels)
                # print("mis_mask", mis_mask)
                
            else:
                fake_labels = self.calculate_sim(x)
                # print("fake_labels", fake_labels)
                mis_mask = self.generate_mis_mask(fake_labels)
                # mis_mask = self.auto_generate_mis_mask(fake_labels)

            
            
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

    def calculate_mis_prob(self):
        mean = self.mean.view(self.mean.size(0), -1)
        dis_matrix = torch.cdist(mean, mean).cuda()

        inf_mask = torch.isinf(mean).any(dim=1)  # 返回一个布尔向量，表示哪些行包含 inf
        dis_matrix[inf_mask] = float('inf')  # 设置含 inf 的行
        dis_matrix[:, inf_mask] = float('inf')  # 设置含 inf 的列
        dis_matrix[torch.eye(dis_matrix.size(0), dtype=torch.bool)] = float('inf')
        # print("dis_matrix", dis_matrix)

        easy_misclassification_probs = 1 / (dis_matrix + 1e-10)
        # print("easy_misclassification_probs", easy_misclassification_probs)
        # 将未更新类别的误分类概率设置为 -1（表示不参与排序）
        misclassification_probs = easy_misclassification_probs.max(dim=1)[0]
        misclassification_probs[mean[:, 0].isinf()] = -1  # 假设类别 1 未更新
        self.misclassification_probs = misclassification_probs.detach()
        # print("misclassification_probs", self.misclassification_probs)

    def auto_generate_mis_mask(self, labels):
        B = labels.size(0)
        mis_class = torch.zeros(B).cuda()
        for i in range(B):
            mis_class[i] = self.misclassification_probs[labels[i]].cuda()
        # print("mis_class", mis_class.shape)
        logits = self.generate_topk(mis_class.unsqueeze(1))
        logits = F.softmax(logits, dim=1)
        # print("logits", logits)
        mis_mask = torch.argmax(logits, dim=1)
        return mis_mask+1

    def generate_mis_mask(self, labels):
        B = labels.size(0)
        mis_mask = torch.zeros(B)
        if self.num_classes == 10:
            for i in range(B):
                # param0 
                if self.misclassification_probs[labels[i]] < 0.035:
                    mis_mask[i] = 1
                elif self.misclassification_probs[labels[i]] < 0.045:
                    mis_mask[i] = 2
                elif self.misclassification_probs[labels[i]] < 0.055:
                    mis_mask[i] = 3
                elif self.misclassification_probs[labels[i]] < 0.065:
                    mis_mask[i] = 4
                elif self.misclassification_probs[labels[i]] < 0.075:
                    mis_mask[i] = 5
                elif self.misclassification_probs[labels[i]] < 0.085:
                    mis_mask[i] = 6
                elif self.misclassification_probs[labels[i]] < 0.095:
                    mis_mask[i] = 7
                else:
                    mis_mask[i] = 10
            return mis_mask

        elif self.num_classes == 100:
            for i in range(B):
                # param5 0.469
                if self.misclassification_probs[labels[i]] < 0.030:
                    mis_mask[i] = 1
                elif self.misclassification_probs[labels[i]] < 0.035:
                    mis_mask[i] = 2
                elif self.misclassification_probs[labels[i]] < 0.040:
                    mis_mask[i] = 3
                elif self.misclassification_probs[labels[i]] < 0.045:
                    mis_mask[i] = 4
                elif self.misclassification_probs[labels[i]] < 0.050:
                    mis_mask[i] = 5
                elif self.misclassification_probs[labels[i]] < 0.055:
                    mis_mask[i] = 6
                elif self.misclassification_probs[labels[i]] < 0.060:
                    mis_mask[i] = 7
                else:
                    mis_mask[i] = 8
            return mis_mask
    
        else:
            # for i in range(B):
            #     # param0 0.1985 0.2766 0.2843 0.2903 0.3050 0.3391 0.3442
            #     if self.misclassification_probs[labels[i]] < 0.03:
            #         mis_mask[i] = 1
            #     elif self.misclassification_probs[labels[i]] < 0.04:
            #         mis_mask[i] = 2
            #     elif self.misclassification_probs[labels[i]] < 0.05:
            #         mis_mask[i] = 3
            #     elif self.misclassification_probs[labels[i]] < 0.06:
            #         mis_mask[i] = 4
            #     elif self.misclassification_probs[labels[i]] < 0.08:
            #         mis_mask[i] = 5
            #     elif self.misclassification_probs[labels[i]] < 0.10:
            #         mis_mask[i] = 6
            #     elif self.misclassification_probs[labels[i]] < 0.12:
            #         mis_mask[i] = 7
            #     elif self.misclassification_probs[labels[i]] < 0.15:
            #         mis_mask[i] = 8
            #     elif self.misclassification_probs[labels[i]] < 0.17:
            #         mis_mask[i] = 9
            #     else:
            #         mis_mask[i] = 10
            # return mis_mask
            for i in range(B):
                # param1 0.2023 0.3433 0.342 0.3095 0.2923 0.2882 0.2847
                if self.misclassification_probs[labels[i]] < 0.035:
                    mis_mask[i] = 1
                elif self.misclassification_probs[labels[i]] < 0.045:
                    mis_mask[i] = 2
                elif self.misclassification_probs[labels[i]] < 0.055:
                    mis_mask[i] = 3
                elif self.misclassification_probs[labels[i]] < 0.065:
                    mis_mask[i] = 4
                elif self.misclassification_probs[labels[i]] < 0.075:
                    mis_mask[i] = 5
                elif self.misclassification_probs[labels[i]] < 0.085:
                    mis_mask[i] = 6
                elif self.misclassification_probs[labels[i]] < 0.095:
                    mis_mask[i] = 7
                else:
                    mis_mask[i] = 10
            return mis_mask
            # for i in range(B):
            #     # param2 0.1943
            #     if self.misclassification_probs[labels[i]] < 0.030:
            #         mis_mask[i] = 1
            #     elif self.misclassification_probs[labels[i]] < 0.040:
            #         mis_mask[i] = 2
            #     elif self.misclassification_probs[labels[i]] < 0.045:
            #         mis_mask[i] = 3
            #     elif self.misclassification_probs[labels[i]] < 0.050:
            #         mis_mask[i] = 4
            #     elif self.misclassification_probs[labels[i]] < 0.055:
            #         mis_mask[i] = 5
            #     elif self.misclassification_probs[labels[i]] < 0.060:
            #         mis_mask[i] = 6
            #     elif self.misclassification_probs[labels[i]] < 0.065:
            #         mis_mask[i] = 7
            #     elif self.misclassification_probs[labels[i]] < 0.070:
            #         mis_mask[i] = 8
            #     elif self.misclassification_probs[labels[i]] < 0.075:
            #         mis_mask[i] = 9
            #     else:
            #         mis_mask[i] = 10
            # return mis_mask
            # for i in range(B):
            #     # param5 0.2024 0.3191 0.3275 0.3081 0.2771 0.2871 0.2702
            #     if self.misclassification_probs[labels[i]] < 0.040:
            #         mis_mask[i] = 1
            #     elif self.misclassification_probs[labels[i]] < 0.070:
            #         mis_mask[i] = 2
            #     elif self.misclassification_probs[labels[i]] < 0.090:
            #         mis_mask[i] = 3
            #     elif self.misclassification_probs[labels[i]] < 0.110:
            #         mis_mask[i] = 4
            #     elif self.misclassification_probs[labels[i]] < 0.130:
            #         mis_mask[i] = 5
            #     elif self.misclassification_probs[labels[i]] < 0.150:
            #         mis_mask[i] = 6
            #     elif self.misclassification_probs[labels[i]] < 0.170:
            #         mis_mask[i] = 7
            #     else:
            #         mis_mask[i] = 8
            # return mis_mask

    def calculate_sim(self, x):        
        # 计算每个样本与每个类别的余弦相似度
        x = x.view(x.size(0), -1)  # 将特征图展平
        mean_flat = self.mean.view(self.num_classes, -1)
        similarity_matrix = torch.mm(x, mean_flat.t())  # 形状为 (batch_size, num_classes)
        
        # 找到每个样本的最大相似度对应的类别索引，作为伪标签
        pseudo_labels = torch.argmax(similarity_matrix, dim=1)
        
        return pseudo_labels