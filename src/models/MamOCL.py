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


@NECKS.register_module()
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
        # print("x:",x)
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
        W = torch.bmm(x_flatten, x_new_flatten.transpose(1, 2))
        result = torch.bmm(F, W) # (B, L, D) (74, 16, 512)
        # decline L dim by averaging
        mean_result = result.mean(dim=1)  # (B, D)   # 用平均是否合理，是否应该用（-1）保留维度进行后续处理
        # print(mean_result.shape)
        # mean_result = self.avg(result).view(B, -1)
        
        # return outputs
        final_result = self.a * F.mean(dim=1) + (1-self.a) * mean_result
        # return F.mean(dim=1)
        return final_result
