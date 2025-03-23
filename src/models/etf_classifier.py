import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from numbers import Number
import torchvision.transforms.functional as TF
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
from PIL import ImageFilter

# 该函数用于计算预测结果与实际目标之间的匹配度
def dot_regression_accuracy(preds, targets, topk=1, thr=0.):
    preds = preds.float()
    pred_scores, pred_labels = preds.topk(topk, dim=1)
    pred_labels = pred_labels.t()
    
    corrects = pred_labels.eq(targets.view(1, -1).expand_as(pred_labels))
    corrects = corrects & (pred_scores.t() > thr)
    return corrects.squeeze()

# 该函数用于初始化ETF分类器的权重矩阵
def etf_initialize(in_channel, num_classes):
    orth_vec = generate_random_orthogonal_matrix(in_channel, num_classes)
    i_nc_nc = torch.eye(num_classes)
    one_nc_nc: torch.Tensor = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))
    etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                        math.sqrt(num_classes / (num_classes - 1)))
    print("ETF Classifier Shape", etf_vec.shape)
    return etf_vec

# 类似于etf_initialize，但它直接使用一个预先生成的正交矩阵orth_vec
def dynamic_etf_initialize(num_classes, orth_vec):
    i_nc_nc = torch.eye(num_classes)
    one_nc_nc: torch.Tensor = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))
    etf_vec = torch.mul(torch.matmul(orth_vec[:, :num_classes], i_nc_nc - one_nc_nc),
                        math.sqrt(num_classes / (num_classes - 1)))
    return etf_vec

# 生成一个随机的正交矩阵
def generate_random_orthogonal_matrix(in_channel, num_classes):
    rand_mat = np.random.random(size=(in_channel, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    '''
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    '''
    return orth_vec

class DotRegressionLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 reg_lambda=0.
                 ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_lambda = reg_lambda

    def forward(self,
                feat,
                target,
                pure_num=None,
                augmented_num=None,
                h_norm2=None,
                m_norm2=None,
                avg_factor=None,
                ):
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)

        if self.reduction == "mean":
            if augmented_num is None:
                loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)
            else:
                loss = ((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2
                loss = 0.5 * ((torch.mean(loss[:pure_num]) + torch.mean(loss[pure_num:])) / 2)

        elif self.reduction == "none":
            loss = 0.5 * (((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)

        return loss * self.loss_weight


