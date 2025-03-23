import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils.rnn import pad_sequence

class MoEProjectionLayer(nn.Module):
    def __init__(self, num_experts, input_dim, K, dt_rank, d_state, **factory_kwargs):
        super(MoEProjectionLayer, self).__init__()
        self.num_experts = num_experts
        self.K = K
        self.dt_rank = dt_rank
        self.d_state = d_state

        self.x_proj_weight = nn.Parameter(torch.stack([
            torch.stack([
                nn.Linear(input_dim, (self.dt_rank), bias=False, **factory_kwargs).weight
                for _ in range(self.K)
            ], dim=0) 
            for _ in range(self.num_experts)
        ], dim=0))  

        self.gate = NoisyTopkRouter(n_embed=input_dim, num_experts=num_experts)
    
    def forward(self, xs, mis_mask):
        batch_size, k, d, l = xs.size()
        masked_xs_flat = xs.permute(0, 1, 3, 2)  # (b, k, l, d)
        gate_weights, logits = self.gate(masked_xs_flat, mis_mask)  # gate_weights: (b, num_experts), topk_indices: (b, top_k)
        final_output = torch.zeros(batch_size, k, 16, l).cuda()
        # 遍历每个专家
        for i in range(self.num_experts):  # 使用 experts 的数量
            selected_expert_weight = self.x_proj_weight[i]  # (k, c, d)
            x_proj = torch.einsum("b k d l, k c d -> b k c l", xs, selected_expert_weight)
            gating_scores = gate_weights[..., i]  # (b)
            # print("gating_scores:", gating_scores)
            # (b, k, c, l) * (b,1,1,1)
            weighted_output = x_proj * gating_scores.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (b, k, c, l)
            final_output += weighted_output

        z_loss = self.calculate_z_loss(logits)
        return final_output, z_loss

    def calculate_z_loss(self, logits):
        # Calculate z-loss as per equation (5) in the provided image
        B, N = logits.size()
        z_loss = (1 / B) * torch.sum((torch.log(torch.sum(torch.exp(logits), dim=-1))) ** 2)
        return z_loss

class NoisyTopkRouter(nn.Module):
    # class gate
    def __init__(self, n_embed, num_experts):
        super(NoisyTopkRouter, self).__init__()
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear =nn.Linear(n_embed, num_experts)
    
    def forward(self, mh_output, mis_mask):
        # topk
        mh_output = mh_output.mean(dim=(1, 2))
        logits = self.topkroute_linear(mh_output)
        logits = F.softmax(logits, dim=-1)
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noise = F.softmax(noise, dim=-1)
        noisy_logits = logits + noise

        # 创建一个用于存储 top_k_logits 和 indices 的空列表
        top_k_logits_list = []
        indices_list = []

        # 遍历每个样本的 mis_mask
        for i in range(noisy_logits.size(0)):
            k = int(mis_mask[i].item())  # 获取当前样本的 top_k 值
            top_k_logit, index = noisy_logits[i].topk(k, dim=-1)  # 获取 top_k
            top_k_logits_list.append(top_k_logit)
            indices_list.append(index)

        # 将列表转换为张量，使用 pad_sequence 进行填充
        top_k_logits_padded = pad_sequence(top_k_logits_list, batch_first=True, padding_value=0.0)
        indices_padded = pad_sequence(indices_list, batch_first=True, padding_value=0)

        # 创建用于 scatter 的张量，并处理填充值
        zeros = torch.zeros_like(noisy_logits)
        for i in range(zeros.size(0)):
            valid_indices = indices_padded[i] >= 0  # 获取有效的索引
            zeros[i].scatter_(-1, indices_padded[i][valid_indices], top_k_logits_padded[i][valid_indices])
        
        sparse_logits = zeros
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output.cuda(), noisy_logits.cuda()  # (B, num_experts)




































# class NoisyTopkRouter(nn.Module):
#     # token gate
#     def __init__(self, n_embed, num_experts, top_k):
#         super(NoisyTopkRouter, self).__init__()
#         self.top_k = top_k
#         self.topkroute_linear = nn.Linear(n_embed, num_experts)
#         self.noise_linear =nn.Linear(n_embed, num_experts)

#         # torch.nn.init.kaiming_uniform_(self.topkroute_linear.weight, nonlinearity='relu')
#         # torch.nn.init.kaiming_uniform_(self.noise_linear.weight, nonlinearity='relu')

    
#     def forward(self, mh_output):
#         # mh_ouput is the output tensor from multihead self attention block
#         logits = self.topkroute_linear(mh_output)

#         noise_logits = self.noise_linear(mh_output)

#         #Adding scaled unit gaussian noise to the logits
#         noise = torch.randn_like(logits)*F.softplus(noise_logits)
#         noisy_logits = logits + noise

#         # print("logits:", logits.shape)
#         # print("noise:", noise.shape)
#         print("noise_logits:", noise_logits.shape)
        
#         # noisy_logits = logits #去掉noise

#         top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
#         zeros = torch.full_like(noisy_logits, float('-inf'))
#         sparse_logits = zeros.scatter(-1, indices, top_k_logits)
#         router_output = F.softmax(sparse_logits, dim=-1)
#         return router_output, indices # (B, num_experts), (B, top_k)
