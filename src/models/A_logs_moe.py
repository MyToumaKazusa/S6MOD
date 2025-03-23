import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat


class MoESystem(nn.Module):
    def __init__(self, num_experts, top_k, d_state, input_dim, copies, merge, K, dt_rank):
        super(MoESystem, self).__init__()
        print("MoESystem: ", num_experts, top_k)
        # Initialize gating (router) and experts
        self.num_experts = num_experts
        self.router = NoisyTopkRouter(input_dim, num_experts, top_k)
        self.experts = A_logs_MoEProjectionLayer(num_experts, top_k, d_state, input_dim, copies, merge, K, dt_rank)

    def forward(self, inputs):
        # Pass through the router to get expert weights and indices
        router_output, indices, logits = self.router(inputs)
        # Use the experts based on router output
        expert_outputs = self.experts(indices).cuda()
        # Combine expert outputs based on the router's weights
        combined_output = torch.einsum('bnik,bn->bik', expert_outputs, router_output).mean(dim=0)

        # Calculate auxiliary loss for load balancing
        # load_balancing_loss = self.calculate_load_balancing_loss(logits)
        # return combined_output, load_balancing_loss

        z_loss = self.calculate_z_loss(logits)
        return combined_output, z_loss

    def calculate_load_balancing_loss(self, g):
        # x: a tensor of shape (T, N), where T is the number of tokens, N is the number of experts
        # g: a tensor of shape (T, N), representing gating probabilities
        T, N = g.shape
        
        # Step 1: Calculate D_i (Equation 2.6)
        D = torch.zeros(N, dtype=torch.float32).cuda()
        for i in range(N):
            D[i] = torch.sum(torch.argmax(g, dim=1) == i).float().cuda() / T  # Fraction of tokens assigned to expert i
        
        # Step 2: Calculate P_i (Equation 2.7)
        P = torch.mean(g, dim=0).cuda()  # Average probability assigned to each expert, shape: (N,)
        
        # Step 3: Calculate load balancing loss (Equation 2.5)
        L_load_balancing = N * torch.sum(D * P)
        
        return L_load_balancing

    def calculate_z_loss(self, logits):
        # Calculate z-loss as per equation (5) in the provided image
        B, N = logits.size()
        z_loss = (1 / B) * torch.sum((torch.log(torch.sum(torch.exp(logits), dim=-1))) ** 2)
        return z_loss


class NoisyTopkRouter(nn.Module):
    # Noisy Top-k Gating Layer
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)
    
    def forward(self, mh_output):
        # Compute logits and add noise for the noisy top-k gating mechanism
        # x (b, d, h, w) -> (b, d)
        mh_output = mh_output.mean(dim=(2, 3))  # Reducing dimensions to prepare input for gate
        logits = self.topkroute_linear(mh_output)
        logits = F.softmax(logits, dim=-1)
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noise = F.softmax(noise, dim=-1)
        noisy_logits = logits + noise
        # noisy_logits = logits

        # print("logits: ", F.softmax(logits, dim=-1))
        # print("noise: ", F.softmax(noise, dim=-1))
        # print("noisy_logits: ", noisy_logits)

        # Select top-k experts
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        router_output = F.softmax(top_k_logits, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        sparse_logits = F.softmax(sparse_logits, dim=-1)
        # print("router_output: ", router_output)
        return router_output.cuda(), indices.cuda(), noisy_logits.cuda()  # (B, top_k), (B, top_k) # TODO: add "cuda()"


class A_logs_MoEProjectionLayer(nn.Module):
    def __init__(self, num_experts, top_k, d_state, input_dim, copies, merge, K, dt_rank):
        super(A_logs_MoEProjectionLayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_state = d_state
        self.input_dim = input_dim
        self.copies = copies
        self.merge = merge
        self.K = K
        self.dt_rank = dt_rank

        # Create expert parameters
        self.A_logs_experts = self.create_A_logs_experts(d_state, input_dim, copies, num_experts, merge)

    @staticmethod
    def create_A_logs_experts(d_state, input_dim, copies, num_experts, merge, device=None):
        # Create A_log for each expert in the MoE system
        A_logs_experts = []
        for _ in range(num_experts):
            A_log = A_logs_MoEProjectionLayer.A_log_init(d_state, input_dim, copies, device, merge)
            A_logs_experts.append(A_log)
        return nn.ParameterList(A_logs_experts)

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # Initialize A_log parameters
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    def forward(self, indices):
        # Use the selected experts based on indices from the router
        batch_size = indices.size(0)
        selected_experts = []
        for b in range(batch_size):
            expert_indices = indices[b]  # (top_k,)
            selected_experts.append([self.A_logs_experts[i] for i in expert_indices])
        
        outputs = []
        for b in range(batch_size):
            expert_output = []
            for expert in selected_experts[b]:
                expert_output.append(expert)  # Example transformation
            outputs.append(torch.stack(expert_output, dim=0))
        
        return torch.stack(outputs, dim=0)  # Stack along the batch dimension
