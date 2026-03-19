import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class AgentAttention(nn.Module):
    """
    Agent Attention: On the Integration of Softmax and Linear Attention (ECCV 2024)

    Thay vì Q attend trực tiếp vào K (O(N^2)), dùng agent tokens A làm trung gian:
      1. A attend vào K,V → agent_v       (O(n*N))
      2. Q attend vào A   → output        (O(N*n))
    Tổng: O(N*n) với n << N

    Args:
        dim: embedding dimension
        num_heads: number of attention heads
        qkv_bias: enable bias for QKV projection
        attn_drop: attention dropout rate
        proj_drop: output projection dropout rate
        agent_num: number of agent tokens (default 49 = 7x7)
        window: spatial size of feature map (H or W in tokens)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 agent_num=49, window=14):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num
        self.window = window

        # Depthwise convolution — bổ sung thông tin cục bộ vào output
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim,
                             kernel_size=3, padding=1, groups=dim)

        # Positional bias: agent → key
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))

        # Positional bias: query → agent
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))

        # Positional bias cho cls token
        self.ac_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1))
        self.ca_bias = nn.Parameter(torch.zeros(1, num_heads, 1, agent_num))

        # Khởi tạo bias
        for bias in [self.an_bias, self.na_bias, self.ah_bias, self.aw_bias,
                     self.ha_bias, self.wa_bias, self.ac_bias, self.ca_bias]:
            trunc_normal_(bias, std=.02)

        # Average pooling: Q tokens → agent tokens
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

    def forward(self, x):
        """
        Args:
            x: (B, N, C) — N = 1 + H*W (cls token + image tokens)

        Returns:
            (B, N, C)
        """
        b, n, c = x.shape
        h = w = int((n - 1) ** 0.5)  # spatial size (bỏ cls token)
        num_heads = self.num_heads
        head_dim = c // num_heads

        # --- Project Q, K, V ---
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b, n, c)

        # --- Tạo agent tokens bằng pooling Q (bỏ cls token) ---
        agent_tokens = self.pool(
            q[:, 1:, :].reshape(b, h, w, c).permute(0, 3, 1, 2)
        ).reshape(b, c, -1).permute(0, 2, 1)  # (b, agent_num, c)

        # Reshape thành multi-head: (b, num_heads, tokens, head_dim)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        # --- Bước 1: Agent attend vào K → agent_v ---
        # Position bias: nội suy 7×7 → H×W, cộng factorized bias
        position_bias1 = F.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        position_bias = torch.cat([self.ac_bias.repeat(b, 1, 1, 1), position_bias], dim=-1)

        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v  # (b, heads, agent_num, head_dim)

        # --- Bước 2: Q attend vào agent tokens → output ---
        agent_bias1 = F.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        agent_bias = torch.cat([self.ca_bias.repeat(b, 1, 1, 1), agent_bias], dim=-2)

        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v  # (b, heads, N, head_dim)

        # --- Reshape + DWC shortcut ---
        x = x.transpose(1, 2).reshape(b, n, c)
        v_ = v[:, :, 1:, :].transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x[:, 1:, :] = x[:, 1:, :] + self.dwc(v_).permute(0, 2, 3, 1).reshape(b, n - 1, c)

        # --- Output projection ---
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
