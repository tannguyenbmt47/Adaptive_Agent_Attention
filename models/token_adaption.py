import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TokenSparse(nn.Module):
    """
    Chọn lọc token quan trọng nhất dựa trên attention score.
    Giữ lại top-k token, phần còn lại gộp thành 1 fusion token.
    """
    def __init__(self, embed_dim=512, sparse_ratio=0.6):
        super().__init__()
        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio

    def forward(self, tokens, attention_x, attention_y):
        B, L, C = tokens.size()
        score = attention_x + attention_y  # (B, L)
        num_keep_token = math.ceil(L * self.sparse_ratio)
        score_sort, score_index = torch.sort(score, dim=1, descending=True)
        keep_policy = score_index[:, :num_keep_token]  # (B, k)
        score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1)
        select_tokens = torch.gather(tokens, dim=1, index=keep_policy.unsqueeze(-1).expand(-1, -1, C))
        non_keep_policy = score_index[:, num_keep_token:]
        non_tokens = torch.gather(tokens, dim=1, index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C))
        non_keep_score = score_sort[:, num_keep_token:]
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True)  # (B, 1, C)
        return select_tokens, extra_token, score_mask

class TokenAggregation(nn.Module):
    """
    Gộp các token đã chọn thành super-token bằng trọng số học được.
    """
    def __init__(self, dim=512, keeped_patches=64, dim_ratio=0.2):
        super().__init__()
        hidden_dim = int(dim * dim_ratio)
        self.weight = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, keeped_patches)
        )
        self.scale = nn.Parameter(torch.ones(1, 1, 1))

    def forward(self, x, keep_policy=None):
        weight = self.weight(x)  # (B, N, N_s)
        weight = weight.transpose(2, 1) * self.scale  # (B, N_s, N)
        if keep_policy is not None:
            keep_policy = keep_policy.unsqueeze(1)
            weight = weight - (1 - keep_policy) * 1e10
        weight = F.softmax(weight, dim=2)
        x = torch.bmm(weight, x)  # (B, N_s, C)
        return x

class TokenAdaptionModule(nn.Module):
    """
    Module tổng hợp: Token Sparsification + Aggregation
    """
    def __init__(self, embed_dim=512, num_patches=196, sparse_ratio=0.5, aggr_ratio=0.4, dim_ratio=0.2):
        super().__init__()
        self.sparse = TokenSparse(embed_dim=embed_dim, sparse_ratio=sparse_ratio)
        keeped_patches = int(num_patches * aggr_ratio * sparse_ratio)
        self.aggregation = TokenAggregation(dim=embed_dim, keeped_patches=keeped_patches, dim_ratio=dim_ratio)

    def forward(self, tokens, attention_x, attention_y):
        select_tokens, extra_token, score_mask = self.sparse(tokens, attention_x, attention_y)
        all_tokens = torch.cat([select_tokens, extra_token], dim=1)
        super_tokens = self.aggregation(all_tokens)
        return super_tokens, score_mask
