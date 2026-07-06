import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(norm_x + self.eps))


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class AdaLNZero(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # 6 modulation parameters: (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        self.linear = nn.Linear(embed_dim, 6 * embed_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond):
        return self.linear(cond).chunk(6, dim=-1)


class QKNormAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        
        return self.proj(attn_out)


class LatentDiTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = QKNormAttention(embed_dim, num_heads)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), 
            SwiGLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.adaln = AdaLNZero(embed_dim)

    def forward(self, x, cond):
        g1, b1, s1, g2, b2, s2 = self.adaln(cond)
        
        # 1. Attention Stream with AdaLN modulation
        # Modulate input sequence features
        modulated_x = self.norm1(x) * (1 + g1.unsqueeze(1)) + b1.unsqueeze(1)
        # Apply scaled residual gate connection
        x = x + s1.unsqueeze(1) * self.attn(modulated_x)
        
        # 2. Feed-Forward Stream with AdaLN modulation
        modulated_x2 = self.norm2(x) * (1 + g2.unsqueeze(1)) + b2.unsqueeze(1)
        x = x + s2.unsqueeze(1) * self.ffn(modulated_x2)
        
        return x


class LatentDiT(nn.Module):
    def __init__(self, latent_dim=16, embed_dim=128, num_heads=4, depth=4, num_classes=10):
        super().__init__()
        
        self.input_proj = nn.Linear(latent_dim, embed_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.class_embed = nn.Embedding(num_classes, embed_dim)

        self.num_context_tokens = 16
        self.context_tokens = nn.Parameter(torch.randn(1, self.num_context_tokens, embed_dim))
        
        self.blocks = nn.ModuleList([
            LatentDiTBlock(embed_dim, num_heads) for _ in range(depth)
        ])

        self.final_norm = RMSNorm(embed_dim)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, embed_dim * 2))
        self.output_proj = nn.Linear(embed_dim, latent_dim)

    def forward(self, z, labels, alpha):
        B = z.size(0)

        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)
        alpha = alpha.to(z.dtype)
        labels = labels.long()

        x_data = self.input_proj(z).unsqueeze(1)
        cond = self.time_embed(alpha) + self.class_embed(labels)

        ctx = self.context_tokens.expand(B, -1, -1)
        x = torch.cat([ctx, x_data], dim=1)

        for block in self.blocks:
            x = block(x, cond)

        x_data_out = x[:, -1:]
        x_data_out = self.final_norm(x_data_out)
        scale, shift = self.final_adaLN(cond).chunk(2, dim=-1)
        x_data_out = x_data_out * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        return self.output_proj(x_data_out).squeeze(1)

    def sample(self, noise, labels, strength):
        return self.forward(noise, labels, strength)