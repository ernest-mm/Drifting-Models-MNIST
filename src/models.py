import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU activation layer: Swish(x_gate) * x_up
    """
    def forward(self, x):
        # Expects a tensor split in half along the last dimension
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class AdaLNZero(nn.Module):
    """
    Implements the adaLN-zero parameter generation tracking mechanism.
    Predicts scale, shift, and gating parameters from a conditioning vector.
    """
    def __init__(self, embed_dim):
        super().__init__()
        # 6 parameters: 2 for Attn (scale/shift), 2 for MLP (scale/shift), 2 for Gating scales
        self.linear = nn.Linear(embed_dim, 6 * embed_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond):
        gamma1, beta1, gamma2, beta2, scale1, scale2 = self.linear(cond).chunk(6, dim=-1)
        return (gamma1, beta1, gamma2, beta2, scale1, scale2)


class LatentDiT(nn.Module):
    """
    A streamlined, vector-space Diffusion Transformer that models 
    the drifting field explicitly on compressed latent features.
    """
    def __init__(self, latent_dim=16, embed_dim=128, num_heads=4, depth=4, num_classes=11):
        super().__init__()
        
        # Map our flat size-16 latent vector up to the processing dimensions
        self.input_proj = nn.Linear(latent_dim, embed_dim)
        
        # Time/Alpha continuous scalar embedder
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Categorical class identity embedder
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        
        # Transformer Layer stacks
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6),
                AdaLNZero(embed_dim),
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6),
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4), # Generates 2x hidden states for the chunk operation
                    SwiGLU(),
                    nn.Linear(embed_dim * 2, embed_dim)
                )
            ]) for _ in range(depth)
        ])
        
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, embed_dim * 2))
        self.output_proj = nn.Linear(embed_dim, latent_dim)

    def forward(self, z, labels, alpha):
        # z: [B, 16] Latent vectors
        B = z.size(0)
        
        # Project inputs to tokens: treat the dimension as a sequence of length 1
        x = self.input_proj(z).unsqueeze(1) # [B, 1, embed_dim]
        
        # Blend class information and alpha guidance context
        cond = self.time_embed(alpha) + self.class_embed(labels) # [B, embed_dim]
        
        # Pass tokens through your Transformer Blocks
        for ln1, adaln, mha, ln2, fwd in self.blocks:
            # 1. Attention Stream
            g1, b1, g2, b2, s1, s2 = adaln(cond)
            norm_x = ln1(x)
            modulated_x = norm_x * (1 + g1.unsqueeze(1)) + b1.unsqueeze(1)
            attn_out, _ = mha(modulated_x, modulated_x, modulated_x)
            x = x + s1.unsqueeze(1) * attn_out
            
            # 2. Feed-Forward Stream
            norm_x2 = ln2(x)
            modulated_x2 = norm_x2 * (1 + g2.unsqueeze(1)) + b2.unsqueeze(1)
            x = x + s2.unsqueeze(1) * fwd(modulated_x2)
            
        # Final prediction scaling back down to latent coordinates
        x = self.final_norm(x)
        scale, shift = self.final_adaLN(cond).chunk(2, dim=-1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        return self.output_proj(x).squeeze(1) # [B, 16]