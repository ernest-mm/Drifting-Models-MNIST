import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Position Embeddings (RoPE) as specified in Section A.2.
    RoPE injects positional information directly into the Query and Key states.
    """
    def __init__(self, dim, max_seq_len=256):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, self.inv_freq)
        # Separate sine and cosine tracks
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        # x shape: [B, N, num_heads, d_head]
        seq_len = x.size(1)
        cos = self.cos[:seq_len, None, :] # [N, 1, d_head//2]
        sin = self.sin[:seq_len, None, :] # [N, 1, d_head//2]
        # Repeat frequencies to match head dimensions
        cos = cos.repeat(1, 1, 2)
        sin = sin.repeat(1, 1, 2)
        return x * cos + self._rotate_half(x) * sin


class AdaLNZero(nn.Module):
    """
    Implements the adaLN-zero block from DiT (Peebles & Xie, 2023), 
    used for processing class-conditioning and CFG alpha values (Section 4).
    It predicts scale and shift parameters for standard RMSNorm layers.
    """
    def __init__(self, embed_dim, cond_dim):
        super().__init__()
        # Dictates modulation parameters: 2 for attention (scale/shift), 
        # 2 for MLP (scale/shift), and 2 for gating scales. Total = 6 parameters.
        self.linear = nn.Linear(cond_dim, 6 * embed_dim)
        # Initialize final projection to zero so block acts as an identity function at start
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, cond):
        # x: [B, N, embed_dim], cond: [B, cond_dim]
        gamma1, beta1, gamma2, beta2, scale1, scale2 = self.linear(cond).chunk(6, dim=-1)
        return (gamma1, beta1, gamma2, beta2, scale1, scale2)


class DiTBlock(nn.Module):
    """
    A unified modern Transformer block utilizing SwiGLU, QK-Norm, RMSNorm,
    and adaLN-zero conditioning modulation tracks (Section A.2).
    """
    def __init__(self, embed_dim, num_heads, cond_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Norms and Attention
        self.norm1 = nn.RMSNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim) # QK-Norm layer
        self.k_norm = nn.RMSNorm(self.head_dim) # QK-Norm layer
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Modulation Block
        self.adaLN = AdaLNZero(embed_dim, cond_dim)
        
        # SwiGLU MLP implementation (Section A.2: "Following Yao et al., we use SwiGLU")
        self.norm2 = nn.RMSNorm(embed_dim)
        hidden_dim = int(2 * (4 * embed_dim) / 3) # Standard SwiGLU compression size
        self.mlp_gate = nn.Linear(embed_dim, hidden_dim)
        self.mlp_up = nn.Linear(embed_dim, hidden_dim)
        self.mlp_down = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, cond, rope):
        # Compute modulation variables using adaptive layer normalization
        g1, b1, g2, b2, s1, s2 = self.adaLN(cond)
        
        # --- Multi-Head Self-Attention Track ---
        res = x
        # Apply scale and shift parameters to the pre-norm data
        normed_x = self.norm1(x) * (1 + g1.unsqueeze(1)) + b1.unsqueeze(1)
        
        B, N, C = normed_x.shape
        qkv = self.qkv(normed_x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, B, N, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # Shapes: [B, N, num_heads, head_dim]
        
        # Apply QK-Norm to stabilize high-dimensional gradient fields
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Inject Rotary Position Embeddings (RoPE)
        q = rope(q).permute(0, 2, 1, 3) # [B, num_heads, N, head_dim]
        k = rope(k).permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # Flash / Scaled Dot-Product Attention
        attn_out = F.scaled_dot_product_attention(q, k, v) # [B, num_heads, N, head_dim]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, C)
        attn_out = self.proj(attn_out)
        
        # Residual gating apply scale1 parameter
        x = res + s1.unsqueeze(1) * attn_out
        
        
        # --- SwiGLU MLP Track ---
        res = x
        normed_x = self.norm2(x) * (1 + g2.unsqueeze(1)) + b2.unsqueeze(1)
        
        # SwiGLU execution: Swish(gate) * up
        mlp_hidden = F.silu(self.mlp_gate(normed_x)) * self.mlp_up(normed_x)
        mlp_out = self.mlp_down(mlp_hidden)
        
        # Residual gating apply scale2 parameter
        x = res + s2.unsqueeze(1) * mlp_out
        return x


class MiniatureDiT(nn.Module):
    """
    The main generator network (f_theta) designed following the 
    Diffusion Transformer (DiT) framework mapping noise inputs directly 
    to pixel outputs (Section 4 & A.2).
    """
    def __init__(self, img_size=28, in_channels=1, patch_size=2, embed_dim=128, num_heads=4, depth=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Token patchification framework (Section A.2: "Input noise is patchified into tokens")
        num_patches = (img_size // patch_size) ** 2
        patch_dim = (patch_size ** 2) * in_channels
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        
        # Conditioning Tracks (Class 0-9 and continuous CFG scale alpha value)
        cond_dim = embed_dim
        self.class_embed = nn.Embedding(11, cond_dim) # 10 digits + 1 for unconditional null mask token
        self.alpha_embed = nn.Linear(1, cond_dim)      # Continuous mapping for CFG alpha strength
        
        # Combined conditioning mapping layer
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # In-Context tokens (Section A.2: "We prepend 16 learnable tokens for in-context conditioning")
        self.num_in_context = 16
        self.in_context_tokens = nn.Parameter(torch.randn(1, self.num_in_context, embed_dim))
        
        # RoPE Position tracker
        self.rope = RotaryEmbedding(dim=embed_dim // num_heads, max_seq_len=num_patches + self.num_in_context)
        
        # Core Stack of Transformer Layers
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, cond_dim) for _ in range(depth)
        ])
        
        # Final pixel unpatchification projection layer
        self.final_norm = nn.RMSNorm(embed_dim)
        self.final_proj = nn.Linear(embed_dim, patch_dim)

    def forward(self, epsilon, c, alpha):
        """
        Args:
            epsilon: Random input variables tensor, shape [B, 1, 28, 28]
            c: Conditional class tensor (integers 0-9), shape [B]
            alpha: Classifier-Free Guidance scales tensor, shape [B, 1]
            
        Returns:
            x: Reconstructed pixel-space generation tensor, shape [B, 1, 28, 28]
        """
        B, C, H, W = epsilon.shape
        P = self.patch_size
        
        # 1. Patchify the continuous input noise grid into sequential 1D vector tokens
        # Output shape after permutation: [B, (H/P)*(W/P), P*P*C]
        x = epsilon.unfold(2, P, P).unfold(3, P, P)
        x = x.contiguous().view(B, C, -1, P, P).permute(0, 2, 3, 4, 1).contiguous().view(B, -1, (P**2)*C)
        x = self.patch_embed(x) # Shape: [B, num_patches, embed_dim]
        
        # 2. Extract and fuse class-conditioning and CFG scaling tensors
        c_emb = self.class_embed(c)    # Shape: [B, embed_dim]
        a_emb = self.alpha_embed(alpha) # Shape: [B, embed_dim]
        cond = self.cond_mlp(c_emb + a_emb) # Unified conditioning vector, shape [B, embed_dim]
        
        # 3. Handle In-Context Conditioning Tokens (Section A.2)
        # Prepend learnable template tokens modded by global conditioning features
        ctx = self.in_context_tokens.repeat(B, 1, 1) + cond.unsqueeze(1)
        x = torch.cat([ctx, x], dim=1) # Shape: [B, num_in_context + num_patches, embed_dim]
        
        # 4. Forward execution through multi-scale DiT layer blocks
        for block in self.blocks:
            x = block(x, cond, self.rope)
            
        # 5. Extract patch dimensions, skipping prepend contextual prefix channels
        x = x[:, self.num_in_context:] # Shape: [B, num_patches, embed_dim]
        x = self.final_norm(x)
        x = self.final_proj(x)        # Convert back to raw patch sequence [B, num_patches, P*P*C]
        
        # 6. Unpatchify vectors back to continuous image grids [B, 1, 28, 28]
        num_patches_side = H // P
        x = x.view(B, num_patches_side, num_patches_side, P, P, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
        return x