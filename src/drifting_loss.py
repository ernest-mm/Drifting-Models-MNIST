import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DriftingLoss(nn.Module):
    def __init__(self, temperatures=(0.05, 0.1, 0.2)):
        super().__init__()
        self.temperatures = tuple(temperatures)

    def compute_drift_field_single_tau(self, x, y_pos, y_neg, tau):
        """
        Computes Drifting Field V following Algorithm 2 of Deng et al. (2026).
        """
        N, D = x.shape
        N_pos = y_pos.shape[0]
        N_neg = y_neg.shape[0]

        # 1. Pairwise L2 distances
        dist_pos = torch.cdist(x, y_pos, p=2)  # [N, N_pos]
        dist_neg = torch.cdist(x, y_neg, p=2)  # [N, N_neg]

        # Self-distance masking for negative samples (generator outputs)
        eye_mask = torch.eye(N, device=x.device) * 1e6
        dist_neg = dist_neg + eye_mask

        # 2. Temperature scaling
        logit_pos = -dist_pos / tau
        logit_neg = -dist_neg / tau

        # Concatenate along candidate axis
        logit = torch.cat([logit_pos, logit_neg], dim=1)  # [N, N_pos + N_neg]

        # 3. Dual Softmax Normalization
        A_row = F.softmax(logit, dim=-1)
        A_col = F.softmax(logit, dim=-2)
        A = torch.sqrt(A_row * A_col + 1e-12)

        # Split back to positive and negative matrices
        A_pos, A_neg = torch.split(A, [N_pos, N_neg], dim=1)

        # 4. Mass-balanced weighting
        W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
        W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

        # 5. Drift Vector V = Pos_Attraction - Neg_Repulsion
        drift_pos = torch.matmul(W_pos, y_pos)
        drift_neg = torch.matmul(W_neg, y_neg)

        return drift_pos - drift_neg

    def forward(self, x, y_pos):
        B, D = x.shape
        
        # Generator outputs are used as negative samples
        y_neg = x.detach()

        # Global distance normalization scaling S_j
        with torch.no_grad():
            cross_dist = torch.cdist(x, y_pos, p=2)
            S_j = cross_dist.mean() / math.sqrt(D)
            S_j = torch.clamp(S_j, min=1e-6)

        x_norm = x / S_j
        y_pos_norm = y_pos / S_j
        y_neg_norm = y_neg / S_j

        V_aggregated = torch.zeros_like(x_norm)

        for tau_base in self.temperatures:
            tau_tilde = max(tau_base * math.sqrt(D), 1e-6)
            V_aggregated += self.compute_drift_field_single_tau(x_norm, y_pos_norm, y_neg_norm, tau_tilde)

        V_final = (V_aggregated / len(self.temperatures)) * S_j
        return V_final