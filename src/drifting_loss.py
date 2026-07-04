import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DriftingLoss(nn.Module):
    def __init__(self, temperatures=[0.02, 0.05, 0.2]):
        super().__init__()
        self.temperatures = temperatures

    def compute_V_at_temperature(self, x, y_pos, y_neg, tau):
        N = x.size(0)
        N_pos = y_pos.size(0)
        N_neg = y_neg.size(0)

        # 1. Compute Pairwise Euclidean (L2) Distance Matrices
        dist_pos = torch.cdist(x, y_pos, p=2) 
        dist_neg = torch.cdist(x, y_neg, p=2) 

        # 2. Ignore Self-Interaction out-of-place (Prevents Autograd runtime errors)
        if torch.tensor([y_neg.data_ptr() == x.data_ptr()], dtype=torch.bool, device=x.device).item():
            dist_neg = dist_neg + torch.eye(N, device=x.device) * 1e6

        # 3. Scale by temperature and convert distances to similarity logits
        logit_pos = -dist_pos / tau
        logit_neg = -dist_neg / tau 

        # 4. Concatenate for Joint Normalization
        logit = torch.cat([logit_pos, logit_neg], dim=1)

        # 5. Dual-Dimension Normalization
        A_row = F.softmax(logit, dim=-1)
        A_col = F.softmax(logit, dim=-2)
        A = torch.sqrt(A_row * A_col)

        # 6. Split the normalized weights back into positive and negative tracks
        A_pos, A_neg = torch.split(A, [N_pos, N_neg], dim=1)

        # 7. Compute Cross-Mass Weights
        W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
        W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

        # 8. Compute the Expected Positions (Equation 8 / Mean-Shift Aggregation)
        drift_pos = torch.matmul(W_pos, y_pos)
        drift_neg = torch.matmul(W_neg, y_neg) 

        # 9. Net Vector Field (Equation 10: V = V+ - V-)
        V_tau = drift_pos - drift_neg
        return V_tau

    def forward(self, x, y_pos):
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            D = C * H * W
            x_flat = x.view(B, D)
            y_pos_flat = y_pos.view(B, D)
        else:
            B, D = x.shape 
            x_flat = x
            y_pos_flat = y_pos
            
        y_neg_flat = x_flat 

        # FEATURE NORMALIZATION (Section A.6, Equation 21 strict correction)
        with torch.no_grad():
            # Calculate distances strictly cross-wise between generated (x) and real (y) pairs
            cross_dist = torch.cdist(x_flat, y_pos_flat, p=2)
            mean_cross_dist = cross_dist.mean()
            
            # Equation 21: Normalization Scale S_j
            S_j = mean_cross_dist / math.sqrt(D)
            S_j = torch.clamp(S_j, min=1e-6).detach()

        # Normalize features by scale S_j (Equation 18)
        x_norm = x_flat / S_j
        y_pos_norm = y_pos_flat / S_j
        y_neg_norm = y_neg_flat / S_j

        # MULTI-TEMPERATURE AGGREGATION & DRIFT NORMALIZATION (Section A.6)
        V_aggregated = torch.zeros_like(x_norm)
        
        for tau_base in self.temperatures:
            # Equation 22 adjustment: tau_tilde = tau * sqrt(C_j)
            tau_tilde = tau_base * math.sqrt(D)
            
            # Compute V for this specific temperature threshold
            V_tau = self.compute_V_at_temperature(x_norm, y_pos_norm, y_neg_norm, tau=tau_tilde)
            
            # Equation 25: Compute Drift Normalization Scale lambda_j
            with torch.no_grad():
                lambda_j = torch.sqrt((V_tau ** 2).sum(dim=-1).mean() / D).detach()
                lambda_j = torch.clamp(lambda_j, min=1e-6)
                
            # Equation 23: Accumulate normalized field components
            V_aggregated += (V_tau / lambda_j)

        # LOSS COMPUTATION (Equation 26 & Algorithm 1)
        x_drifted = (x_norm + V_aggregated).detach()
        loss = F.mse_loss(x_norm, x_drifted)
        
        return loss