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

        # 2. Ignore Self-Interaction out-of-place (Prevents Autograd errors)
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

        # 8. Compute the Expected Positions (Mean-Shift Aggregation)
        drift_pos = torch.matmul(W_pos, y_pos)
        drift_neg = torch.matmul(W_neg, y_neg) 

        # 9. Net Vector Field (Equation 10: V = V+ - V-)
        V_tau = drift_pos - drift_neg
        return V_tau

    def forward(self, x, y_pos):
        """
        Calculates and returns the aggregated structural drift field vectors V(x)
        instead of computing an internal scalar loss metric.
        """
        # Ensure input representations are appropriately flattened
        B = x.size(0)
        x_flat = x.view(B, -1)
        y_pos_flat = y_pos.view(B, -1)
        y_neg_flat = x_flat 
        
        D = x_flat.size(-1)

        # FEATURE NORMALIZATION 
        with torch.no_grad():
            # Calculate distances strictly cross-wise between generated (x) and real (y) pairs
            cross_dist = torch.cdist(x_flat, y_pos_flat, p=2)
            mean_cross_dist = cross_dist.mean()
            
            # Normalization Scale S_j
            S_j = mean_cross_dist / math.sqrt(D)
            S_j = torch.clamp(S_j, min=1e-6).detach()

        # Normalize features by scale S_j
        x_norm = x_flat / S_j
        y_pos_norm = y_pos_flat / S_j
        y_neg_norm = y_neg_flat / S_j

        # MULTI-TEMPERATURE AGGREGATION & DRIFT NORMALIZATION
        V_aggregated = torch.zeros_like(x_norm)
        
        for tau_base in self.temperatures:
            tau_tilde = tau_base * math.sqrt(D)
            
            # Compute V for this specific temperature threshold
            V_tau = self.compute_V_at_temperature(x_norm, y_pos_norm, y_neg_norm, tau=tau_tilde)
            
            # Compute Drift Normalization Scale lambda_j
            with torch.no_grad():
                lambda_j = torch.sqrt((V_tau ** 2).sum(dim=-1).mean() / D).detach()
                lambda_j = torch.clamp(lambda_j, min=1e-6)
                
            # Accumulate normalized field components
            V_aggregated += (V_tau / lambda_j)

        V_final = V_aggregated / len(self.temperatures)
        return V_final