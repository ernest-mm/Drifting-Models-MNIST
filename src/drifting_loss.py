import torch
import torch.nn as nn
import torch.nn.functional as F

class DriftingLoss(nn.Module):
    """
    Implements the generative Drifting Model loss framework according to 
    Section 3.3 (Equations 11 & 12 / Algorithm 2) and Section A.6 (Normalization).
    
    This module computes the anti-symmetric velocity drift field V based on 
    competitive attraction-repulsion forces between a batch of generated samples (x),
    positive real data samples (y_pos), and negative generated samples (y_neg).
    """
    def __init__(self, temperatures=[0.02, 0.05, 0.2]):
        super().__init__()
        # Section A.6 ("Multiple temperatures"): Using a list of temperatures 
        # improves robustness across features without hyperparameter tuning.
        self.temperatures = temperatures

    def compute_V_at_temperature(self, x, y_pos, y_neg, tau):
        """
        Computes the drifting field V for a single temperature tau according to 
        Algorithm 2 in Appendix A.2.
        
        Args:
            x: Current generated samples tensor, shape [B, D]
            y_pos: Positive real data samples tensor, shape [N_pos, D]
            y_neg: Negative generated samples tensor, shape [N_neg, D] (usually reuses x)
            tau: Temperature scalar hyperparameter
            
        Returns:
            V_tau: The drifting field velocity vectors for this temperature, shape [B, D]
        """
        N = x.size(0)
        N_pos = y_pos.size(0)
        N_neg = y_neg.size(0)

        # 1. Compute Pairwise Euclidean (L2) Distance Matrices (Equation 12 / Algorithm 2)
        # dist_pos[i, j] = ||x_i - y_pos_j||
        dist_pos = torch.cdist(x, y_pos, p=2) # Shape: [N, N_pos]
        dist_neg = torch.cdist(x, y_neg, p=2) # Shape: [N, N_neg]

        # 2. Ignore Self-Interaction if y_neg reuses the generated points x (Algorithm 2)
        # Prevents a generated sample from repelling itself with infinite force.
        # CHANGE THIS LINE (Line 45 in src/drifting_loss.py):
        if torch.tensor([y_neg.data_ptr() == x.data_ptr()], dtype=torch.bool, device=x.device).item():
            dist_neg = dist_neg + torch.eye(N, device=x.device) * 1e6

        # 3. Scale by temperature and convert distances to similarity logits
        logit_pos = -dist_pos / tau # Shape: [N, N_pos]
        logit_neg = -dist_neg / tau # Shape: [N, N_neg]

        # 4. Concatenate for Joint Normalization (Algorithm 2: logit = cat([logit_pos, logit_neg], dim=1))
        logit = torch.cat([logit_pos, logit_neg], dim=1) # Shape: [N, N_pos + N_neg]

        # 5. Dual-Dimension Normalization (Paragraph 5 & Algorithm 2)
        # "We implement k_tilde using a softmax operation... normalized along both dimensions"
        A_row = F.softmax(logit, dim=-1)   # Normalize over targets (y axis) -> Equation 9 / Z normalization
        A_col = F.softmax(logit, dim=-2)   # Normalize over sources (x axis) -> Extra stabilization
        A = torch.sqrt(A_row * A_col)      # Jointly geometric mean matrix, Shape: [N, N_pos + N_neg]

        # 6. Split the normalized weights back into positive and negative tracks
        A_pos, A_neg = torch.split(A, [N_pos, N_neg], dim=1) # Shapes: [N, N_pos] and [N, N_neg]

        # 7. Compute Cross-Mass Weights (Algorithm 2)
        # W_pos balances the row-wise influence of negatives onto positives and vice-versa
        W_pos = A_pos * A_neg.sum(dim=1, keepdim=True) # Shape: [N, N_pos]
        W_neg = A_neg * A_pos.sum(dim=1, keepdim=True) # Shape: [N, N_neg]

        # 8. Compute the Expected Positions (Equation 8 / Mean-Shift Aggregation)
        drift_pos = torch.matmul(W_pos, y_pos) # Weighted center of attraction, Shape: [N, D]
        drift_neg = torch.matmul(W_neg, y_neg) # Weighted center of repulsion, Shape: [N, D]

        # 9. Net Vector Field (Equation 10: V = V+ - V-)
        V_tau = drift_pos - drift_neg
        return V_tau

    def forward(self, x, y_pos):
        """
        Executes the overall multi-temperature, normalized drifting loss step.
        
        Args:
            x: Generated images from network f_theta, shape [B, 1, 28, 28]
            y_pos: Authentic target images from DataLoader, shape [B, 1, 28, 28]
            
        Returns:
            loss: Scalar Mean Squared Error drifting loss tensor (Equation 26)
        """
        # Save original spatial dimension to rebuild the shape later if needed
        B, C, H, W = x.shape
        D = C * H * W
        
        # Flatten images to vectors for distance metrics in pixel space
        x_flat = x.view(B, D)
        y_pos_flat = y_pos.view(B, D)
        
        # Algorithm 1: "The generated samples also serve as the negative samples in the same batch"
        y_neg_flat = x_flat 

        # --- FEATURE NORMALIZATION (Section A.6, Equations 18-22) ---
        # "We want to perform normalization such that the kernel k(·,·) and drift V 
        # are insensitive to the absolute magnitude of features."
        # For pixel space, dimensionality C_j is D. Target average distance is sqrt(D).
        with torch.no_grad():
            # Compute raw average pairwise distance across all available batch samples
            all_samples = torch.cat([x_flat, y_pos_flat], dim=0)
            raw_dist = torch.cdist(all_samples, all_samples, p=2)
            mean_raw_dist = raw_dist.mean()
            
            # Equation 21: Normalization Scale S_j
            S_j = mean_raw_dist / math.sqrt(D) if 'math' in globals() else mean_raw_dist / (D ** 0.5)
            # Apply stop-gradient to S_j since it acts as a constant scaling factor for the batch
            S_j = S_j.detach()

        # Normalize features by scale S_j (Equation 18)
        x_norm = x_flat / S_j
        y_pos_norm = y_pos_flat / S_j
        y_neg_norm = y_neg_flat / S_j

        # --- MULTI-TEMPERATURE AGGREGATION & DRIFT NORMALIZATION (Section A.6) ---
        V_aggregated = torch.zeros_like(x_norm)
        
        for tau_base in self.temperatures:
            # Equation 22 adjustment: tau_tilde = tau * sqrt(C_j)
            tau_tilde = tau_base * (D ** 0.5)
            
            # Compute V for this specific temperature threshold
            V_tau = self.compute_V_at_temperature(x_norm, y_pos_norm, y_neg_norm, tau=tau_tilde)
            
            # Equation 25: Compute Drift Normalization Scale lambda_j for this specific V_tau
            with torch.no_grad():
                # lambda_j = sqrt( E[ (1/C_j) * ||V_j||^2 ] )
                lambda_j = torch.sqrt((V_tau ** 2).sum(dim=-1).mean() / D).detach()
                # Safeguard against division by zero
                lambda_j = torch.clamp(lambda_j, min=1e-6)
                
            # Equation 23: V_tilde = V / lambda_j. Accumulate into total aggregated field.
            V_aggregated += (V_tau / lambda_j)

        # --- LOSS COMPUTATION (Equation 26 & Algorithm 1) ---
        # L_j = MSE( x_tilde - sg(x_tilde + V_tilde) )
        # Using stop-gradient (sg / .detach()) handles the distribution mechanics correctly.
        x_drifted = (x_norm + V_aggregated).detach()
        loss = F.mse_loss(x_norm, x_drifted)
        
        return loss