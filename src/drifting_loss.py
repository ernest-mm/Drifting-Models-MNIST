import torch
import torch.nn as nn
import math

class DriftingLoss(nn.Module):
    def __init__(self, temperatures=(0.05, 0.1, 0.2)):
        super().__init__()
        self.temperatures = tuple(temperatures)

    def compute_field(self, x, targets, tau):
        dist = torch.cdist(x, targets, p=2)
        weights = torch.softmax(-dist / tau, dim=-1)
        attraction = torch.matmul(weights, targets)
        return attraction - x

    def forward(self, x, y_pos):
        B = x.size(0)
        x_flat = x.view(B, -1)
        y_pos_flat = y_pos.view(B, -1)

        if y_pos_flat.size(0) != B:
            raise ValueError("x and y_pos must have the same batch size")

        if B > 1:
            y_neg_flat = torch.roll(y_pos_flat, shifts=1, dims=0)
        else:
            y_neg_flat = y_pos_flat

        D = x_flat.size(-1)

        with torch.no_grad():
            cross_dist = torch.cdist(x_flat, y_pos_flat, p=2)
            S_j = cross_dist.mean() / math.sqrt(D)
            S_j = torch.clamp(S_j, min=1e-6).detach()

        x_norm = x_flat / S_j
        y_pos_norm = y_pos_flat / S_j
        y_neg_norm = y_neg_flat / S_j

        V_aggregated = torch.zeros_like(x_norm)

        for tau_base in self.temperatures:
            tau_tilde = max(tau_base * math.sqrt(D), 1e-6)
            V_pos = self.compute_field(x_norm, y_pos_norm, tau_tilde)
            V_neg = self.compute_field(x_norm, y_neg_norm, tau_tilde)
            V_aggregated += V_pos - 0.5 * V_neg

        V_final = V_aggregated / len(self.temperatures)
        return V_final, S_j