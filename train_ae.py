import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.dataset import get_mnist_loaders
from src.autoencoder import DigitVAE
import os

def train_ae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_mnist_loaders(batch_size=256)
    
    # Initialize the correct variational architecture matching your autoencoder.py
    model = DigitVAE(latent_dim=16).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Hyperparameter to balance reconstruction fidelity against latent space smoothness
    beta = 0.01  
    
    print("[*] Pre-training Variational Autoencoder feature bottleneck (10 Epochs)...")
    model.train()
    for epoch in range(1, 11):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        
        for x, _ in train_loader:
            x = x.to(device)
            
            # Forward pass through the VAE pipeline
            reconstruction, mu, logvar = model(x)
            
            # 1. Reconstruction Loss (Fidelity constraint)
            recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
            
            # 2. KL Divergence Loss (Distribution smoothing constraint)
            # Analytical solution for closed-form KL matching a unit Gaussian N(0, I)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / x.size(0)  # Normalize across the batch size
            
            # Combined Loss Function
            loss = recon_loss + beta * kl_loss
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track separate metrics for granular logging
            total_loss += loss.item() * x.size(0)
            total_recon += recon_loss.item() * x.size(0)
            total_kl += kl_loss.item() * x.size(0)
            
        dataset_size = len(train_loader.dataset)
        print(f"    Epoch {epoch:02d}/10 | "
              f"Total Loss: {total_loss / dataset_size:.5f} | "
              f"MSE Recon: {total_recon / dataset_size:.5f} | "
              f"KL Div: {total_kl / dataset_size:.5f}")
        
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "./checkpoints/autoencoder.pt")
    print("[+] Continuous variational latent bottleneck weights saved successfully.")

if __name__ == "__main__":
    train_ae()