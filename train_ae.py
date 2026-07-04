import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.dataset import get_mnist_loaders
from src.autoencoder import DigitVAE
import os

def train_ae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Using a slightly lower batch size for more frequent gradient updates
    train_loader, _ = get_mnist_loaders(batch_size=128)
    
    model = DigitVAE(latent_dim=16).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Target weight for the KL Divergence term
    target_beta = 0.01  
    total_epochs = 15
    
    print(f"[*] Pre-training Variational Autoencoder with KL-Annealing ({total_epochs} Epochs)...")
    model.train()
    for epoch in range(1, total_epochs + 1):
        # Linear KL Warmup: Gradually scale beta from 0 to target_beta over the first 8 epochs
        # This prevents early posterior collapse and yields distinct continuous manifolds
        beta = min(target_beta, target_beta * (epoch / 8.0))
        
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        
        for x, _ in train_loader:
            x = x.to(device)
            
            reconstruction, mu, logvar = model(x)
            
            # 1. Reconstruction Loss
            recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
            
            # 2. KL Divergence Loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / x.size(0) 
            
            # Combined Objective
            loss = recon_loss + beta * kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_recon += recon_loss.item() * x.size(0)
            total_kl += kl_loss.item() * x.size(0)
            
        dataset_size = len(train_loader.dataset)
        print(f"    Epoch {epoch:02d}/{total_epochs} | "
              f"Beta: {beta:.4f} | "
              f"Total Loss: {total_loss / dataset_size:.5f} | "
              f"MSE Recon: {total_recon / dataset_size:.5f} | "
              f"KL Div: {total_kl / dataset_size:.5f}")
        
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "./checkpoints/autoencoder.pt")
    print("[+] Continuous variational latent bottleneck weights saved successfully.")

if __name__ == "__main__":
    train_ae()