import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.dataset import get_mnist_loaders
from src.autoencoder import DigitVAE
import os
from src.utils import save_image_grid

def train_ae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_mnist_loaders(batch_size=128)
    
    model = DigitVAE(latent_dim=16).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    target_beta = 0.01
    total_epochs = 15

    print(f"[*] Pre-training Variational Autoencoder with KL annealing ({total_epochs} epochs)...")
    for epoch in range(1, total_epochs + 1):
        beta = min(target_beta, target_beta * (epoch / 8.0))
        
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        
        model.train()
        progress = tqdm(train_loader, desc=f"AE Epoch {epoch}/{total_epochs}")
        for x, _ in progress:
            x = x.to(device)
            
            reconstruction, mu, logvar = model(x)
            
            recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
            
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / x.size(0)
            
            loss = recon_loss + beta * kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_recon += recon_loss.item() * x.size(0)
            total_kl += kl_loss.item() * x.size(0)
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
        dataset_size = len(train_loader.dataset)
        print(f"    Epoch {epoch:02d}/{total_epochs} | "
              f"Beta: {beta:.4f} | "
              f"Total Loss: {total_loss / dataset_size:.5f} | "
              f"MSE Recon: {total_recon / dataset_size:.5f} | "
              f"KL Div: {total_kl / dataset_size:.5f}")
        
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "./checkpoints/autoencoder.pt")
    print("[+] Continuous variational latent bottleneck weights saved successfully.")

    model.eval()
    with torch.no_grad():
        samples, _ = next(iter(train_loader))
        samples = samples[:64].to(device)
        reconstructions, _, _ = model(samples)
        save_image_grid(reconstructions, "./outputs/autoencoder_reconstructions.png", nrow=8)

if __name__ == "__main__":
    train_ae()