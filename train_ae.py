import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_mnist_loaders
from src.autoencoder import DigitAutoencoder
import os

def train_ae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_mnist_loaders(batch_size=256)
    
    model = DigitAutoencoder(latent_dim=16).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("[*] Pre-training Autoencoder feature bottleneck (10 Epochs)...")
    model.train()
    for epoch in range(1, 11):
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            recon = model(x)
            loss = criterion(recon, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            
        print(f"    Epoch {epoch}/10 | MSE Reconstruction Loss: {total_loss / len(train_loader.dataset):.5f}")
        
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "./checkpoints/autoencoder.pt")
    print("[+] Autoencoder compressed weight space saved successfully.")

if __name__ == "__main__":
    train_ae()