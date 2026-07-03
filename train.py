import argparse
import os
import torch
import torch.optim as optim
from tqdm import tqdm

from src.dataset import get_mnist_loaders
from src.autoencoder import DigitAutoencoder
from src.models import LatentDiT
from src.drifting_loss import DriftingLoss
from src.utils import save_image_grid

def main():
    parser = argparse.ArgumentParser(description="Train a Latent-Space Drifting Model on MNIST")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Total number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Utilizing computation device: {device}")
    
    # Ensure local directory paths are fully initialized
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    train_loader, _ = get_mnist_loaders(batch_size=args.batch_size)

    # 1. Load your frozen feature space encoder
    ae = DigitAutoencoder(latent_dim=16).to(device)
    ae.load_state_dict(torch.load("./checkpoints/autoencoder.pt", map_location=device))
    ae.eval()

    # 2. Setup Vector DiT and Drift Loss Engine
    model = LatentDiT(latent_dim=16).to(device)
    criterion = DriftingLoss(temperatures=[0.02, 0.05, 0.2])
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("[*] Optimizing Drifting Fields in structural latent space...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        
        # Wrapped loop tracker for clean visibility
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            B = images.size(0)
            
            # Compress real images to target feature destinations
            with torch.no_grad():
                y_pos = ae.encode(images) # [B, 16]
                
            alpha = torch.rand(B, 1, device=device) * 3.0 + 1.0
            drop_mask = torch.rand(B, device=device) < 0.1
            conditioned_labels = torch.where(drop_mask, torch.tensor(10, device=device), labels)
            
            # Drift from random latent state coordinates
            epsilon = torch.randn_like(y_pos)
            z_predicted = model(epsilon, conditioned_labels, alpha)
            
            loss = criterion(z_predicted, y_pos)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Protect multihead self-attention paths from explosive gradient spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.5f}"})

        print(f"[=>] Epoch {epoch} Completed. Average Vector Loss: {running_loss/len(train_loader):.5f}")

        # Save out a generation grid to visually confirm sharp digit structures
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                sample_labels = torch.arange(10, device=device).repeat(8)[:64]
                fixed_alpha = torch.ones(64, 1, device=device) * 2.5
                fixed_noise = torch.randn(64, 16, device=device)
                
                # Single pass prediction + decode straight to canvas pixels
                latent_out = model(fixed_noise, sample_labels, fixed_alpha)
                pixel_out = ae.decode(latent_out)
                
                save_image_grid(pixel_out, f"./outputs/epoch_{epoch}.png", nrow=10)

    torch.save(model.state_dict(), "./checkpoints/drifting_generator_v2.pt")
    print("[+] Training fully updated and verified!")

if __name__ == "__main__":
    main()