import argparse
import os
import torch
import torch.optim as optim
from tqdm import tqdm

from src.dataset import get_mnist_loaders
from src.autoencoder import DigitVAE  # Updated from DigitAutoencoder
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
    ae = DigitVAE(latent_dim=16).to(device)  # Updated from DigitAutoencoder
    ae.load_state_dict(torch.load("./checkpoints/autoencoder.pt", map_location=device))
    ae.eval() # Crucial: freeze batchnorm/dropout profiles in the encoder mesh

    # 2. Instantiate core generation network elements
    model = LatentDiT(latent_dim=16).to(device)
    criterion = DriftingLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("[*] Initiating Latent Drifting Model training phase...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, conditioned_labels in pbar:
            images = images.to(device)
            conditioned_labels = conditioned_labels.to(device)
            B = images.size(0)
            
            # Extract target anchor features cleanly using the VAE's mu projection mapping
            with torch.no_grad():
                y_pos = ae.encode(images)[0]  # Fixed: Extracted mu [0] to resolve the tuple view error
            
            # Map clean conditioning vector parameters matching Section 4 conventions
            # Alpha controls the classifier-free guidance scaling thresholds implicitly
            alpha = torch.randn(B, 1, device=device)
            epsilon = torch.randn(B, 16, device=device)
            
            # Predict downstream vector maps from standard Gaussian configurations
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
    print("[+] Training complete. Latent DiT weights saved.")

if __name__ == "__main__":
    main()