import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.dataset import get_mnist_loaders
from src.autoencoder import DigitVAE
from src.models import LatentDiT
from src.utils import load_state_dict_any, save_image_grid
from src.drifting_loss import DriftingLoss

def main():
    parser = argparse.ArgumentParser(description="Train a Latent-Space Drifting Model on MNIST")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Total number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Utilizing computation device: {device}")

    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    train_loader, _ = get_mnist_loaders(batch_size=args.batch_size)

    # 1. Load Frozen Autoencoder Encoder/Decoder
    ae_path = "./checkpoints/autoencoder.pt"
    if not os.path.exists(ae_path):
        raise FileNotFoundError("Autoencoder checkpoint not found. Run train_ae.py first.")

    ae = DigitVAE(latent_dim=16).to(device)
    ae.load_state_dict(load_state_dict_any(ae_path, map_location=device))
    ae.eval()
    for param in ae.parameters():
        param.requires_grad = False

    # 2. Generator & Loss Setup
    model = LatentDiT(latent_dim=16).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    drifting_criterion = DriftingLoss(temperatures=(0.05, 0.1, 0.2))

    iteration_losses = []

    print("[*] Initiating latent drifting model training phase...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, conditioned_labels in pbar:
            images = images.to(device)
            conditioned_labels = conditioned_labels.to(device)
            B = images.size(0)
            
            # Encode real images into latent space
            with torch.no_grad():
                y_pos, _ = ae.encode(images)
                y_pos = y_pos.detach()

            strength = torch.ones(B, 1, device=device)
            epsilon = torch.randn(B, 16, device=device)
            
            # Predict generated latents
            z_predicted = model(epsilon, conditioned_labels, strength)

            # Compute Drifting Field V on detached latents
            with torch.no_grad():
                V = drifting_criterion(z_predicted.detach(), y_pos)

            # Construct Stop-Gradient Target (Paper Eq. 5 & 6)
            target = (z_predicted.detach() + V).detach()

            # Optimize Generator towards Target
            loss = F.mse_loss(z_predicted, target)

            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_value = loss.item()
            running_loss += loss_value
            iteration_losses.append(loss_value)
            
            pbar.set_postfix({"drift_loss": f"{loss_value:.6f}"})

        print(f"[=>] Epoch {epoch} Completed. Average Drift Loss: {running_loss/len(train_loader):.6f}")

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.inference_mode():
                sample_labels = torch.arange(10, device=device).repeat(8)[:64]
                fixed_alpha = torch.ones(64, 1, device=device)
                fixed_noise = torch.randn(64, 16, device=device)
                
                latent_out = model.sample(fixed_noise, sample_labels, fixed_alpha)
                pixel_out = ae.decode(latent_out)
                
                save_image_grid(pixel_out, f"./outputs/epoch_{epoch}.png", nrow=10)

    torch.save(model.state_dict(), "./checkpoints/drifting_generator_v2.pt")
    print("[+] Training complete. Latent DiT weights saved.")

if __name__ == "__main__":
    main()