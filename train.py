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

def main():
    parser = argparse.ArgumentParser(description="Train a Latent-Space Drifting Model on MNIST")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Total number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    parser.add_argument("--drift_step", type=float, default=1.0, help="Step size along the drift field vector")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Utilizing computation device: {device}")

    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    train_loader, _ = get_mnist_loaders(batch_size=args.batch_size)

    ae_path = "./checkpoints/autoencoder.pt"
    if not os.path.exists(ae_path):
        raise FileNotFoundError("Autoencoder checkpoint not found. Run train_ae.py first.")

    ae = DigitVAE(latent_dim=16).to(device)
    ae.load_state_dict(load_state_dict_any(ae_path, map_location=device))
    ae.eval()

    model = LatentDiT(latent_dim=16).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
            
            with torch.inference_mode():
                y_pos = ae.encode(images)[0]
            
            strength = torch.rand(B, 1, device=device)
            epsilon = torch.randn(B, 16, device=device)
            target = y_pos + 0.1 * torch.randn_like(y_pos)
            
            z_predicted = model(epsilon, conditioned_labels, strength)

            recon_loss = F.mse_loss(z_predicted, target)
            latent_penalty = 0.01 * z_predicted.pow(2).mean()
            loss = recon_loss + latent_penalty
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_value = loss.item()
            running_loss += loss_value
            iteration_losses.append(loss_value)
            
            pbar.set_postfix({"latent_mse": f"{recon_loss.item():.5f}"})

        print(f"[=>] Epoch {epoch} Completed. Average Pushforward MSE: {running_loss/len(train_loader):.5f}")

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

    loss_curve_path = "./outputs/loss_curve.csv"
    with open(loss_curve_path, "w", encoding="utf-8") as handle:
        handle.write("iteration,loss\n")
        for index, value in enumerate(iteration_losses, start=1):
            handle.write(f"{index},{value:.8f}\n")
    print(f"[+] Performance log finalized at: {loss_curve_path}")

if __name__ == "__main__":
    main()