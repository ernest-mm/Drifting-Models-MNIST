import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.dataset import get_mnist_loaders
from src.autoencoder import DigitVAE  
from src.models import LatentDiT
from src.drifting_loss import DriftingLoss
from src.utils import save_image_grid

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

    ae = DigitVAE(latent_dim=16).to(device)  
    ae.load_state_dict(torch.load("./checkpoints/autoencoder.pt", map_location=device))
    ae.eval() 

    model = LatentDiT(latent_dim=16).to(device)
    get_field = DriftingLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    iteration_losses = []

    print("[*] Initiating Latent Drifting Model training phase...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, conditioned_labels in pbar:
            images = images.to(device)
            conditioned_labels = conditioned_labels.to(device)
            B = images.size(0)
            
            with torch.no_grad():
                y_pos = ae.encode(images)[0]  
            
            alpha = torch.randn(B, 1, device=device)
            epsilon = torch.randn(B, 16, device=device)
            
            # 1. Predict downstream latent coordinates
            z_predicted = model(epsilon, conditioned_labels, alpha)
            
            # 2. Extract the raw structural field vectors and feature scale factor S_j
            # Note: Ensure your drifting_loss.py forward function returns (V_final, S_j)
            V, S_j = get_field(z_predicted, y_pos)
            
            # 3. Project to the mathematically required normalized space
            z_predicted_norm = z_predicted / S_j
            
            # 4. Create the detached pushforward target inside normalized space
            z_target_norm = (z_predicted_norm + args.drift_step * V).detach()
            
            # 5. Calculate the Mean Squared Error tracking matching Algorithm 1
            loss = F.mse_loss(z_predicted_norm, z_target_norm)
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_value = loss.item()
            running_loss += loss_value
            iteration_losses.append(loss_value)
            
            pbar.set_postfix({"Pushforward_MSE": f"{loss_value:.5f}"})

        print(f"[=>] Epoch {epoch} Completed. Average Pushforward MSE: {running_loss/len(train_loader):.5f}")

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                sample_labels = torch.arange(10, device=device).repeat(8)[:64]
                fixed_alpha = torch.ones(64, 1, device=device) * 2.5
                fixed_noise = torch.randn(64, 16, device=device)
                
                latent_out = model(fixed_noise, sample_labels, fixed_alpha)
                pixel_out = ae.decode(latent_out)
                
                save_image_grid(pixel_out, f"./outputs/epoch_{epoch}.png", nrow=10)

    torch.save(model.state_dict(), "./checkpoints/drifting_generator_v2.pt")
    print("[+] Training complete. Latent DiT weights saved.")

    # Generate and save the visual loss tracking chart
    plt.figure(figsize=(10, 5))
    plt.plot(iteration_losses, label="Pushforward MSE", color="royalblue", linewidth=1)
    plt.title("Pushforward Optimization Trajectory Per Iteration")
    plt.xlabel("Global Training Iteration Index")
    plt.ylabel("MSE Loss Magnitude")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    chart_output_path = "./outputs/loss_curve.png"
    plt.savefig(chart_output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[+] Performance charts finalized at: {chart_output_path}")

if __name__ == "__main__":
    main()