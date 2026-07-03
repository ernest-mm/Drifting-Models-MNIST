import argparse
import os
import torch
import torch.optim as optim
from tqdm import tqdm

# Import our modular components from the src package
from src.dataset import get_mnist_loaders
from src.models import MiniatureDiT
from src.drifting_loss import DriftingLoss
from src.utils import save_image_grid, CheckpointManager

def main():
    parser = argparse.ArgumentParser(description="Train a Pixel-Space Drifting Model on MNIST Digits")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for AdamW optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    parser.add_argument("--sample_every", type=int, default=5, help="Epoch interval to save generation samples")
    parser.add_argument("--checkpoint_dir", type=type(""), default="./checkpoints", help="Directory to save weights")
    args = parser.parse_args()

    # 1. Setup hardware acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Utilizing computation device: {device}")

    # 2. Load the digit data loaders (Section 4 & Section A.9)
    train_loader, _ = get_mnist_loaders(batch_size=args.batch_size)

    # 3. Instantiate the Miniature Diffusion Transformer (f_theta)
    # 28x28 grayscale images with a patch size of 2 -> 14x14 = 196 tokens
    model = MiniatureDiT(
        img_size=28,
        in_channels=1,
        patch_size=2,
        embed_dim=128,
        num_heads=4,
        depth=4
    ).to(device)

    # 4. Setup the mathematical Drifting Loss engine (Section 3.3 & Appendix A.6)
    criterion = DriftingLoss(temperatures=[0.02, 0.05, 0.2])

    # 5. Configure standard professional optimizer settings
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_manager = CheckpointManager(checkpoint_dir=args.checkpoint_dir)
    
    # Optional: Resume training if a historical checkpoint is present
    start_epoch = checkpoint_manager.load(model, optimizer, device=device)

    print("[*] Starting Drifting Model optimization loop...")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        
        # Progress bar wrapper for tracking matrix performance per iteration
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for y_pos, labels in pbar:
            # y_pos represents the real positive target data batch
            y_pos = y_pos.to(device)
            labels = labels.to(device)
            B = y_pos.size(0)

            # --- CLASSIFIER-FREE GUIDANCE TRAINING SETUP (Section 4 & Section A.7) ---
            # Randomly sample CFG scale alpha value for each element in the batch.
            # Following standard practice, we sample uniformly between 1.0 and 4.0.
            alpha = torch.rand(B, 1, device=device) * 3.0 + 1.0
            
            # Unconditional conditioning drop track (10% probability)
            # Index 10 acts as our learned "null" token class placeholder mapping
            drop_mask = torch.rand(B, device=device) < 0.1
            conditioned_labels = torch.where(drop_mask, torch.tensor(10, device=device), labels)

            # --- ALGORITHM 1 STEP ACTIONS ---
            # Step 1: Draw N samples of continuous standard Gaussian noise epsilon
            epsilon = torch.randn_like(y_pos)

            # Step 2: Feed noise and conditioning parameters to the generator f_theta
            x = model(epsilon, conditioned_labels, alpha)

            # Step 3 & 4: Compute the normalized vector field V and execute backward pass
            # x acts simultaneously as the generated positions and the negative repelling set
            loss = criterion(x, y_pos)

            # Standard gradient update steps
            optimizer.zero_grad()
            loss.backward()
            # Apply gradient clipping to ensure the self-attention heads remain perfectly stable
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        epoch_loss = running_loss / len(train_loader)
        print(f"[=>] Epoch {epoch} Completed. Average Vector Loss: {epoch_loss:.6f}")

        # Periodically dump visualization sheets to visually monitor generation morphing
        if epoch % args.sample_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                # Generate tracking sample grid using fixed testing parameters
                # Let's request an ordered batch of classes 0-9 to check conditioning
                sample_labels = torch.arange(10, device=device).repeat(8)[:64] # 64 elements
                fixed_alpha = torch.ones(64, 1, device=device) * 2.0 # Standard guidance strength
                fixed_noise = torch.randn(64, 1, 28, 28, device=device)
                
                # Single-Forward-Pass generation evaluation (1-NFE inference execution)
                sampled_digits = model(fixed_noise, sample_labels, fixed_alpha)
                
                # Serialize grid array to disk
                sample_path = f"./outputs/epoch_{epoch}.png"
                save_image_grid(sampled_digits, sample_path, nrow=10)
                print(f"[*] Saved training visualization grid sheet to: {sample_path}")

            # Persist latest network parameters state to disk checkpoints directory
            checkpoint_manager.save(model, optimizer, epoch, epoch_loss)

    print("[*] Model training completely optimized.")

if __name__ == "__main__":
    main()