import argparse
import os
import torch

from src.autoencoder import DigitAutoencoder
from src.models import LatentDiT
from src.utils import save_image_grid

def main():
    parser = argparse.ArgumentParser(description="One-Step (1-NFE) Latent Conditional Digit Generation")
    parser.add_argument("--digit", type=int, default=7, choices=list(range(10)), help="The specific digit (0-9) you want to generate")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-Free Guidance scale (higher = sharper/more strict)")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--output_path", type=str, default="./outputs/generated_digits.png", help="Path to save the generated image grid")
    args = parser.parse_args()

    # 1. Setup hardware acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Inference running on device: {device}")

    # 2. Load the structural decoder mesh
    ae = DigitAutoencoder(latent_dim=16).to(device)
    ae.load_state_dict(torch.load("checkpoints/autoencoder.pt", map_location=device))
    ae.eval()

    # 3. Load the trained Latent DiT vector field
    model = LatentDiT(latent_dim=16).to(device)
    model.load_state_dict(torch.load("checkpoints/drifting_generator_v2.pt", map_location=device))
    model.eval()

    # 4. Prepare conditioning tensors for generation
    labels = torch.full((args.num_samples,), args.digit, dtype=torch.long, device=device)
    alphas = torch.full((args.num_samples, 1), args.cfg_scale, dtype=torch.float, device=device)

    # 5. Generate with 1-NFE (One Neural Function Evaluation)
    print(f"[*] Generating {args.num_samples} samples of digit '{args.digit}' with CFG scale {args.cfg_scale}...")
    
    with torch.no_grad():
        # Step 1: Draw random latent variables epsilon from flat standard Gaussian distribution [B, 16]
        epsilon = torch.randn(args.num_samples, 16, device=device)
        
        # Step 2: Map flat noise vectors straight to target latent features
        latent_out = model(epsilon, labels, alphas)
        
        # Step 3: Decode latent coordinates back to raw pixels
        generated_pixels = ae.decode(latent_out)

    # 6. Serialize and save the output matrix grid to disk
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    nrow = int(args.num_samples ** 0.5) if int(args.num_samples ** 0.5) > 0 else 1
    save_image_grid(generated_pixels, args.output_path, nrow=nrow)
    print(f"[+] Success! Generated image grid exported to: {args.output_path}")

if __name__ == "__main__":
    main()