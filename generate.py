import argparse
import os
import torch

from src.autoencoder import DigitVAE
from src.models import LatentDiT
from src.utils import load_state_dict_any, save_image_grid

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

    ae_path = "checkpoints/autoencoder.pt"
    model_path = "checkpoints/drifting_generator_v2.pt"
    if not os.path.exists(ae_path):
        raise FileNotFoundError("Autoencoder checkpoint not found. Run train_ae.py first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Generator checkpoint not found. Run train.py first.")

    ae = DigitVAE(latent_dim=16).to(device)
    ae.load_state_dict(load_state_dict_any(ae_path, map_location=device))
    ae.eval()

    model = LatentDiT(latent_dim=16).to(device)
    model.load_state_dict(load_state_dict_any(model_path, map_location=device))
    model.eval()

    labels = torch.full((args.num_samples,), args.digit, dtype=torch.long, device=device)
    alphas = torch.full((args.num_samples, 1), args.cfg_scale, dtype=torch.float, device=device)

    print(f"[*] Generating {args.num_samples} samples of digit '{args.digit}' with CFG scale {args.cfg_scale}...")
    
    with torch.inference_mode():
        epsilon = torch.randn(args.num_samples, 16, device=device)
        latent_out = model(epsilon, labels, alphas)
        generated_pixels = ae.decode(latent_out)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    nrow = int(args.num_samples ** 0.5) if args.num_samples >= 4 else args.num_samples
    save_image_grid(generated_pixels, args.output_path, nrow=nrow)
    print(f"[+] Output image successfully saved to {args.output_path}")

if __name__ == "__main__":
    main()