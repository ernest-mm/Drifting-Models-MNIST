import argparse
import os
import torch

# Import our modular components from the src package
from src.models import MiniatureDiT
from src.utils import save_image_grid, CheckpointManager

def main():
    parser = argparse.ArgumentParser(description="One-Step (1-NFE) Conditional Digit Generation")
    parser.add_argument("--digit", type=int, default=7, choices=list(range(10)), help="The specific digit (0-9) you want to generate")
    parser.add_argument("--cfg_scale", type=float, default=2.0, help="Classifier-Free Guidance scale (1.0 = no guidance, higher = sharper/more strict)")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory where trained model weights are stored")
    parser.add_argument("--output_path", type=str, default="./outputs/generated_digits.png", help="Path to save the generated image grid")
    args = parser.parse_args()

    # 1. Setup hardware acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Inference running on device: {device}")

    # 2. Instantiate the exact same model architecture structural blueprint
    model = MiniatureDiT(
        img_size=28,
        in_channels=1,
        patch_size=2,
        embed_dim=128,
        num_heads=4,
        depth=4
    ).to(device)

    # 3. Load your pre-trained model weights
    checkpoint_manager = CheckpointManager(checkpoint_dir=args.checkpoint_dir)
    # Passing None to optimizer since we are strictly doing inference/evaluation
    checkpoint_manager.load(model, optimizer=None, device=device)
    
    # Set model to evaluation mode (freezes dropout, adaLN tracking, etc.)
    model.eval()

    # 4. Prepare conditioning tensors for generation
    # Create matching batches for class labels and continuous alpha tracking scales
    labels = torch.full((args.num_samples,), args.digit, dtype=torch.long, device=device)
    alphas = torch.full((args.num_samples, 1), args.cfg_scale, dtype=torch.float, device=device)

    # 5. Generate with 1-NFE (One Neural Function Evaluation)
    print(f"[*] Generating {args.num_samples} samples of digit '{args.digit}' with CFG scale {args.cfg_scale}...")
    
    with torch.no_grad():
        # Step 1: Draw random variables epsilon from standard Gaussian distribution
        epsilon = torch.randn(args.num_samples, 1, 28, 28, device=device)
        
        # Step 2: Perform the one-step pushforward mapping operation (Section 6)
        # Note: At inference time, our CFG formulation naturally handles the requested alpha scale
        generated_pixels = model(epsilon, labels, alphas)

    # 6. Serialize and save the output matrix grid to disk
    # Compute row count automatically to make a clean layout block
    nrow = int(args.num_samples ** 0.5) if int(args.num_samples ** 0.5) > 0 else 1
    save_image_grid(generated_pixels, args.output_path, nrow=nrow)
    print(f"[+] Success! Generated image grid exported to: {args.output_path}")

if __name__ == "__main__":
    main()