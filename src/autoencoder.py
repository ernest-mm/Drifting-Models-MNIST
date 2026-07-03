import torch
import torch.nn as nn

class DigitAutoencoder(nn.Module):
    """
    A lightweight, robust Convolutional Autoencoder that compresses MNIST digits
    down to a compact latent vector space where structural features dominate.
    """
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: [B, 1, 28, 28] -> [B, 16]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, 14, 14]
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 32, 7, 7]
            nn.ReLU(inplace=True),
            nn.Flatten(),                                          # [B, 32 * 7 * 7 = 1568]
            nn.Linear(1568, latent_dim)                            # [B, latent_dim]
        )
        
        # Decoder: [B, latent_dim] -> [B, 1, 28, 28]
        self.decoder_input = nn.Linear(latent_dim, 1568)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # [B, 16, 14, 14]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 1, 28, 28]
            nn.Tanh() # Matches our data distribution of [-1, 1]
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        h = self.decoder_input(z).view(-1, 32, 7, 7)
        return self.decoder(h)

    def forward(self, x):
        return self.decode(self.encode(x))