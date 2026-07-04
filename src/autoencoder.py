import torch
import torch.nn as nn

class DigitVAE(nn.Module):
    """
    A lightweight Variational Autoencoder (VAE).
    Mimics the continuous, regularized latent space behavior of the 
    Stable Diffusion VAE used in the Drifting Models paper.
    """
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        
        # --- Encoder Engine ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, 14, 14]
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 32, 7, 7]
            nn.ReLU(inplace=True),
            nn.Flatten()                                           # [B, 1568]
        )
        
        # --- Variational Projections ---
        # Instead of one flat vector, we predict a distribution (mean and log variance)
        self.fc_mu = nn.Linear(1568, latent_dim)
        self.fc_logvar = nn.Linear(1568, latent_dim)
        
        # --- Decoder Engine ---
        self.decoder_input = nn.Linear(latent_dim, 1568)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.Tanh() # Matches data normalized to [-1, 1]
        )

    def encode(self, x):
        """Returns the distribution parameters for the latent space."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Samples a specific vector from the predicted distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decodes the latent vector back into image space."""
        h = self.decoder_input(z).view(-1, 32, 7, 7)
        return self.decoder(h)

    def forward(self, x):
        """Full pass: used during the pre-training phase of the VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar