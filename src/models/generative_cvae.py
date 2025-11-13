
import torch
import torch.nn as nn
import torch.nn.functional as F


class GenerativeCVAE(nn.Module):
    """
    Conditional Variational Autoencoder for generating drug feature vectors.
    Trains only on positive drug vectors (from known associated pairs).
    """

    def __init__(self, input_dim=135, latent_dim=64, hidden_dim=256):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # --------------- Encoder ---------------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # --------------- Decoder ---------------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    # ---------------------------------------------------------------
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    # ---------------------------------------------------------------
    def reparameterize(self, mu, logvar):
        """
        z = mu + std * eps
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---------------------------------------------------------------
    def decode(self, z):
        return self.decoder(z)

    # ---------------------------------------------------------------
    def forward(self, x):
        """
        x: [batch, input_dim]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ---------------------------- Loss --------------------------------

def vae_loss_function(recon_x, x, mu, logvar):
    """
    L = reconstruction_loss + KL divergence
    """

    recon_loss = F.mse_loss(recon_x, x, reduction="mean")

    # KL divergence: -0.5 * Σ(1 + logσ² - μ² - σ²)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl, recon_loss.item(), kl.item()
