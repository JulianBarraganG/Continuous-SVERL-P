import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np

#Define the Neural Conditioner, which comes from this paper: https://arxiv.org/pdf/1902.08401
#It is a basically a variational autoencoder with GAN-like training, where the generator is a neural conditioner, and the discriminator is a neural network that tries to distinguish between real and fake data.

class NC(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        """
        Neural Conditioner (NC) model for missing data imputation. We use it to predict missing features in a state vector.
        
        Parameters
        ----------
            input_dim (int): Dimension of the input data.
            latent_dim (int): Dimension of the latent space. Should typically be smaller than input_dim.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: [x_a (input_dim) + a (input_dim) + r (input_dim)] = 3*input_dim
        self.encoder = nn.Sequential(
            nn.Linear(3 * input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Decoder: [h (latent_dim) + x_a (input_dim) + a (input_dim) + r (input_dim)] = latent_dim + 3*input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 3 * input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x_a, a, r):
        """
        Forward pass through the NC model.

        Parameters
        ----------
            x_a (torch.Tensor): Observed features (input_dim).
            a (torch.Tensor): Mask indicating observed features (input_dim).
            r (torch.Tensor): Mask indicating missing features (input_dim).

        Returns
        -------
            torch.Tensor: Reconstructed features (input_dim).
        """
        # Input shapes: [batch_size, input_dim] for x_a, a, r
        # z shape: [batch_size, latent_dim]
        encoder_input = torch.cat([x_a, a, r], dim=1)
        h = self.encoder(encoder_input)
        
        decoder_input = torch.cat([h, x_a, a, r], dim=1)
        return self.decoder(decoder_input) * r

    def pred(self, observed_features, mask): 
        """
        Predict missing features using the trained NC model.
        
        Parameters
        ----------
            observed_features (numpy.ndarray): Observed features (input_dim).
            mask (numpy.ndarray): Mask indicating observed features (input_dim).
        Returns
        -------
            numpy.ndarray: Predicted state vector (input_dim). The observed features are kept as is, and the missing features are predicted.
        """
        # Convert to PyTorch tensors
        observed_features *= mask
        x_a = torch.FloatTensor(np.nan_to_num(observed_features, nan=0.0) * mask)
        a = torch.FloatTensor(mask)
        r = 1 - a  # Predict missing features
        
        # Generate predictions (multiple samples for uncertainty)
        with torch.no_grad():
            z = torch.randn(self.latent_dim)
            x_a = x_a.unsqueeze(0)
            a = a.unsqueeze(0)
            r = r.unsqueeze(0)
            pred = self(x_a, a, r, z).mean(0).numpy()

        for i in range(len(pred)):
            if mask[i] == 1: 
                pred[i] = observed_features[i]
        return pred


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        """ 
        Discriminator model for the Neural Conditioner (NC). It distinguishes between real and fake data, 
        and is trained using the output of the NC model GAN-style
        
        Parameters
        ----------
            input_dim (int): Dimension of the input data.
        """
        super().__init__()
        # Input: [x_r (input_dim) + x_a (input_dim) + a (input_dim) + r (input_dim)] = 4*input_dim
        self.net = nn.Sequential(
            nn.Linear(4 * input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_r, x_a, a, r):
        """
        Forward pass through the Discriminator model.
        
        Parameters
        ----------
            x_r (torch.Tensor): Reconstructed features (input_dim).
            x_a (torch.Tensor): Observed features (input_dim).
            a (torch.Tensor): Mask indicating observed features (input_dim).
            r (torch.Tensor): Mask indicating missing features (input_dim).
        Returns
        -------
            torch.Tensor: Discriminator output (probability of being real).
        """
        inputs = torch.cat([x_r, x_a, a, r], dim=1)
        return self.net(inputs)

def train_nc(nc, discriminator, dataloader, epochs):
    """
    Train the Neural Conditioner (NC) and Discriminator models.
    
    Parameters
    ----------
        nc (NC): Parametrized Neural Conditioner model.
        discriminator (Discriminator): Parametrized Discriminator model.
        dataloader (DataLoader): DataLoader for the training data.
        epochs (int): Number of training epochs.
    """
    nc.train()
    discriminator.train()
    
    opt_nc = torch.optim.Adam(nc.parameters(), lr=1e-4)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for x_real in dataloader:
            batch_size = x_real.size(0)
            input_dim = x_real.size(1)
            
            # Create random masks
            a = torch.zeros_like(x_real)
            r = torch.zeros_like(x_real)
            
            # Ensure at least 1 feature observed and 1 predicted
            for i in range(batch_size):
                num_observed = torch.randint(1, input_dim, (1,))  # At least 1, at most input_dim-1
                observed_idx = torch.randperm(input_dim)[:num_observed]
                a[i, observed_idx] = 1
                r[i, :] = 1 - a[i, :]
                if r[i].sum() == 0:  # Ensure at least 1 predicted
                    r[i, torch.randint(0, input_dim, (1,))] = 1
            
            # Generate samples
            #z = torch.randn(batch_size, nc.latent_dim)
            x_a = x_real * a
            x_r_fake = nc(x_a, a, r)
            x_r_real = x_real * r
            
            # Train discriminator
            opt_d.zero_grad()
            d_real = discriminator(x_r_real, x_a, a, r)
            d_fake = discriminator(x_r_fake.detach(), x_a, a, r)
            loss_d = - (torch.log(d_real) + torch.log(1 - d_fake)).mean()
            loss_d.backward()
            opt_d.step()
            
            # Train generator
            opt_nc.zero_grad()
            d_fake = discriminator(x_r_fake, x_a, a, r)
            loss_g = - torch.log(d_fake).mean()
            loss_g.backward()
            opt_nc.step()
        
        print(f"Epoch {epoch+1}/{epochs} | G Loss: {loss_g.item():.4f} | D Loss: {loss_d.item():.4f}")

