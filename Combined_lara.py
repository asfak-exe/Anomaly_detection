import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset


class LARA_RuminateBlock(nn.Module):
    def __init__(self, input_dim=944, latent_dim=32, n_samples=3, N_mc=10):
        super(LARA_RuminateBlock, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_samples = n_samples
        self.N_mc = N_mc
  
    @torch.no_grad()
    def forward(self, x_new, vae_encoder, vae_decoder):
        batch_size = x_new.size(0)
        device = x_new.device

        # 1. Generate Simulated Historical Data using the frozen VAE
        mu,logvar= vae_encoder(x_new)
        x_recon_mean = vae_decoder(mu) 
        
        std_x = 0.1 
        x_hist_simulated = x_recon_mean.unsqueeze(1) + torch.randn(
            batch_size, self.n_samples, self.input_dim, device=device
        ) * std_x 

        # 2. Monte Carlo Sampling for Bayesian target estimation
        z_samples = torch.randn(self.N_mc, self.latent_dim, device=device)
        x_new_expanded = x_new.unsqueeze(1) 
        mc_recon = vae_decoder(z_samples) 
        mc_recon_expanded = mc_recon.unsqueeze(0) 

        log_p_x_new = -torch.sum((x_new_expanded - mc_recon_expanded) ** 2, dim=-1) / (2 * (std_x ** 2))
        log_p_x_hist = -torch.sum((x_hist_simulated - x_recon_mean.unsqueeze(1)) ** 2, dim=-1) / (2 * (std_x ** 2))
        log_p_x_hist_mean = torch.mean(log_p_x_hist, dim=-1, keepdim=True) 

        log_weights = log_p_x_new + log_p_x_hist_mean 
        weights = torch.softmax(log_weights, dim=-1) 
        z_target = torch.matmul(weights, z_samples)

        return z_target


class LARA_AdjustingFunctions(nn.Module):
    def __init__(self, input_dim=944, latent_dim=32):
        super(LARA_AdjustingFunctions, self).__init__()
        self.M_z = nn.Linear(latent_dim, latent_dim)
        self.M_x = nn.Linear(input_dim, input_dim)

    def forward(self, z_current, x_recon_current):
        z_adjusted = self.M_z(z_current)
        x_adjusted = self.M_x(x_recon_current)
        return z_adjusted, x_adjusted


def lara_convex_loss(z_adjusted, z_target, x_adjusted, x_new):
    huber_loss= nn.HuberLoss()
    return huber_loss(z_adjusted, z_target) + huber_loss(x_adjusted, x_new)




def retrain_lara(vae_encoder, vae_decoder, new_data_loader, input_dim, latent_dim, epochs=10, lr=0.001):
    """
    Retrains the LARA linear adapters on a new data distribution while keeping the VAE frozen.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure the core VAE does not receive gradient updates
    model = vae_encoder.__self__
    for param in model.parameters(): param.requires_grad = False
    
    # Initialize adapters
    ruminate_block = LARA_RuminateBlock(input_dim=input_dim, latent_dim=latent_dim).to(device)
    adjusting_funcs = LARA_AdjustingFunctions(input_dim=input_dim, latent_dim=latent_dim).to(device)
    
    optimizer = optim.Adam(adjusting_funcs.parameters(), lr=lr)
    adjusting_funcs.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_new in new_data_loader:
           
            x_new = x_new[0] if isinstance(x_new, list) or isinstance(x_new, tuple) else x_new
            x_new = x_new.to(device)
            
            optimizer.zero_grad()
            
            # Phase A: Estimate the Target
            with torch.no_grad():
                z_target = ruminate_block(x_new, vae_encoder, vae_decoder)
                z_current, _ = vae_encoder(x_new)
                x_recon_current = vae_decoder(z_current)
            
            # Phase B: Adjust and Learn
            z_adjusted, x_adjusted = adjusting_funcs(z_current, x_recon_current)
            loss = lara_convex_loss(z_adjusted, z_target, x_adjusted, x_new)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        
        print(f"LARA Retraining Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(new_data_loader):.4f}")
        
    return adjusting_funcs


def adapt_model(model, scaled_values, epochs=5, batch_size=256, lr=0.001):
    """Train LARA adapters on uploaded, already-scaled observations."""
    values = torch.tensor(scaled_values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(values), batch_size=batch_size, shuffle=False)
    return retrain_lara(
        model.encode,
        model.decode,
        loader,
        input_dim=values.shape[1],
        latent_dim=model.fc_mu.out_features,
        epochs=epochs,
        lr=lr,
    )


def detect_anomaly(x_live, vae_encoder, vae_decoder, adjusting_funcs, threshold):
    """
    Evaluates a live data point using the frozen VAE + trained LARA adapters.
    """
    vae_encoder.__self__.eval()
    adjusting_funcs.eval()
    
    with torch.no_grad():
        z_current, _ = vae_encoder(x_live)
        x_recon_current = vae_decoder(z_current)
        _, x_adjusted = adjusting_funcs(z_current, x_recon_current)
        
        reconstruction_error = torch.mean((x_live - x_adjusted) ** 2, dim=-1)
        is_anomaly = reconstruction_error > threshold
        
    return is_anomaly, reconstruction_error


