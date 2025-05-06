# train_vae.py
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from train_dataset import PerilousDataset

# 1) define your Conv-VAE
class ConvVAE(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        # encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu    = nn.Linear(128*32*32, z_dim)
        self.fc_logvar= nn.Linear(128*32*32, z_dim)
        # decoder
        self.fc_dec   = nn.Linear(z_dim, 128*32*32)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (128,32,32)),
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,2,1), nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = (0.5*logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self,x):
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        out = self.dec(self.fc_dec(z))
        return out, mu, logvar

# 2) training loop
def train():
    ds    = PerilousDataset("data/perilous", img_size=256)
    dl    = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ConvVAE(z_dim=128).to(device)
    opt    = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1,51):
        total_loss = 0
        for imgs in dl:
            imgs = imgs.to(device)
            recon, mu, logvar = model(imgs)
            # BCE or MSE reconstruction loss
            recon_loss = nn.functional.mse_loss(recon, imgs, reduction="sum")
            # KL divergence
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 1e-3 * kld

            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch:>2}  loss: {total_loss/len(ds):.4f}")
        torch.save(model.state_dict(), f"checkpoints/vae_ep{epoch}.pth")

if __name__=="__main__":
    train()