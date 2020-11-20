import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, channels):
        super(UnFlatten, self).__init__()
        self.channels = channels

    def forward(self, input):
        return input.view(input.size(0), self.channels, 1, 1)

class UnFlatten1D(nn.Module):
    def __init__(self, channels, size):
        super(UnFlatten1D, self).__init__()
        self.channels = channels
        self.size = size

    def forward(self, input):
        return input.view(input.size(0), self.channels, self.size)

class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=1024, z_dim=32, gpu=True):
        self.gpu = gpu
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(h_dim),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        if self.gpu:
            eps = torch.cuda.FloatTensor(*mu.size()).normal_()
        else:
            eps = torch.FloatTensor(*mu.size()).normal_()
        z = mu + std * eps
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

class VAE1D(VAE):
    def __init__(self, image_channels=1, h_dim=1024, z_dim=32, gpu=True):
        assert h_dim % 4 == 0

        super(VAE1D, self).__init__()
        self.gpu = gpu

        self.encoder = nn.Sequential(
            nn.Conv1d(image_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=4),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=6, stride=4),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=4),
            nn.ReLU(),
            Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten1D(h_dim//4, 4),  # (batch, 1024) -> (batch, 256, 4)
            nn.ConvTranspose1d(h_dim//4, 128, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=6, stride=4),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=9, stride=4),
            nn.ReLU(),
            nn.ConvTranspose1d(32, image_channels, kernel_size=8, stride=4),
            nn.Sigmoid(),
        )
