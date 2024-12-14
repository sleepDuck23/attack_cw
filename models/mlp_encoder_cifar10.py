import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLPLayer


class EncoderNet(nn.Module):
    def __init__(self, dimX, dimH, dimZ, dimY, n_layers, name="encoder"):
        super(EncoderNet, self).__init__()
        self.name = name
        fc_layers = [dimX + dimY] + [dimH] * n_layers + [dimZ]
        self.enc_layers = nn.ModuleList()

        for i in range(len(fc_layers) - 1):
            activation = "relu" if i + 2 < len(fc_layers) else "linear"
            layer_name = f"{name}_mlp_l{i}"
            self.enc_layers.append(
                MLPLayer(fc_layers[i], fc_layers[i + 1], activation, layer_name)
            )

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        for layer in self.enc_layers:
            out = layer(out)
        return out


class EncoderGaussian(nn.Module):
    def __init__(self, dimX, dimH, dimZ, dimY, n_layers, name="encoder_gaussian"):
        super(EncoderGaussian, self).__init__()
        self.mlp = EncoderNet(dimX, dimH, dimZ * 2, dimY, n_layers, name)

    def forward(self, x, y):
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)  # Flatten if input is 4D (e.g., images)
        out = self.mlp(x, y)
        mu, log_sigma = torch.chunk(out, 2, dim=1)
        return mu, log_sigma


def sample_gaussian(mu, log_sigma):
    return mu + torch.exp(log_sigma) * torch.randn_like(mu)


def recon(x, y, gen, enc, sampling=False):
    out = enc(x, y)
    if isinstance(out, (tuple, list)):
        mu, log_sigma = out
        z = sample_gaussian(mu, log_sigma) if sampling else mu
    else:
        z = out
    return gen(z, y)


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from torch.utils.data import DataLoader

    from utils.cifar10 import CustomCIFAR10Dataset

    # CIFAR-10 dimensions
    dimX = 3 * 32 * 32  # Flattened CIFAR-10 images
    dimH = 256
    dimZ = 128
    dimY = 10
    n_layers = 3

    # Instantiate encoder
    encoder = EncoderGaussian(dimX, dimH, dimZ, dimY, n_layers)

    # Load Custom CIFAR-10 Dataset
    cifar_dataset = CustomCIFAR10Dataset(
        path="./data", train=True, labels=[0, 1], conv=False
    )
    cifar_loader = DataLoader(cifar_dataset, batch_size=64, shuffle=True)

    # Test encoder with CIFAR-10 data
    for images, labels in cifar_loader:
        labels_one_hot = F.one_hot(
            labels, num_classes=dimY
        ).float()  # One-hot encode labels
        images = images.view(images.size(0), -1)  # Flatten images
        mu, log_sigma = encoder(images, labels_one_hot)
        sampled_z = sample_gaussian(mu, log_sigma)
        print(f"Input shape: {images.shape}")
        print(f"Mu shape: {mu.shape}")
        print(f"Log Sigma shape: {log_sigma.shape}")
        print(f"Sampled Z shape: {sampled_z.shape}")
        break
