import torch
import torch.nn as nn

from .mlp import MLPLayer


class Generator(nn.Module):
    def __init__(self, dimX, dimH, dimZ, dimY, last_activation, name="generator"):
        super(Generator, self).__init__()
        self.name = name

        # Construct p(y|z)
        fc_layers_pyz = [dimZ, dimH, dimY]
        self.pyz_layers = nn.ModuleList()
        for i in range(len(fc_layers_pyz) - 1):
            activation = "linear" if i + 1 == len(fc_layers_pyz) - 1 else "relu"
            layer_name = f"{name}_pyz_mlp_l{i}"
            self.pyz_layers.append(
                MLPLayer(fc_layers_pyz[i], fc_layers_pyz[i + 1], activation, layer_name)
            )

        # Construct p(z|x)
        fc_layers_pzx = [dimX, dimH, dimH, dimZ * 2]
        self.pzx_layers = nn.ModuleList()
        for i in range(len(fc_layers_pzx) - 1):
            activation = "linear" if i + 1 == len(fc_layers_pzx) - 1 else "relu"
            layer_name = f"{name}_pzx_mlp_l{i}"
            self.pzx_layers.append(
                MLPLayer(fc_layers_pzx[i], fc_layers_pzx[i + 1], activation, layer_name)
            )

    def pyz_params(self, z):
        out = z
        for layer in self.pyz_layers:
            out = layer(out)
        return out

    def pzx_params(self, x):
        out = x
        for layer in self.pzx_layers:
            out = layer(out)
        mu, log_sigma = torch.chunk(out, 2, dim=1)
        return mu, log_sigma


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

    # Instantiate generator
    generator = Generator(dimX, dimH, dimZ, dimY, last_activation="linear")

    # Load Custom CIFAR-10 Dataset
    cifar_dataset = CustomCIFAR10Dataset(
        path="./data", train=True, labels=[0, 1], conv=False
    )
    cifar_loader = DataLoader(cifar_dataset, batch_size=64, shuffle=True)

    # Test generator with CIFAR-10 data
    for images, labels in cifar_loader:
        images = images.view(images.size(0), -1)  # Flatten images
        z = torch.randn((images.size(0), dimZ))  # Random latent vector
        pyz_output = generator.pyz_params(z)
        mu, log_sigma = generator.pzx_params(images)
        print(f"Input shape: {images.shape}")
        print(f"p(y|z) output shape: {pyz_output.shape}")
        print(f"p(z|x) Mu shape: {mu.shape}")
        print(f"p(z|x) Log Sigma shape: {log_sigma.shape}")
        break
