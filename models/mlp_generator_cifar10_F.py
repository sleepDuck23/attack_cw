import torch
import torch.nn as nn

from .mlp import MLPLayer


class Generator(nn.Module):
    def __init__(self, dimX, dimH, dimZ, dimY, last_activation, name="generator"):
        super(Generator, self).__init__()
        self.name = name

        # Construct p(y|z)
        fc_layers_pyz = [dimZ, dimH, dimH, dimY]
        self.pyz_layers = nn.ModuleList()
        for i in range(len(fc_layers_pyz) - 1):
            activation = "linear" if i + 1 == len(fc_layers_pyz) - 1 else "relu"
            layer_name = f"{name}_pyz_mlp_l{i}"
            self.pyz_layers.append(
                MLPLayer(fc_layers_pyz[i], fc_layers_pyz[i + 1], activation, layer_name)
            )

        # Construct p(x|z)
        fc_layers_pxz = [dimZ, dimH, dimH, dimX]
        self.pxz_layers = nn.ModuleList()
        for i in range(len(fc_layers_pxz) - 1):
            activation = last_activation if i + 1 == len(fc_layers_pxz) - 1 else "relu"
            layer_name = f"{name}_pxz_mlp_l{i}"
            self.pxz_layers.append(
                MLPLayer(fc_layers_pxz[i], fc_layers_pxz[i + 1], activation, layer_name)
            )

    def pyz_params(self, z):
        out = z
        for layer in self.pyz_layers:
            out = layer(out)
        return out

    def pxz_params(self, z):
        out = z
        for layer in self.pxz_layers:
            out = layer(out)
        return out


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
    generator = Generator(dimX, dimH, dimZ, dimY, last_activation="sigmoid")

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
        pxz_output = generator.pxz_params(z)
        print(f"Input latent z shape: {z.shape}")
        print(f"p(y|z) output shape: {pyz_output.shape}")
        print(f"p(x|z) output shape: {pxz_output.shape}")
        break
