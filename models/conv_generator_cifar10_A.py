import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .deconv import DeconvLayer
from .mlp import MLPLayer


class Generator(nn.Module):
    def __init__(
        self,
        input_shape,
        dimH,
        dimZ,
        dimY,
        n_channel,
        last_activation,
        name="generator",
    ):
        super(Generator, self).__init__()

        # Define p(y|z) as an MLP
        fc_layers_pyz = [dimZ, dimH, dimY]
        self.pyz_layers = nn.ModuleList()
        for i in range(len(fc_layers_pyz) - 1):
            activation = "linear" if i + 1 == len(fc_layers_pyz) - 1 else "relu"
            layer_name = f"{name}_pyz_l{i}"
            self.pyz_layers.append(
                MLPLayer(fc_layers_pyz[i], fc_layers_pyz[i + 1], activation, layer_name)
            )

        # Define p(x|y, z)
        filter_width = 5
        self.decoder_input_shape = [
            (4, 4, n_channel * 4),  # Initial shape after MLP
            (8, 8, n_channel * 2),
            (16, 16, n_channel),
            (input_shape[1], input_shape[2], input_shape[0]),
        ]
        fc_layers_pxzy = [dimZ + dimY, dimH, int(np.prod(self.decoder_input_shape[0]))]

        # First MLP layers for p(x|y, z)
        self.mlp_layers_pxzy = nn.ModuleList()
        for i in range(len(fc_layers_pxzy) - 1):
            layer_name = f"{name}_pxzy_mlp_l{i}"
            self.mlp_layers_pxzy.append(
                MLPLayer(fc_layers_pxzy[i], fc_layers_pxzy[i + 1], "relu", layer_name)
            )

        # Deconvolutional layers for p(x|y, z)
        self.conv_layers = nn.ModuleList()
        for i in range(len(self.decoder_input_shape) - 1):
            activation = (
                last_activation
                if i + 1 == len(self.decoder_input_shape) - 1
                else "relu"
            )
            layer_name = f"{name}_conv_l{i}"
            output_shape = self.decoder_input_shape[i + 1]
            input_shape = self.decoder_input_shape[i]
            filter_shape = (
                input_shape[-1],  # in_channels (input shape's channel dimension)
                output_shape[-1],  # out_channels (output shape's channel dimension)
                filter_width,  # filter height
                filter_width,  # filter width
            )
            self.conv_layers.append(
                DeconvLayer(
                    input_shape, output_shape, filter_shape, activation, layer_name
                )
            )

    def pyz_params(self, z):
        out = z
        for layer in self.pyz_layers:
            out = layer(out)
        return out

    def pxzy_params(self, z, y):
        out = torch.cat([z, y], dim=1)
        for layer in self.mlp_layers_pxzy:
            out = layer(out)

        # Reshape MLP output to match the first decoder input shape
        decoder_initial_shape = self.decoder_input_shape[0]
        out = out.view(
            out.shape[0],
            decoder_initial_shape[2],
            decoder_initial_shape[0],
            decoder_initial_shape[1],
        )

        for layer in self.conv_layers:
            out = layer(out)
        return out


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from torch.utils.data import DataLoader

    from utils.cifar10 import CustomCIFAR10Dataset

    # CIFAR-10 dimensions
    input_shape = (3, 32, 32)  # CIFAR-10 images
    dimH = 256
    dimZ = 128
    dimY = 10
    n_channel = 64

    # Instantiate generator
    generator = Generator(
        input_shape, dimH, dimZ, dimY, n_channel, last_activation="sigmoid"
    )

    # Load Custom CIFAR-10 Dataset
    cifar_dataset = CustomCIFAR10Dataset(
        path="./data", train=True, labels=[0, 1], conv=True
    )
    cifar_loader = DataLoader(cifar_dataset, batch_size=64, shuffle=True)

    # Test generator with CIFAR-10 data
    for images, labels in cifar_loader:
        z = torch.randn((labels.size(0), dimZ))  # Random latent vector
        labels_one_hot = F.one_hot(
            labels, num_classes=dimY
        ).float()  # One-hot encode labels
        pyz_output = generator.pyz_params(z)
        pxzy_output = generator.pxzy_params(z, labels_one_hot)
        print(f"Latent vector z shape: {z.shape}")
        print(f"p(y|z) output shape: {pyz_output.shape}")
        print(f"p(x|y,z) output shape: {pxzy_output.shape}")
        break
