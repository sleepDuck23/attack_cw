import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnet import ConvNet, construct_filter_shapes
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

        # Define p(y|z, x)
        layer_channels = [n_channel, n_channel * 2, n_channel * 4]
        filter_width = 5
        filter_shapes = construct_filter_shapes(layer_channels, filter_width)
        self.gen_conv = ConvNet(
            input_shape=input_shape,
            filter_shapes=filter_shapes,
            fc_layer_sizes=[dimH],
            activation="relu",
            last_activation="relu",
            dropout=False,  # Add dropout if needed
        )
        conv_output_shape = self.gen_conv.get_output_shape()

        print(
            f"Generator shared Conv Net {name}_pyzx_conv architecture: {conv_output_shape}, {[dimH]}"
        )

        fc_layers_pyzx = [dimZ + dimH, dimH, dimY]
        self.pyzx_layers = nn.ModuleList()
        for i in range(len(fc_layers_pyzx) - 1):
            activation = "linear" if i + 1 == len(fc_layers_pyzx) - 1 else "relu"
            layer_name = f"{name}_pyzx_mlp_l{i}"
            self.pyzx_layers.append(
                MLPLayer(
                    fc_layers_pyzx[i], fc_layers_pyzx[i + 1], activation, layer_name
                )
            )

        # Define p(x|z)
        self.decoder_input_shape = [
            (4, 4, n_channel * 4),
            (8, 8, n_channel * 2),
            (16, 16, n_channel),
            (input_shape[1], input_shape[2], input_shape[0]),
        ]
        fc_layers_pxz = [dimZ, dimH, int(np.prod(self.decoder_input_shape[0]))]

        self.mlp_layers_pxz = nn.ModuleList()
        for i in range(len(fc_layers_pxz) - 1):
            layer_name = f"{name}_pxz_mlp_l{i}"
            self.mlp_layers_pxz.append(
                MLPLayer(fc_layers_pxz[i], fc_layers_pxz[i + 1], "relu", layer_name)
            )

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
                input_shape[-1],
                output_shape[-1],
                filter_width,
                filter_width,
            )
            self.conv_layers.append(
                DeconvLayer(
                    input_shape, output_shape, filter_shape, activation, layer_name
                )
            )

    def pyzx_params(self, z, x):
        fea = self.gen_conv(x)
        out = torch.cat([fea, z], dim=1)
        for layer in self.pyzx_layers:
            out = layer(out)
        return out

    def pxz_params(self, z):
        out = z
        for layer in self.mlp_layers_pxz:
            out = layer(out)

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


def sample_gaussian(mu, log_sig):
    return mu + torch.exp(log_sig) * torch.randn_like(mu)


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
        labels_one_hot = F.one_hot(labels, num_classes=dimY).float()
        pxz_output = generator.pxz_params(z)
        pyzx_output = generator.pyzx_params(z, images)

        print(f"Latent vector z shape: {z.shape}")
        print(f"p(x|z) output shape: {pxz_output.shape}")
        print(f"p(y|z,x) output shape: {pyzx_output.shape}")
        break
