import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnet import ConvNet, construct_filter_shapes
from .deconv import DeconvLayer
from .mlp import MLPLayer

"""
generator p(z)p(x|z)p(y|x, z), DFZ
"""


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

        # Construct p(y|z, x) using ConvNet and MLP
        layer_channels = [n_channel for _ in range(3)]
        filter_width = 5
        filter_shapes = construct_filter_shapes(layer_channels, filter_width)
        fc_layer_sizes = [dimH]
        self.gen_conv = ConvNet(
            input_shape,
            filter_shapes,
            fc_layer_sizes,
            activation="relu",
            last_activation="relu",
        )
        print(
            f"Generator shared Conv Net {name}_pyzx_conv architecture: "
            f"{self.gen_conv.get_output_shape()}, {fc_layer_sizes}"
        )

        fc_layers_pyzx = [dimZ + dimH, dimH, dimY]
        self.pyzx_layers = nn.ModuleList()
        for i in range(len(fc_layers_pyzx) - 1):
            activation = "linear" if i + 1 == len(fc_layers_pyzx) - 1 else "relu"
            layer_name = f"{name}_pyzx_l{i}"
            self.pyzx_layers.append(
                MLPLayer(
                    fc_layers_pyzx[i], fc_layers_pyzx[i + 1], activation, layer_name
                )
            )

        # Construct p(x|z) using MLP and Deconv layers
        decoder_input_shape = [
            (4, 4, n_channel),
            (7, 7, n_channel),
            (14, 14, n_channel),
            (input_shape[1], input_shape[2], input_shape[0]),
        ]

        fc_layers_pxz = [dimZ, dimH, int(np.prod(decoder_input_shape[0]))]
        self.mlp_layers_pxz = nn.ModuleList()
        for i in range(len(fc_layers_pxz) - 1):
            layer_name = f"{name}_pxz_mlp_l{i}"
            self.mlp_layers_pxz.append(
                MLPLayer(fc_layers_pxz[i], fc_layers_pxz[i + 1], "relu", layer_name)
            )

        self.conv_layers = nn.ModuleList()
        for i in range(len(decoder_input_shape) - 1):
            activation = (
                last_activation if i + 1 == len(decoder_input_shape) - 1 else "relu"
            )
            layer_name = f"{name}_conv_l{i}"
            output_shape = decoder_input_shape[i + 1]
            input_shape = decoder_input_shape[i]
            filter_shape = (
                input_shape[-1],  # in_channels
                output_shape[-1],  # out_channels
                filter_width,  # filter height
                filter_width,  # filter width
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

        # Reshape MLP output to match the first decoder input shape
        decoder_initial_shape = self.conv_layers[0].input_shape
        out = out.view(
            out.shape[0],
            decoder_initial_shape[2],  # Channels
            decoder_initial_shape[0],  # Height
            decoder_initial_shape[1],  # Width
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

    from utils.mnist import CustomMNISTDataset

    # MNIST dimensions
    input_shape = (1, 28, 28)  # MNIST images
    dimH = 256
    dimZ = 128
    dimY = 10
    n_channel = 64

    # Instantiate generator
    generator = Generator(
        input_shape, dimH, dimZ, dimY, n_channel, last_activation="sigmoid"
    )

    # Load Custom MNIST Dataset
    mnist_dataset = CustomMNISTDataset(
        path="./data", train=True, digits=[0, 1], conv=True
    )
    mnist_loader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)

    # Test generator with MNIST data
    for images, labels in mnist_loader:
        labels_one_hot = F.one_hot(labels, num_classes=dimY).float()
        z = torch.randn((labels.size(0), dimZ))  # Random latent vector
        pyzx_output = generator.pyzx_params(z, images)
        pxz_output = generator.pxz_params(z)

        print(f"Latent vector z shape: {z.shape}")
        print(f"p(y|z,x) output shape: {pyzx_output.shape}")
        print(f"p(x|z) output shape: {pxz_output.shape}")
        break
