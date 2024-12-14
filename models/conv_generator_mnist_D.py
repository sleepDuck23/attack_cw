import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnet import ConvNet, construct_filter_shapes
from .mlp import MLPLayer

"""
generator p(x)p(z|x)p(y|x, z), DFX
note here this is actually a discriminative model: we assume p(x) = p_D(x)
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

        # Construct p(z|x) using ConvNet and MLP
        self.gen_conv2 = ConvNet(
            input_shape,
            filter_shapes,
            fc_layer_sizes,
            activation="relu",
            last_activation="relu",
        )
        print(
            f"Generator shared Conv Net {name}_pzx_conv architecture: "
            f"{self.gen_conv2.get_output_shape()}, {fc_layer_sizes}"
        )

        fc_layers_pzx = [dimH, dimH, dimZ * 2]
        self.pzx_layers = nn.ModuleList()
        for i in range(len(fc_layers_pzx) - 1):
            activation = "linear" if i + 1 == len(fc_layers_pzx) - 1 else "relu"
            layer_name = f"{name}_pzx_l{i}"
            self.pzx_layers.append(
                MLPLayer(fc_layers_pzx[i], fc_layers_pzx[i + 1], activation, layer_name)
            )

    def pyzx_params(self, z, x):
        fea = self.gen_conv(x)
        out = torch.cat([fea, z], dim=1)
        for layer in self.pyzx_layers:
            out = layer(out)
        return out

    def pzx_params(self, x):
        fea = self.gen_conv2(x)
        for layer in self.pzx_layers:
            fea = layer(fea)
        mu, log_sig = torch.chunk(fea, 2, dim=1)
        return mu, log_sig


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
        mu, log_sigma = generator.pzx_params(images)
        pyzx_output = generator.pyzx_params(z, images)

        print(f"Latent vector z shape: {z.shape}")
        print(f"p(z|x) mu shape: {mu.shape}, log_sigma shape: {log_sigma.shape}")
        print(f"p(y|z,x) output shape: {pyzx_output.shape}")
        break
