import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnet import ConvNet, construct_filter_shapes
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

        # Construct p(y|z, x)
        layer_channels = [n_channel, n_channel * 2, n_channel * 4]
        filter_width = 5
        filter_shapes = construct_filter_shapes(layer_channels, filter_width)

        # ConvNet for p(y|z, x)
        self.gen_conv = ConvNet(
            input_shape=input_shape,
            filter_shapes=filter_shapes,
            fc_layer_sizes=[dimH],
            activation="relu",
            last_activation="relu",
            dropout=False,
        )
        print(
            f"Generator shared Conv Net {name}_pyzx_conv architecture: {self.gen_conv.get_output_shape()}, {[dimH]}"
        )

        # MLP for p(y|z, x)
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

        # Construct p(z|x)
        # Another ConvNet for p(z|x)
        self.gen_conv2 = ConvNet(
            input_shape=input_shape,
            filter_shapes=filter_shapes,
            fc_layer_sizes=[dimH],
            activation="relu",
            last_activation="relu",
            dropout=False,
        )
        print(
            f"Generator shared Conv Net {name}_pzx_conv architecture: {self.gen_conv2.get_output_shape()}, {[dimH]}"
        )

        # MLP for p(z|x)
        fc_layers_pzx = [dimH, dimH, dimZ * 2]
        self.pzx_layers = nn.ModuleList()
        for i in range(len(fc_layers_pzx) - 1):
            activation = "linear" if i + 1 == len(fc_layers_pzx) - 1 else "relu"
            layer_name = f"{name}_pzx_mlp_l{i}"
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
        out = self.gen_conv2(x)
        for layer in self.pzx_layers:
            out = layer(out)
        mu, log_sig = torch.chunk(out, 2, dim=1)
        return mu, log_sig


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
        pzx_mu, pzx_log_sig = generator.pzx_params(images)
        pyzx_output = generator.pyzx_params(z, images)

        print(f"Latent vector z shape: {z.shape}")
        print(f"p(z|x) mu shape: {pzx_mu.shape}, log_sigma shape: {pzx_log_sig.shape}")
        print(f"p(y|z,x) output shape: {pyzx_output.shape}")
        break
