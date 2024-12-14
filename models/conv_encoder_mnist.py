import torch
import torch.nn as nn

from .convnet import ConvNet, construct_filter_shapes
from .mlp import MLPLayer


class ConvEncoder(nn.Module):
    def __init__(self, input_shape, dimH, dimZ, dimY, n_channel, dropout, name):
        super(ConvEncoder, self).__init__()

        # Define layer channels and construct convolutional filters
        layer_channels = [n_channel] * 3
        filter_width = 5
        filter_shapes = construct_filter_shapes(layer_channels, filter_width)

        # Encoder ConvNet
        fc_layer_sizes = [dimH]
        self.enc_conv = ConvNet(
            input_shape=input_shape,
            filter_shapes=filter_shapes,
            fc_layer_sizes=fc_layer_sizes,
            activation="relu",
            last_activation="relu",
            dropout=dropout,
        )

        # Encoder MLP
        fc_layer = [dimH + dimY, dimH, dimZ]
        self.enc_mlp_layers = nn.ModuleList()
        for i in range(len(fc_layer) - 1):
            activation = "linear" if i + 2 == len(fc_layer) else "relu"
            layer_name = f"{name}_mlp_l{i}"
            self.enc_mlp_layers.append(
                MLPLayer(fc_layer[i], fc_layer[i + 1], activation, layer_name)
            )

    def apply_conv(self, x):
        return self.enc_conv(x)

    def apply_mlp(self, x, y):
        out = torch.cat([x, y], dim=1)
        for layer in self.enc_mlp_layers:
            out = layer(out)
        return out

    def forward(self, x):
        return self.enc_conv(x)


class GaussianConvEncoder(nn.Module):
    def __init__(self, input_shape, dimH, dimZ, dimY, n_channel, name):
        super(GaussianConvEncoder, self).__init__()
        self.encoder_conv = ConvEncoder(
            input_shape=input_shape,
            dimH=dimH,
            dimZ=dimZ * 2,
            dimY=dimY,
            n_channel=n_channel,
            dropout=False,
            name=name,
        )

    def enc_mlp(self, x, y):
        out = self.encoder_conv.apply_mlp(x, y)
        mu, log_sigma = torch.chunk(out, 2, dim=1)
        return mu, log_sigma

    def forward(self, x, y):
        conv_out = self.encoder_conv.apply_conv(x)
        mu, log_sigma = self.enc_mlp(conv_out, y)
        return mu, log_sigma


def sample_gaussian(mu, log_sigma):
    return mu + torch.exp(log_sigma) * torch.randn_like(mu)


def recon(x, y, gen, enc, sampling=False):
    out = enc(x, y)
    if isinstance(out, (list, tuple)):
        mu, log_sigma = out
        z = sample_gaussian(mu, log_sigma) if sampling else mu
    else:
        z = out
    return gen(z, y)


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from utils.mnist import CustomMNISTDataset

    # MNIST dimensions
    input_shape = (1, 28, 28)  # MNIST images are grayscale (1 channel)
    dimH = 256
    dimZ = 64
    dimY = 10
    n_channel = 32

    # Instantiate encoder
    encoder = GaussianConvEncoder(
        input_shape, dimH, dimZ, dimY, n_channel, name="conv_encoder"
    )

    # Load Custom MNIST Dataset
    mnist_dataset = CustomMNISTDataset(
        path="./data", train=True, digits=None, conv=True
    )
    mnist_loader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)

    # Test encoder with MNIST data
    for images, labels in mnist_loader:
        labels_one_hot = F.one_hot(
            labels, num_classes=dimY
        ).float()  # One-hot encode labels
        mu, log_sigma = encoder(images, labels_one_hot)
        sampled_z = sample_gaussian(mu, log_sigma)
        print(f"Input image shape: {images.shape}")
        print(f"Mu shape: {mu.shape}")
        print(f"Log Sigma shape: {log_sigma.shape}")
        print(f"Sampled Z shape: {sampled_z.shape}")
        break
