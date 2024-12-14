import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

alpha = 0.2
p = 0.3


class DropoutLayer(nn.Module):
    def __init__(self, p):
        super(DropoutLayer, self).__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=self.training)


def construct_filter_shapes(layer_channels, filter_width=5):
    return [(n_channel, filter_width, filter_width) for n_channel in layer_channels]


class ConvNet(nn.Module):
    def __init__(
        self,
        input_shape,
        filter_shapes,
        fc_layer_sizes,
        activation="relu",
        batch_norm=False,
        last_activation=None,
        weight_init="glorot_normal",
        subsample=None,
        dropout=False,
    ):
        super(ConvNet, self).__init__()
        self.layers = nn.ModuleList()
        self.weight_init = weight_init
        self.conv_output_shape = []

        num_conv_layers = len(filter_shapes)
        num_fc_layers = len(fc_layer_sizes)

        if last_activation is None:
            last_activation = activation

        if subsample is None:
            subsample = [(2, 2) for _ in range(num_conv_layers)]

        in_channels = input_shape[0]
        x = torch.zeros(1, *input_shape)  # Dummy input for shape calculation
        for l, (n_channel, height, width) in enumerate(filter_shapes):
            stride = subsample[l]
            padding = (height // 2, width // 2)  # Calculate padding for `same`
            conv = nn.Conv2d(
                in_channels,
                n_channel,
                kernel_size=(height, width),
                stride=stride,
                padding=padding,
                bias=not batch_norm,
            )
            self.layers.append(conv)
            in_channels = n_channel

            x = conv(x)
            self.conv_output_shape.append(list(x.shape[1:]))

            if batch_norm:
                self.layers.append(nn.BatchNorm2d(n_channel))

            if dropout:
                self.layers.append(nn.Dropout(p))

            if activation == "lrelu":
                self.layers.append(nn.LeakyReLU(negative_slope=alpha))
            elif activation == "elu":
                self.layers.append(nn.ELU())
            elif activation == "prelu":
                self.layers.append(nn.PReLU())
            else:
                self.layers.append(nn.ReLU())

        self.flatten = nn.Flatten()

        flatten_size = self.calculate_flatten_size(input_shape)

        # Pass the correct `in_features` to the first fully connected layer
        in_features = flatten_size
        for l, fc_size in enumerate(fc_layer_sizes):
            dense = nn.Linear(
                in_features,
                fc_size,
                bias=True if l + 1 == num_fc_layers else not batch_norm,
            )
            self.layers.append(dense)
            in_features = fc_size

            if batch_norm and l + 1 < num_fc_layers:
                self.layers.append(nn.BatchNorm1d(fc_size))

            if dropout and l + 1 < num_fc_layers:
                self.layers.append(nn.Dropout(p))

            if l + 1 < num_fc_layers:
                if activation == "lrelu":
                    self.layers.append(nn.LeakyReLU(negative_slope=alpha))
                elif activation == "elu":
                    self.layers.append(nn.ELU())
                elif activation == "prelu":
                    self.layers.append(nn.PReLU())
                else:
                    self.layers.append(nn.ReLU())
            else:
                if last_activation == "lrelu":
                    self.layers.append(nn.LeakyReLU(negative_slope=alpha))
                elif last_activation == "elu":
                    self.layers.append(nn.ELU())
                elif last_activation == "prelu":
                    self.layers.append(nn.PReLU())
                else:
                    self.layers.append(nn.ReLU())

        self.initialize_weights()

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Flatten input once we reach the first linear layer
                x = x.view(x.size(0), -1)
            x = layer(x)
        return x

    def initialize_weights(self):
        def init_function(layer):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if self.weight_init == "glorot_normal":
                    nn.init.xavier_normal_(layer.weight)
                elif self.weight_init == "glorot_uniform":
                    nn.init.xavier_uniform_(layer.weight)
                elif self.weight_init == "he_normal":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                elif self.weight_init == "he_uniform":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

        self.apply(init_function)

    def calculate_flatten_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
        return int(np.prod(x.shape[1:]))

    def get_output_shape(self):
        return self.conv_output_shape


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from torch.utils.data import DataLoader

    from utils.mnist import CustomMNISTDataset

    input_shape = (1, 28, 28)
    filter_shapes = construct_filter_shapes([32, 64], filter_width=3)
    fc_layer_sizes = [128, 10]

    model = ConvNet(
        input_shape,
        filter_shapes,
        fc_layer_sizes,
        activation="relu",
        batch_norm=True,
        dropout=True,
    )

    mnist_dataset = CustomMNISTDataset(
        path="./data", train=True, digits=None, conv=True
    )
    mnist_loader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)

    images, labels = next(iter(mnist_loader))
    output = model(images)
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {output.shape}")
