import numpy as np
import torch
import torch.nn as nn


def init_weights(input_size, output_size, constant=1.0, seed=123):
    """Glorot and Bengio, 2010's initialization of network weights"""
    torch.manual_seed(seed)
    scale = constant * np.sqrt(6.0 / (input_size + output_size))
    return torch.empty(input_size, output_size).uniform_(-scale, scale)


class MLPLayer(nn.Module):
    def __init__(self, d_in, d_out, activation, name):
        super(MLPLayer, self).__init__()
        self.name = name
        self.W = nn.Parameter(init_weights(d_in, d_out))
        self.b = nn.Parameter(torch.zeros(d_out))
        self.activation = activation

    def forward(self, x):
        a = torch.matmul(x, self.W) + self.b
        if self.activation == "relu":
            return torch.relu(a)
        elif self.activation == "sigmoid":
            return torch.sigmoid(a)
        elif self.activation == "linear":
            return a
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from torch.utils.data import DataLoader

    from utils.mnist import CustomMNISTDataset

    # Load Custom MNIST Dataset
    mnist_dataset = CustomMNISTDataset(
        path="./data", train=True, digits=None, conv=False
    )
    mnist_loader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)

    # Define MLP layer
    input_size = 784  # MNIST images flattened
    hidden_size = 128
    output_size = 10

    # Create and test MLP layer
    mlp_hidden = MLPLayer(
        input_size, hidden_size, activation="relu", name="hidden_layer"
    )
    mlp_output = MLPLayer(
        hidden_size, output_size, activation="linear", name="output_layer"
    )

    # Test MLP with MNIST
    images, labels = next(iter(mnist_loader))
    images = images.view(images.size(0), -1)  # Flatten images for MLP
    hidden_output = mlp_hidden(images)
    final_output = mlp_output(hidden_output)

    print(f"Input shape: {images.shape}")
    print(f"Hidden layer output shape: {hidden_output.shape}")
    print(f"Final output shape: {final_output.shape}")
