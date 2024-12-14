import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeconvLayer(nn.Module):
    def __init__(self, input_shape, output_shape, filter_shape, activation, name):
        super(DeconvLayer, self).__init__()
        self.name = name
        self.activation = activation
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.strides = (2, 2)
        self.padding = (filter_shape[2] // 2, filter_shape[3] // 2)
        self.output_padding = (
            (
                output_shape[0]
                - (
                    (input_shape[0] - 1) * self.strides[0]
                    - 2 * self.padding[0]
                    + filter_shape[2]
                )
            )
            % self.strides[0],
            (
                output_shape[1]
                - (
                    (input_shape[1] - 1) * self.strides[1]
                    - 2 * self.padding[1]
                    + filter_shape[3]
                )
            )
            % self.strides[1],
        )
        scale = 1.0 / np.prod(filter_shape[:3])
        torch.manual_seed(int(np.random.randint(0, 1000)))
        self.weight = nn.Parameter(torch.empty(filter_shape).uniform_(-scale, scale))

    def forward(self, x):
        x = F.conv_transpose2d(
            x,
            self.weight,
            stride=self.strides,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation == "split":
            x1, x2 = torch.chunk(x, 2, dim=1)
            return torch.sigmoid(x1), x2
        else:
            return x
