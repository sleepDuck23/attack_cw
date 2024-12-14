from __future__ import print_function

import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from alg.vae_new import construct_optimizer
from utils.utils import init_variables, load_data, load_params, save_params


def main(data_name, vae_type, n_iter, batch_size, K, checkpoint, lr):
    dimY = 10

    if data_name == "mnist":
        if vae_type == "A":
            from models.conv_generator_mnist_A import Generator
        elif vae_type == "B":
            from models.conv_generator_mnist_B import Generator
        elif vae_type == "C":
            from models.conv_generator_mnist_C import Generator
        elif vae_type == "D":
            from models.conv_generator_mnist_D import Generator
        elif vae_type == "E":
            from models.conv_generator_mnist_E import Generator
        elif vae_type == "F":
            from models.conv_generator_mnist_F import Generator
        elif vae_type == "G":
            from models.conv_generator_mnist_G import Generator
        else:
            raise ValueError(f"Unknown VAE type: {vae_type}")
        from models.conv_encoder_mnist import GaussianConvEncoder as encoder

        input_shape = (1, 28, 28)
        n_channel = 64
        dimZ = 64
        dimH = 500
    elif data_name == "cifar10" or data_name == "gtsrb":
        if vae_type == "A":
            from models.conv_generator_cifar10_A import Generator
        elif vae_type == "B":
            from models.conv_generator_cifar10_B import Generator
        elif vae_type == "C":
            from models.conv_generator_cifar10_C import Generator
        elif vae_type == "D":
            from models.conv_generator_cifar10_D import Generator
        elif vae_type == "E":
            from models.conv_generator_cifar10_E import Generator
        elif vae_type == "F":
            from models.conv_generator_cifar10_F import Generator
        elif vae_type == "G":
            from models.conv_generator_cifar10_G import Generator
        else:
            raise ValueError(f"Unknown VAE type: {vae_type}")

        from models.conv_encoder_cifar10 import GaussianConvEncoder as encoder

        input_shape = (3, 32, 32)
        n_channel = 128
        dimZ = 128
        dimH = 1000
    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    if data_name == "gtsrb":
        dimY = 43

    generator = Generator(input_shape, dimH, dimZ, dimY, n_channel, "sigmoid", "gen")
    encoder = encoder(input_shape, dimH, dimZ, dimY, n_channel, "enc")
    enc_conv = encoder.encoder_conv
    enc_mlp = encoder.enc_mlp
    if vae_type == "A":
        dec = (generator.pyz_params, generator.pxzy_params)
    elif vae_type == "B":
        dec = (generator.pzy_params, generator.pxzy_params)
    elif vae_type == "C":
        dec = (generator.pyzx_params, generator.pxz_params)
    elif vae_type == "D":
        dec = (generator.pyzx_params, generator.pzx_params)
    elif vae_type == "E":
        dec = (generator.pyz_params, generator.pzx_params)
    elif vae_type == "F":
        dec = (generator.pyz_params, generator.pxz_params)
    elif vae_type == "G":
        dec = (generator.pzy_params, generator.pxz_params)
    else:
        raise ValueError(f"Unknown VAE type: {vae_type}")

    X_ph = torch.zeros(batch_size, *input_shape)
    Y_ph = torch.zeros(batch_size, dimY)
    ll = "l2"
    fit, eval_acc = construct_optimizer(
        X_ph, Y_ph, (enc_conv, enc_mlp), dec, ll, K, vae_type
    )

    train_dataset, test_dataset = load_data(
        data_name, path="./data", labels=None, conv=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    X_train = []
    Y_train = []
    for x_batch, y_batch in train_loader:
        X_train.append(x_batch)
        Y_train.append(F.one_hot(y_batch, num_classes=dimY).float())
    X_train = torch.cat(X_train)
    Y_train = torch.cat(Y_train)

    X_test = []
    Y_test = []
    for x_batch, y_batch in test_loader:
        X_test.append(x_batch)
        Y_test.append(F.one_hot(y_batch, num_classes=dimY).float())
    X_test = torch.cat(X_test)
    Y_test = torch.cat(Y_test)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    # initialise sessions
    if not os.path.isdir("save/"):
        os.mkdir("save/")
        print("create path save/")
    path_name = data_name + "_conv_vae_%s/" % (vae_type + "_" + str(dimZ))
    if not os.path.isdir("save/" + path_name):
        os.mkdir("save/" + path_name)
        print("create path save/" + path_name)
    filename = "save/" + path_name + "checkpoint"
    if checkpoint < 0:
        print("training from scratch")
        init_variables(encoder)
        init_variables(generator)
    else:
        load_params((encoder, generator), filename, checkpoint)
    checkpoint += 1

    encoder = encoder.to(device)
    generator = generator.to(device)

    n_iter_ = min(n_iter, 20)
    beta = 1.0
    model_params = list(encoder.parameters()) + list(generator.parameters())
    optimizer = optim.Adam(model_params, lr=lr)
    for i in range(int(n_iter / n_iter_)):
        fit(optimizer, X_train, Y_train, n_iter_, lr, beta)
        # print training and test accuracy
        eval_acc(X_test, Y_test, "test", beta)

    save_params((encoder, generator), filename, checkpoint)
    checkpoint += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("vae_type", type=str, help="Type of VAE")
    parser.add_argument("checkpoint", type=int, default=-1, help="Checkpoint index")
    parser.add_argument(
        "--data_name",
        type=str,
        help="Name of dataset",
        default="mnist",
        choices=["mnist", "cifar10", "gtsrb"],
    )
    parser.add_argument(
        "--n_iter", type=int, default=100, help="Number of training iterations"
    )
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument(
        "--K", type=int, default=10, help="Number of Monte Carlo samples"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    main(
        args.data_name,
        args.vae_type,
        args.n_iter,
        args.batch_size,
        args.K,
        args.checkpoint,
        args.lr,
    )
