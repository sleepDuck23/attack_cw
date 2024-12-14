import time

import numpy as np
import torch
import torch.optim as optim
from torch.nn.functional import softmax


def logsumexp(x):
    x_max, _ = torch.max(x, dim=0, keepdim=True)
    x_ = x - x_max
    tmp = torch.log(torch.clamp(torch.sum(torch.exp(x_), dim=0), min=1e-9))
    return tmp + x_max.squeeze(0)


def bayes_classifier(x, enc, dec, ll, dimY, lowerbound, K=1, beta=1.0):
    enc_conv, enc_mlp = enc
    fea = enc_conv(x)
    N = x.size(0)
    logpxy = []
    for i in range(dimY):
        y = torch.zeros(N, dimY, device=x.device)
        y[:, i] = 1
        bound = lowerbound(x, fea, y, enc_mlp, dec, ll, K, IS=True, beta=beta)
        logpxy.append(bound.unsqueeze(1))
    logpxy = torch.cat(logpxy, dim=1)
    pyx = softmax(logpxy, dim=1)
    return pyx, logpxy


def construct_optimizer(X_ph, Y_ph, enc, dec, ll, K, vae_type="A"):
    enc_conv, enc_mlp = enc
    if ll in ["l1_logistic", "l2_logistic", "gaussian_logistic", "laplace_logistic"]:
        alpha = 0.01
        X_ = alpha + (1 - alpha * 2) * X_ph
        X_ = torch.log(X_) - torch.log(1 - X_)
        ll_ = ll.split("_")[0]
    else:
        X_ = X_ph
        ll_ = ll

    fea = enc_conv(X_)

    if vae_type == "A":
        from .lowerbound_functions import lowerbound_A as lowerbound_func
    elif vae_type == "B":
        from .lowerbound_functions import lowerbound_B as lowerbound_func
    elif vae_type == "C":
        from .lowerbound_functions import lowerbound_C as lowerbound_func
    elif vae_type == "D":
        from .lowerbound_functions import lowerbound_D as lowerbound_func
    elif vae_type == "E":
        from .lowerbound_functions import lowerbound_E as lowerbound_func
    elif vae_type == "F":
        from .lowerbound_functions import lowerbound_F as lowerbound_func
    elif vae_type == "G":
        from .lowerbound_functions import lowerbound_G as lowerbound_func
    else:
        raise ValueError(f"Unknown VAE type: {vae_type}")

    beta_ph = 1.0
    bound = lowerbound_func(X_, fea, Y_ph, enc_mlp, dec, ll_, K, IS=True, beta=beta_ph)
    bound = torch.mean(bound)

    dimY = Y_ph.size(-1)

    def train_step(optimizer, X, Y, beta):
        optimizer.zero_grad()
        fea = enc_conv(X)
        bound = lowerbound_func(X, fea, Y, enc_mlp, dec, ll_, K, IS=True, beta=beta)
        loss = -torch.mean(bound)
        loss.backward()
        optimizer.step()
        return loss.item()

    def fit(optimizer, X, Y, n_iter, lr, beta):
        N = X.size(0)
        print(f"Training for {n_iter} epochs with lr={lr:.5f}, beta={beta:.2f}")
        begin = time.time()
        batch_size = X_ph.size(0)
        n_iter_vae = N // batch_size
        for iteration in range(1, n_iter + 1):
            enc_conv.train()
            indices = torch.randperm(N)
            bound_total = 0.0
            for j in range(n_iter_vae):
                start = j * batch_size
                end = (j + 1) * batch_size
                batch_indices = indices[start:end]
                X_batch = X[batch_indices]
                Y_batch = Y[batch_indices]
                cost = train_step(optimizer, X_batch, Y_batch, beta)
                bound_total += cost / n_iter_vae
            end = time.time()
            print(
                f"Iter {iteration}, logp(x|y)={bound_total:.2f}, time={end - begin:.2f}"
            )
            begin = end

    def eval(X, Y, data_name="train", beta=1.0):
        enc_conv.eval()
        N = X.size(0)
        begin = time.time()
        batch_size = X_ph.size(0)
        n_batch = N // batch_size
        acc_total = 0.0
        bound_total = 0.0
        with torch.no_grad():
            for j in range(n_batch):
                start = j * batch_size
                end = min((j + 1) * batch_size, N)
                X_batch = X[start:end]
                Y_batch = Y[start:end]
                fea = enc_conv(X_batch)
                bound = lowerbound_func(
                    X_batch, fea, Y_batch, enc_mlp, dec, ll_, K, IS=True, beta=beta
                )
                y_pred = bayes_classifier(
                    X_batch, enc, dec, ll_, dimY, lowerbound_func, K=10, beta=beta
                )
                correct = (
                    (torch.argmax(Y_batch, dim=1) == torch.argmax(y_pred, dim=1))
                    .float()
                    .mean()
                )
                acc_total += correct.item() / n_batch
                bound_total += torch.mean(bound).item() / n_batch
        end = time.time()
        print(
            f"{data_name} data approx Bayes classifier acc={acc_total * 100:.2f}, "
            f"bound={bound_total:.2f}, time={end - begin:.2f}, beta={beta:.2f}"
        )
        return acc_total, bound_total

    return fit, eval


if __name__ == "__main__":
    import os
    import sys
    from importlib import import_module

    import torch.nn.functional as F

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from torch.utils.data import DataLoader

    from models.conv_encoder_mnist import GaussianConvEncoder
    from utils.utils import load_data

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    digits = [1, 7]
    # Hyperparameters
    batch_size = 8
    input_shape = (1, 28, 28)  # MNIST images
    dimH = 128
    dimZ = 32
    dimY = len(digits)
    n_channel = 16
    ll = "bernoulli"  # Likelihood function
    K = 5
    n_iter = 1  # Number of training iterations
    lr = 1e-3  # Learning rate
    beta = 1.0  # Beta parameter for scaling

    # Load MNIST dataset
    train_dataset, test_dataset = load_data(
        "mnist", path="./data", labels=digits, conv=True, seed=seed
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Prepare data tensors
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

    letters = ["A", "B", "C", "D", "E", "F", "G"]

    for letter in letters:
        print(f"\nTesting VAE type {letter}")

        # Import the appropriate lowerbound function
        module_name = f"models.conv_generator_mnist_{letter}"
        module = import_module(module_name)
        GeneratorClass = getattr(module, "Generator")

        # Instantiate encoder and decoder models
        encoder = GaussianConvEncoder(
            input_shape=input_shape,
            dimH=dimH,
            dimZ=dimZ,
            dimY=dimY,
            n_channel=n_channel,
            name=f"conv_encoder_{letter}",
        ).to(device)

        generator = GeneratorClass(
            input_shape=input_shape,
            dimH=dimH,
            dimZ=dimZ,
            dimY=dimY,
            n_channel=n_channel,
            last_activation="sigmoid",
            name=f"generator_{letter}",
        ).to(device)

        # Define encoder components
        enc_conv = encoder.encoder_conv
        enc_mlp = encoder.enc_mlp

        # Define decoder components based on the VAE type
        if letter == "A":
            dec = (generator.pyz_params, generator.pxzy_params)
        elif letter == "B":
            dec = (generator.pzy_params, generator.pxzy_params)
        elif letter == "C":
            dec = (generator.pyzx_params, generator.pxz_params)
        elif letter == "D":
            dec = (generator.pyzx_params, generator.pzx_params)
        elif letter == "E":
            dec = (generator.pyz_params, generator.pzx_params)
        elif letter == "F":
            dec = (generator.pyz_params, generator.pxz_params)
        elif letter == "G":
            dec = (generator.pzy_params, generator.pxz_params)
        else:
            raise ValueError(f"Unknown VAE type: {letter}")

        # Construct optimizer and training functions
        enc = (enc_conv, enc_mlp)
        X_ph = torch.zeros(batch_size, *input_shape).to(device)
        Y_ph = torch.zeros(batch_size, dimY).to(device)
        fit, eval_fn = construct_optimizer(X_ph, Y_ph, enc, dec, ll, K, vae_type=letter)

        # Combine encoder and decoder parameters for optimization
        model_params = list(encoder.parameters()) + list(generator.parameters())
        optimizer = optim.Adam(model_params, lr=lr)

        # Train the model
        fit(
            optimizer=optimizer,
            X=X_train,
            Y=Y_train,
            n_iter=n_iter,
            lr=lr,
            beta=beta,
        )

        # Evaluate the model on training and test data
        eval_fn(X=X_train, Y=Y_train, data_name="Train", beta=beta)

        eval_fn(X=X_test, Y=Y_test, data_name="Test", beta=beta)
