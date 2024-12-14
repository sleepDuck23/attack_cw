import numpy as np
import torch
import torch.nn.functional as F


def sample_gaussian(mu, log_sig, K):
    mu = mu.repeat(K, 1)
    log_sig = log_sig.repeat(K, 1)
    z = mu + torch.exp(log_sig) * torch.randn_like(mu)
    return mu, log_sig, z


def sample_gaussian_fix_randomness(mu, log_sig, K, seed):
    N = mu.size(0)
    mu = mu.repeat(K, 1)
    log_sig = log_sig.repeat(K, 1)
    np.random.seed(seed * 100)
    torch.manual_seed(seed * 100)
    eps = np.random.randn(K, mu.size(1))
    eps = np.repeat(eps, N, axis=0)
    eps = torch.tensor(eps, dtype=torch.float32)
    z = mu + torch.exp(log_sig) * eps
    return mu, log_sig, z


def log_gaussian_prob(x, mu=0.0, log_sig=torch.tensor(0.0)):
    logprob = (
        -(0.5 * np.log(2 * np.pi) + log_sig)
        - 0.5 * ((x - mu) / torch.exp(log_sig)) ** 2
    )
    return logprob.sum(dim=list(range(1, x.dim())))


def log_bernoulli_prob(x, p=0.5):
    logprob = x * torch.log(torch.clamp(p, 1e-9, 1.0)) + (1 - x) * torch.log(
        torch.clamp(1.0 - p, 1e-9, 1.0)
    )
    return logprob.sum(dim=list(range(1, x.dim())))


def log_logistic_cdf_prob(x, mu, log_scale):
    binsize = 1 / 255.0
    scale = torch.exp(log_scale)
    sample = (torch.floor(x / binsize) * binsize - mu) / scale
    logprob = torch.log(1 - torch.exp(-binsize / scale))
    logprob -= F.softplus(sample)
    logprob -= F.softplus(-sample - binsize / scale)
    return logprob.sum(dim=list(range(1, x.dim())))


def logsumexp(x):
    x_max = torch.max(x, dim=0, keepdim=True)[0]
    x_ = x - x_max
    return torch.log(
        torch.clamp(torch.sum(torch.exp(x_), dim=0), min=1e-20)
    ) + x_max.squeeze(0)


def encoding(enc_mlp, fea, y, K, use_mean=False, fix_samples=False, seed=0):
    mu_qz, log_sig_qz = enc_mlp(fea, y)

    if use_mean:
        z = mu_qz
    elif fix_samples:
        mu_qz, log_sig_qz, z = sample_gaussian_fix_randomness(
            mu_qz, log_sig_qz, K, seed
        )
    else:
        mu_qz, log_sig_qz, z = sample_gaussian(mu_qz, log_sig_qz, K)

    logq = log_gaussian_prob(z, mu_qz, log_sig_qz)
    return z, logq


def lowerbound_A(
    x,
    fea,
    y,
    enc_mlp,
    dec,
    ll,
    K=1,
    IS=False,
    use_mean=False,
    fix_samples=False,
    seed=0,
    z=None,
    beta=1.0,
):
    if use_mean:
        K = 1
        fix_samples = False

    if z is None:
        z, logq = encoding(enc_mlp, fea, y, K, use_mean, fix_samples, seed)
    else:
        mu_qz, log_sig_qz = enc_mlp(fea, y)
        logq = log_gaussian_prob(z, mu_qz, log_sig_qz)

    x_rep = x.repeat(K, *([1] * (x.dim() - 1)))
    y_rep = y.repeat(K, 1)

    pyz, pxzy = dec
    y_logit = pyz(z)
    log_pyz = -F.cross_entropy(y_logit, y_rep.argmax(dim=1), reduction="none")
    log_prior_z = log_gaussian_prob(z, 0.0, torch.tensor(0.0))

    mu_x = pxzy(z, y_rep)
    if ll == "bernoulli":
        logp = log_bernoulli_prob(x_rep, mu_x)
    elif ll == "l2":
        logp = -torch.sum((x_rep - mu_x) ** 2, dim=list(range(1, x.dim())))
    elif ll == "l1":
        logp = -torch.sum(torch.abs(x_rep - mu_x), dim=list(range(1, x.dim())))
    elif ll == "gaussian":
        mu, log_sig = mu_x
        logp = log_gaussian_prob(x_rep, mu, log_sig)

    bound = logp * beta + log_pyz + (log_prior_z - logq)
    if IS and K > 1:
        bound = logsumexp(bound.view(K, -1)) - np.log(K)

    return bound


def lowerbound_B(
    x,
    fea,
    y,
    enc_mlp,
    dec,
    ll,
    K=1,
    IS=False,
    use_mean=False,
    fix_samples=False,
    seed=0,
    z=None,
    beta=1.0,
):
    if use_mean:
        K = 1
        fix_samples = False

    if z is None:
        z, logq = encoding(enc_mlp, fea, y, K, use_mean, fix_samples, seed)
    else:
        mu_qz, log_sig_qz = enc_mlp(fea, y)
        logq = log_gaussian_prob(z, mu_qz, log_sig_qz)

    x_rep = x.repeat(K, *([1] * (x.dim() - 1)))
    y_rep = y.repeat(K, 1)

    pzy, pxzy = dec
    mu_pz, log_sig_pz = pzy(y)
    mu_pz = mu_pz.repeat(K, 1)
    log_sig_pz = log_sig_pz.repeat(K, 1)
    log_prior = log_gaussian_prob(z, mu_pz, log_sig_pz)
    log_py = torch.log(torch.tensor(0.1))

    mu_x = pxzy(z, y_rep)
    if ll == "bernoulli":
        logp = log_bernoulli_prob(x_rep, mu_x)
    elif ll == "l2":
        logp = -torch.sum((x_rep - mu_x) ** 2, dim=list(range(1, x.dim())))
    elif ll == "l1":
        logp = -torch.sum(torch.abs(x_rep - mu_x), dim=list(range(1, x.dim())))
    elif ll == "gaussian":
        mu, log_sig = mu_x
        logp = log_gaussian_prob(x_rep, mu, log_sig)
    elif ll == "logistic_cdf":
        mu, log_scale = mu_x
        logp = log_logistic_cdf_prob(x_rep, mu, log_scale)

    bound = logp * beta + log_py + (log_prior - logq)
    if IS and K > 1:
        bound = logsumexp(bound.view(K, -1)) - np.log(K)

    return bound


def lowerbound_C(
    x,
    fea,
    y,
    enc_mlp,
    dec,
    ll,
    K=1,
    IS=False,
    use_mean=False,
    fix_samples=False,
    seed=0,
    z=None,
    beta=1.0,
):
    if use_mean:
        K = 1
        fix_samples = False

    if z is None:
        z, logq = encoding(enc_mlp, fea, y, K, use_mean, fix_samples, seed)
    else:
        mu_qz, log_sig_qz = enc_mlp(fea, y)
        logq = log_gaussian_prob(z, mu_qz, log_sig_qz)

    x_rep = x.repeat(K, *([1] * (x.dim() - 1)))
    y_rep = y.repeat(K, 1)

    log_prior_z = log_gaussian_prob(z, 0.0, torch.tensor(0.0))

    pyzx, pxz = dec
    mu_x = pxz(z)
    if ll == "bernoulli":
        logp = log_bernoulli_prob(x_rep, mu_x)
    elif ll == "l2":
        logp = -torch.sum((x_rep - mu_x) ** 2, dim=list(range(1, x.dim())))
    elif ll == "l1":
        logp = -torch.sum(torch.abs(x_rep - mu_x), dim=list(range(1, x.dim())))

    logit_y = pyzx(z, x_rep)
    log_pyzx = -F.cross_entropy(logit_y, y_rep.argmax(dim=1), reduction="none")

    bound = logp * beta + log_pyzx + (log_prior_z - logq)
    if IS and K > 1:
        bound = logsumexp(bound.view(K, -1)) - np.log(K)

    return bound


def lowerbound_D(
    x,
    fea,
    y,
    enc_mlp,
    dec,
    ll,
    K=1,
    IS=False,
    use_mean=False,
    fix_samples=False,
    seed=0,
    z=None,
    beta=1.0,
):
    if use_mean:
        K = 1
        fix_samples = False

    if z is None:
        z, logq = encoding(enc_mlp, fea, y, K, use_mean, fix_samples, seed)
    else:
        mu_qz, log_sig_qz = enc_mlp(fea, y)
        logq = log_gaussian_prob(z, mu_qz, log_sig_qz)

    x_rep = x.repeat(K, *([1] * (x.dim() - 1)))
    y_rep = y.repeat(K, 1)

    pyzx, pzx = dec
    mu_pz, log_sig_pz = pzx(x)
    mu_pz = mu_pz.repeat(K, 1)
    log_sig_pz = log_sig_pz.repeat(K, 1)
    log_pzx = log_gaussian_prob(z, mu_pz, log_sig_pz)

    logit_y = pyzx(z, x_rep)
    log_pyzx = -F.cross_entropy(logit_y, y_rep.argmax(dim=1), reduction="none")

    bound = log_pyzx + beta * (log_pzx - logq)
    if IS and K > 1:
        bound = logsumexp(bound.view(K, -1)) - np.log(K)

    return bound


def lowerbound_E(
    x,
    fea,
    y,
    enc_mlp,
    dec,
    ll,
    K=1,
    IS=False,
    use_mean=False,
    fix_samples=False,
    seed=0,
    z=None,
    beta=1.0,
):
    if use_mean:
        K = 1
        fix_samples = False

    if z is None:
        z, logq = encoding(enc_mlp, fea, y, K, use_mean, fix_samples, seed)
    else:
        mu_qz, log_sig_qz = enc_mlp(fea, y)
        logq = log_gaussian_prob(z, mu_qz, log_sig_qz)

    x_rep = x.repeat(K, *([1] * (x.dim() - 1)))
    y_rep = y.repeat(K, 1)

    pyz, pzx = dec
    mu_pz, log_sig_pz = pzx(x)
    mu_pz = mu_pz.repeat(K, 1)
    log_sig_pz = log_sig_pz.repeat(K, 1)
    log_pzx = log_gaussian_prob(z, mu_pz, log_sig_pz)

    logit_y = pyz(z)
    log_pyz = -F.cross_entropy(logit_y, y_rep.argmax(dim=1), reduction="none")

    bound = log_pzx + log_pyz - beta * logq
    if IS and K > 1:
        bound = logsumexp(bound.view(K, -1)) - np.log(K)

    return bound


def lowerbound_F(
    x,
    fea,
    y,
    enc_mlp,
    dec,
    ll,
    K=1,
    IS=False,
    use_mean=False,
    fix_samples=False,
    seed=0,
    z=None,
    beta=1.0,
):
    if use_mean:
        K = 1
        fix_samples = False

    if z is None:
        z, logq = encoding(enc_mlp, fea, y, K, use_mean, fix_samples, seed)
    else:
        mu_qz, log_sig_qz = enc_mlp(fea, y)
        logq = log_gaussian_prob(z, mu_qz, log_sig_qz)

    x_rep = x.repeat(K, *([1] * (x.dim() - 1)))
    y_rep = y.repeat(K, 1)

    log_prior_z = log_gaussian_prob(z, 0.0, torch.tensor(0.0))

    pyz, pxz = dec
    mu_x = pxz(z)
    if ll == "bernoulli":
        logp = log_bernoulli_prob(x_rep, mu_x)
    elif ll == "l2":
        logp = -torch.sum((x_rep - mu_x) ** 2, dim=list(range(1, x.dim())))
    elif ll == "l1":
        logp = -torch.sum(torch.abs(x_rep - mu_x), dim=list(range(1, x.dim())))
    elif ll == "gaussian":
        mu, log_sig = mu_x
        logp = log_gaussian_prob(x_rep, mu, log_sig)

    logit_y = pyz(z)
    log_pyz = -F.cross_entropy(logit_y, y_rep.argmax(dim=1), reduction="none")

    bound = logp * beta + log_pyz + (log_prior_z - logq)
    if IS and K > 1:
        bound = logsumexp(bound.view(K, -1)) - np.log(K)

    return bound


def lowerbound_G(
    x,
    fea,
    y,
    enc_mlp,
    dec,
    ll,
    K=1,
    IS=False,
    use_mean=False,
    fix_samples=False,
    seed=0,
    z=None,
    beta=1.0,
):
    if use_mean:
        K = 1
        fix_samples = False

    if z is None:
        z, logq = encoding(enc_mlp, fea, y, K, use_mean, fix_samples, seed)
    else:
        mu_qz, log_sig_qz = enc_mlp(fea, y)
        logq = log_gaussian_prob(z, mu_qz, log_sig_qz)

    x_rep = x.repeat(K, *([1] * (x.dim() - 1)))
    y_rep = y.repeat(K, 1)

    pzy, pxz = dec
    mu_pz, log_sig_pz = pzy(y)
    mu_pz = mu_pz.repeat(K, 1)
    log_sig_pz = log_sig_pz.repeat(K, 1)
    log_prior = log_gaussian_prob(z, mu_pz, log_sig_pz)
    log_py = torch.log(torch.tensor(0.1))

    mu_x = pxz(z)
    if ll == "bernoulli":
        logp = log_bernoulli_prob(x_rep, mu_x)
    elif ll == "l2":
        logp = -torch.sum((x_rep - mu_x) ** 2, dim=list(range(1, x.dim())))
    elif ll == "l1":
        logp = -torch.sum(torch.abs(x_rep - mu_x), dim=list(range(1, x.dim())))
    elif ll == "gaussian":
        mu, log_sig = mu_x
        logp = log_gaussian_prob(x_rep, mu, log_sig)
    elif ll == "logistic_cdf":
        mu, log_scale = mu_x
        logp = log_logistic_cdf_prob(x_rep, mu, log_scale)

    bound = logp * beta + log_py + (log_prior - logq)
    if IS and K > 1:
        bound = logsumexp(bound.view(K, -1)) - np.log(K)

    return bound


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from importlib import import_module

    import numpy as np
    from torch.utils.data import DataLoader

    from models.conv_encoder_mnist import GaussianConvEncoder
    from utils.mnist import CustomMNISTDataset

    seed = 29
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_size = 4
    input_shape = (1, 28, 28)
    dimH = 256
    dimZ = 64
    dimY = 10
    n_channel = 32
    ll = "bernoulli"
    K = 5
    IS = True
    beta = 1.0

    train_dataset = CustomMNISTDataset(path="./data", train=True, conv=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    x_batch, y_batch = next(iter(train_loader))
    y_one_hot = F.one_hot(y_batch, num_classes=dimY).float()

    encoder = GaussianConvEncoder(
        input_shape=input_shape,
        dimH=dimH,
        dimZ=dimZ,
        dimY=dimY,
        n_channel=n_channel,
        name="conv_encoder",
    )
    fea = encoder.encoder_conv.apply_conv(x_batch)

    letters = ["A", "B", "C", "D", "E", "F", "G"]

    from lowerbound_functions import (
        lowerbound_A,
        lowerbound_B,
        lowerbound_C,
        lowerbound_D,
        lowerbound_E,
        lowerbound_F,
        lowerbound_G,
    )

    lowerbound_funcs = {
        "A": lowerbound_A,
        "B": lowerbound_B,
        "C": lowerbound_C,
        "D": lowerbound_D,
        "E": lowerbound_E,
        "F": lowerbound_F,
        "G": lowerbound_G,
    }

    bounds = {}
    for letter in letters:
        module_name = f"models.conv_generator_mnist_{letter}"
        module = import_module(module_name.replace("/", "."))
        GeneratorClass = getattr(module, "Generator")
        generator = GeneratorClass(
            input_shape=input_shape,
            dimH=dimH,
            dimZ=dimZ,
            dimY=dimY,
            n_channel=n_channel,
            last_activation="sigmoid",
            name=f"generator_{letter}",
        )
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
        lowerbound = lowerbound_funcs[letter]
        bound = lowerbound(
            x=x_batch,
            fea=fea,
            y=y_one_hot,
            enc_mlp=encoder.enc_mlp,
            dec=dec,
            ll=ll,
            K=K,
            IS=IS,
            use_mean=False,
            fix_samples=False,
            seed=seed,
            z=None,
            beta=beta,
        )
        bounds[letter] = bound
        print(f"Lower bound {letter} shape:", bound.shape)
        print(f"Lower bound {letter} values:", bound)
