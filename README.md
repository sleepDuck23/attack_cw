# DeepBayesTorch

Pytorch implementation of the article
[Are Generative Classifiers More Robust to Adversarial Attacks?](https://arxiv.org/abs/1802.06552).

## Install

Create a Python3.12.2 virtual environment

```bash
pyenv virtualenv 3.12.2 bayes
pyenv activate byes
```

Install the dependencies using `poetry`

```bash
pip install poetry
poetry install
```

## Usage

### Train a generative classifier

#### On MNIST

Example of training classifier "A" on MNIST from scratch

```bash
python train_vae.py --data_name mnist A -1
```

Once trained, you can run

```bash
python train_vae.py --data_name mnist A 0
```

to resume training using saved checkpoint 0.

#### On CIFAR10

Example of training classifier "A" on CIFAR10 from scratch

```bash
python train_vae.py --data_name cifar10 A -1
```

Once trained, you can run

```bash
python train_vae.py --data_name mnist A 0
```

to resume training using saved checkpoint 0.

#### On GTSRB

Example of training classifier "A" on GTSRB from scratch

```bash
python train_vae.py --data_name gtsrb A -1
```

Once trained, you can run

```bash
python train_vae.py --data_name gtsrb A 0
```

to resume training using saved checkpoint 0.

### Perform $l_\infty$ attacks on generative classifiers

Make sure you saved all models for a given dataset under this directory structure
(example for mnist):

```bash
save/
├── mnist_conv_vae_A_64
│   └── checkpoint_0.pth
├── mnist_conv_vae_B_64
│   └── checkpoint_0.pth
├── mnist_conv_vae_C_64
│   └── checkpoint_0.pth
├── mnist_conv_vae_D_64
│   └── checkpoint_0.pth
├── mnist_conv_vae_E_64
│   └── checkpoint_0.pth
├── mnist_conv_vae_F_64
│   └── checkpoint_0.pth
└── mnist_conv_vae_G_64
    └── checkpoint_0.pth
```

Then you can launch the experiment for $l_\infty$ attacks with different epsilon
values, with

```bash
python attack_infty.py --data_name mnist --compute --batch_size 100 --save_dir mnist_results --json_file mnist_infty.json
```

This will compute accuracies for each attack, each model, and each epsilon. You
can then plot the results with

```bash
python attack_infty.py --data_name mnist --plot --save_dir mnist_results --json_file mnist_infty.json
```
