import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reshape_and_tile_images(
    array, shape=(28, 28), n_cols=None, margin=0, fill_val=None
):
    if isinstance(array, torch.Tensor):
        array = array.numpy()
    if n_cols is None:
        n_cols = int(np.sqrt(array.shape[0]))
    n_rows = int(np.ceil(float(array.shape[0]) / n_cols))
    if len(shape) == 2:
        order = "C"
    else:
        order = "F"

    def cell(i, j):
        ind = i * n_cols + j
        if ind < array.shape[0]:
            image = array[ind].reshape(*shape, order="C")
        else:
            image = np.zeros(shape)
        if margin > 0:
            tmp = np.ones([shape[0], 1]) * fill_val[ind]
            image = np.concatenate([tmp, image, tmp], axis=1)
            tmp = np.ones([1, shape[1] + 2]) * fill_val[ind]
            image = np.concatenate([tmp, image, tmp], axis=0)
        return image

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)


def plot_images(
    images, shape, path, filename, n_rows=10, margin=0, fill_val=None, color=True
):
    if isinstance(images, torch.Tensor):
        images = images.numpy()
    if images.shape[1] == 1:
        images = np.squeeze(images, axis=1)
    elif images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    images = reshape_and_tile_images(images, shape, n_rows, margin, fill_val)
    if color:
        from matplotlib import cm

        plt.imsave(fname=path + filename + ".png", arr=images, cmap=cm.Greys_r)
    else:
        plt.imsave(fname=path + filename + ".png", arr=images, cmap="Greys")
    print("saving image to " + path + filename + ".png")
    plt.close()


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from cifar10 import CustomCIFAR10Dataset
    from mnist import CustomMNISTDataset
    from torch.utils.data import DataLoader

    # Plot MNIST
    mnist_dataset = CustomMNISTDataset(
        path="./data", train=True, digits=None, conv=True
    )
    mnist_loader = DataLoader(mnist_dataset, batch_size=100, shuffle=True)
    mnist_images, _ = next(iter(mnist_loader))
    plot_images(
        mnist_images,
        shape=(28, 28),
        path="./",
        filename="mnist_example",
        n_rows=10,
        color=False,
    )

    # Plot CIFAR-10
    cifar10_dataset = CustomCIFAR10Dataset(
        path="./data", train=True, labels=None, conv=True
    )
    cifar10_loader = DataLoader(cifar10_dataset, batch_size=100, shuffle=True)
    cifar_images, _ = next(iter(cifar10_loader))
    plot_images(
        cifar_images,
        shape=(32, 32, 3),
        path="./",
        filename="cifar10_example",
        n_rows=10,
        color=True,
    )
