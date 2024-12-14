import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets


class CustomCIFAR10Dataset(Dataset):
    def __init__(self, path="./data", train=True, labels=None, conv=True, seed=0):
        """
        Args:
            train (bool): Whether to load the training set or test set.
            labels (list, optional): List of labels to filter. If None, all labels are included.
            conv (bool): Whether to retain the 4D shape for convolutional models. If False, the data is flattened.
            seed (int): Random seed for shuffling.
        """
        self.train = train
        self.labels = labels
        self.conv = conv
        self.seed = seed

        # Load CIFAR-10 dataset using torchvision
        cifar10 = datasets.CIFAR10(
            root=path,
            train=train,
            download=True,
            transform=None,  # We'll handle transformation manually
        )

        # Extract data and targets
        self.data = np.array(cifar10.data)  # Shape: [N, H, W, C]
        self.targets = np.array(cifar10.targets)

        # Filter specific labels if provided
        if self.labels is not None:
            self.data, self.targets = self._filter_labels(
                self.data, self.targets, self.labels
            )

        # Normalize data
        self.data = self.data / 255.0  # Scale to [0, 1]

        # Change to PyTorch format [N, C, H, W]
        self.data = self.data.transpose(0, 3, 1, 2)  # From [N, H, W, C] to [N, C, H, W]

        # Flatten data for non-convolutional models
        if not self.conv:
            self.data = self.data.reshape(self.data.shape[0], -1)

    def _filter_labels(self, data, targets, labels):
        filtered_indices = [i for i, label in enumerate(targets) if label in labels]
        filtered_data = data[filtered_indices]
        filtered_targets = [
            labels.index(label) for label in targets[filtered_indices]
        ]  # Re-map to new label space

        # Shuffle the filtered data
        np.random.seed(self.seed)
        indices = np.random.permutation(len(filtered_data))
        return (
            filtered_data[indices],
            np.array(filtered_targets, dtype=np.int32)[indices],
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.targets[idx], dtype=torch.long)
        return img, label


if __name__ == "__main__":
    train_dataset = CustomCIFAR10Dataset(
        path="./data", train=True, labels=[0, 1], conv=True
    )

    # Initialize dataset for testing
    test_dataset = CustomCIFAR10Dataset(
        path="./data", train=False, labels=[0, 1], conv=True
    )

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )

    # Iterate through the data
    for images, labels in train_loader:
        print(images.shape)
        print(labels.shape)
        break
