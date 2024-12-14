import torch
from torchvision import datasets, transforms


class CustomMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, path="./data", train=True, digits=None, conv=False):
        """
        Args:
            train (bool): If True, load training data; else load test data.
            digits (list, optional): List of digits to include. If None, all digits are included.
            conv (bool): If True, retain 2D shape for convolutional models. If False, flatten for fully connected models.
        """
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                (
                    transforms.Lambda(lambda x: x)
                    if conv
                    else transforms.Lambda(lambda x: x.view(-1))
                ),  # Reshape if required
            ]
        )

        # Load MNIST dataset
        self.dataset = datasets.MNIST(
            root=path, train=train, download=True, transform=self.transform
        )

        if digits is not None:
            self.label_mapping = {digit: idx for idx, digit in enumerate(digits)}

            self.indices = [
                i for i, (_, label) in enumerate(self.dataset) if label in digits
            ]
        else:
            self.label_mapping = None
            self.indices = range(len(self.dataset))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img, label = self.dataset[actual_idx]
        if self.label_mapping:
            label = self.label_mapping[label]
        return img, label


if __name__ == "__main__":
    mnist_train = CustomMNISTDataset(
        path="./data", train=True, digits=[1, 7], conv=True
    )
    mnist_test = CustomMNISTDataset(
        path="./data", train=False, digits=[1, 7], conv=True
    )
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break
