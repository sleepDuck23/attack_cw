import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class GermanTrafficSignDataset(Dataset):
    def __init__(self, root_dir, train=True, labels=None):
        """
        Args:
            root_dir (str): Directory containing the 'GTSRB' data.
            train (bool): If True, load the training data. Otherwise, load the test data.
            labels (list of int, optional): Specific labels to include. If None, include all labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.train = train
        self.labels = labels
        self.transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]
        )
        self.data = []
        if labels is not None:
            self.label_map = {orig: idx for idx, orig in enumerate(sorted(labels))}
        else:
            self.label_map = None

        if train:
            self._load_training_data()
        else:
            self._load_test_data()

    def _load_training_data(self):
        # Path to the training images
        train_dir = os.path.join(self.root_dir, "GTSRB", "Final_Training", "Images")
        data_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
        # download the data if not present
        if not os.path.exists(train_dir):
            # download the data under the root_dir
            import io
            import zipfile

            import requests

            print("Downloading the data...")
            r = requests.get(data_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(self.root_dir)
            print("Download complete!")

        # Iterate over all class directories
        for class_dir in sorted(os.listdir(train_dir)):
            class_path = os.path.join(train_dir, class_dir)
            if os.path.isdir(class_path):
                class_id = int(class_dir)  # Class ID from folder name

                # Filter by labels if specified
                if self.labels is not None and class_id not in self.labels:
                    continue

                # Load all images in the class directory
                for img_name in os.listdir(class_path):
                    if img_name.endswith(".ppm"):  # Only include image files
                        img_path = os.path.join(class_path, img_name)
                        mapped_id = (
                            self.label_map[class_id] if self.label_map else class_id
                        )
                        self.data.append((img_path, mapped_id))

    def _load_test_data(self):
        # Path to the test images and annotations
        test_dir = os.path.join(self.root_dir, "GTSRB", "Final_Test", "Images")
        test_csv = os.path.join(test_dir, "GT-final_test.csv")
        data_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
        gt_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"
        # download the data if not present
        if not os.path.exists(test_dir):
            # download the data under the root_dir
            import io
            import zipfile

            import requests

            print("Downloading the data...")
            r = requests.get(data_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(self.root_dir)
            r = requests.get(gt_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(self.root_dir)
            os.rename(os.path.join(self.root_dir, "GT-final_test.csv"), test_csv)
            print("Download complete")

        # Load the test annotations
        df = pd.read_csv(test_csv, sep=";")

        for _, row in df.iterrows():
            img_path = os.path.join(test_dir, row["Filename"])
            class_id = row["ClassId"]

            # Filter by labels if specified
            if self.labels is not None and class_id not in self.labels:
                continue

            mapped_id = self.label_map[class_id] if self.label_map else class_id
            self.data.append((img_path, mapped_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        # Open the image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    # Example usage
    root_dir = "./data/"
    dataset = GermanTrafficSignDataset(root_dir=root_dir, train=True, labels=[2, 7])
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for images, labels in dataloader:
        print(images.shape, labels)
        # plot images and labels
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(10, 10))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(np.transpose(images[i], (1, 2, 0)))
            plt.title(f"Class: {labels[i]}")
            plt.axis("off")
        plt.show()
        break
