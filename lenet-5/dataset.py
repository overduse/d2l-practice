from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader

"""
file: d2l-practice/lenet-5/dataset.py

"""

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="../data/",
    train=True,
    download=False,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="../data/",
    train=False,
    download=False,
    transform=ToTensor(),
)

batch_size = 128

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


if __name__ == '__main__':
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
