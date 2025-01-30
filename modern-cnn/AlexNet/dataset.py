from torchvision import datasets
import torchvision.transforms.v2 as v2

import torch
from torch.utils.data import DataLoader

"""
file: d2l-practice/modern-cnn/AlexNet/dataset.py

"""

#TODO:data argument.
train_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomCrop(224, padding=4),
    v2.RandomHorizontalFlip(),
    # v2.RandomRotation(10),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    # v2.ToTensor(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),

    v2.Normalize((0.5,), (0.5,)),
])

test_transform = v2.Compose([
    v2.Resize((224, 224)),
    # v2.ToTensor(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,)),
])

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="../../data/",
    train=True,
    download=False,
    transform=train_transform,
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="../../data/",
    train=False,
    download=False,
    transform=test_transform,
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

