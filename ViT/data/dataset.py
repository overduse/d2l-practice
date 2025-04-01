import torchvision
from torchvision import datasets
import torchvision.transforms.v2 as v2

import torch
from torch.utils.data import DataLoader

from d2l import torch as d2l


class FashionMNIST:
    """The Fashion-MNIST dataset.

    Defined in :numref:`sec_fashion_mnist`"""
    def __init__(self, batch_size=64, resize=(28, 28), num_workers=4):
        super().__init__()
        self.root = "../data"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._trans = v2.Compose([v2.ToImage(), v2.ToDtype(torch.uint8, scale=True),
                            v2.Resize(resize), v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=self._trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=self._trans, download=True)

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                           num_workers=self.num_workers)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)

    def text_labels(self, indices):
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]


if __name__ == '__main__':
    ...
