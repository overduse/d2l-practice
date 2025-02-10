import torch
import torch.nn as nn

from torch.nn import functional as F

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model_path = "../../checkpoints/GoogLeNet/model.pth"

def init_cnn(module):
    """Initial weights for CNNs."""

    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_normal_(module.weight)

class Inception(nn.Module):

    def __init__(self, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, X):
        b1 = F.relu(self.b1_1(X))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(X))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(X))))
        b4 = F.relu(self.b4_2(F.relu(self.b4_1(X))))
        return torch.cat((b1, b2, b3, b4), dim=1)

class GoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            self.b1(), self.b2(), self.b3(), self.b4(),
            self.b5(), nn.LazyLinear(num_classes)
        )

    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def b2(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def b3(self):
        return nn.Sequential(
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def b4(self):
        return nn.Sequential(
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def b5(self):
        return nn.Sequential(
            Inception(256, (160, 320), (32, 128), 128),
            Inception(384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )

    def apply_init(self, input, init=None):
        """initial function."""

        self.forward(input)
        if init is not None:
            self.net.apply(init)

    def layer_summary(self, X_shape):
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)

    def forward(self, X):
        return self.net(X)


if __name__ == '__main__':
    num_classes = 10
    model = GoogLeNet(num_classes)
    model.layer_summary((1, 1, 224, 224))
    print(model)
