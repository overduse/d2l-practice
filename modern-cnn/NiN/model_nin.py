import torch
import torch.nn as nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model_path = "../../checkpoints/NiN/model.pth"

def init_cnn(module):
    """Initial weights for CNNs."""

    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_normal_(module.weight)


def nin_block(out_channels, kernel_size, strides, paddding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, paddding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
    )


class NiN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nin_block(96, kernel_size=11, strides=4, paddding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, kernel_size=7, strides=1, paddding=2),
            nn.MaxPool2d(3, stride=2),
            nin_block(384, kernel_size=3, strides=1, paddding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            nin_block(num_classes, kernel_size=3, strides=1, paddding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def layer_summary(self, X_shape):
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)

    def apply_init(self, input, init=None):
        """initial function."""

        self.forward(input)
        if init is not None:
            self.net.apply(init)

    def forward(self, X):
        return self.net(X)


if __name__ == '__main__':
    num_classes = 10
    model = NiN(num_classes)
    model.layer_summary((1, 3, 224, 224))
    print(model)

