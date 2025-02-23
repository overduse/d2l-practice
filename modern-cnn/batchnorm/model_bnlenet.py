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

model_path = "../../checkpoints/bnlenet/model.pth"

def init_cnn(module):
    """Initial weights for CNNs."""

    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_normal_(module.weight)

class BNLeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5), nn.LazyBatchNorm2d(),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(),
            nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(), nn.LazyLinear(120),
            nn.LazyBatchNorm1d(),
            nn.Sigmoid(), nn.LazyLinear(84),
            nn.LazyBatchNorm1d(),
            nn.Sigmoid(), nn.LazyLinear(num_classes)
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
    #INFO: BN Layer expect more than 1 value per channel when training
    model = BNLeNet(num_classes).eval()
    model.layer_summary((1, 1, 224, 224))
    print(model)
