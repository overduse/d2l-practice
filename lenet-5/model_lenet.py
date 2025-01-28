import torch
import torch.nn as nn


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model_path = "../checkpoints/lenet-5/model.pth"

def init_cnn(module):
    """Initial weights for CNNs."""

    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_normal_(module.weight)


class LeNet(nn.Module):
    """LeNet-5 model."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(num_classes)
        )


    def forward(self, X):
        return self.net(X)

    def apply_init(self, inputs, init=None):
        """initial function."""
        self.forward(inputs)
        if init is not None:
            self.net.apply(init)

    # def apply_init(self, init=None):
    #     """Apply initialization function."""
    #     if init is not None:
    #         self.net.apply(init)


    def layer_summary(self, X_shape):
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)


if __name__ == '__main__':
    """test model structure."""
    from dataset import training_data

    model = LeNet()
    init_input = next(iter(training_data))[0].unsqueeze(0)
    model.apply_init(init_input, init_cnn)
    model.layer_summary((1, 1, 28, 28))
    print("", model, sep="\n")
