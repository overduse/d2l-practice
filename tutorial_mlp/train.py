import torch
from torch import nn

from loop import train, test
from model_mlp import NeuralNetwork, device, model_path
from dataset import train_dataloader, test_dataloader

from pathlib import Path


def main():

    model = NeuralNetwork().to(device)

    # Loss func & optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5

    parent_dir = Path(model_path).parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-----------------------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("TRAINING DONE!")

    torch.save(model.state_dict(), model_path)
    print("Saved PyTorch Model State to ", model_path)


if __name__ == '__main__':
    main()
