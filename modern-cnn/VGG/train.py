import torch
import torch.nn as nn
from model_vgg import VGG11, model_path, init_cnn, device
from dataset import train_dataloader, test_dataloader, training_data
from loop import train, test

from pathlib import Path


def main():


    lr = 1e-2
    epochs = 20
    print(f"epochs: {epochs}")
    parent_dir = Path(model_path).parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    init_input = next(iter(training_data))[0].unsqueeze(0).to(device) # dummy input
    arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    num_classes = 10

    model = VGG11(arch, num_classes).to(device)
    model.apply_init(init_input, init_cnn)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-----------------------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print("TRAINING DONE!")

    torch.save(model.state_dict(), model_path)
    print("Saved PyTorch Model State to ", model_path)


if __name__ == '__main__':
    main()

