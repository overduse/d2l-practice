import torch
import torch.nn as nn
from model_resnet import ResNet18, model_path, init_cnn, device
from dataset import train_dataloader, test_dataloader, training_data
from loop import train, test

from time import time
from pathlib import Path

"""
File: /d2l-practise/modern-cnn/ResNet/train.py

"""

def main():


    lr = 1e-2
    epochs = 5
    print(f"epochs: {epochs}")
    parent_dir = Path(model_path).parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    init_input = next(iter(training_data))[0].unsqueeze(0).to(device) # dummy input
    num_classes = 10

    model = ResNet18(num_classes).to(device).eval()
    model.apply_init(init_input, init_cnn)

    model.train()


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # setting log file
    log_path = parent_dir / "resnet18.log"
    with open(log_path, 'w') as log_file:
        log_file.write("Epoch, Test Accuracy, Test Loss, Training Time (s))\n")

    total_start_time = time()

    for t in range(epochs):
        epoch_start_time = time()


        print(f"Epoch {t+1}\n-----------------------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        training_time = time() - epoch_start_time

        acc, test_loss = test(test_dataloader, model, loss_fn, device)
        with open(log_path, 'a') as log_file:
            log_file.write(f"{t+1}, {acc:>.4f}, {test_loss:>.8f}, {training_time:.2f}")


    total_training_time = time() - total_start_time
    print(f"Total Training Time: {total_training_time:.2f} seconds")
    with open(log_path, 'a') as log_file:
        log_file.write(f"Total Training Time: {total_training_time:.2f} seconds")

    print("TRAINING DONE!")

    torch.save(model.state_dict(), model_path)
    print("Saved PyTorch Model State to ", model_path)


if __name__ == '__main__':
    main()

