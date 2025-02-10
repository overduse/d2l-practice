import torch
import torch.nn as nn
from model_nin import NiN, model_path, init_cnn, device
from dataset import train_dataloader, test_dataloader, training_data
from loop import train, test

from time import time
from pathlib import Path

"""
File: /d2l-practise/modern-cnn/NiN/train.py

"""

def main():


    lr = 1e-1
    epochs = 20
    print(f"epochs: {epochs}")
    parent_dir = Path(model_path).parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    init_input = next(iter(training_data))[0].unsqueeze(0).to(device) # dummy input
    num_classes = 10

    model = NiN(num_classes).to(device)
    model.apply_init(init_input, init_cnn)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # setting log file
    log_path = parent_dir / "nin.log"
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
            log_file.write(f"{t+1}, {acc:>.4f}, {test_loss:>.8f}, {training_time:.2f}\n")


    total_training_time = time() - total_start_time
    print(f"Total Training Time: {total_training_time:.2f} seconds\n")
    with open(log_path, 'a') as log_file:
        log_file.write(f"Total Training Time: {total_training_time:.2f} seconds\n")

    print("TRAINING DONE!")

    torch.save(model.state_dict(), model_path)
    print("Saved PyTorch Model State to ", model_path)


if __name__ == '__main__':
    main()

