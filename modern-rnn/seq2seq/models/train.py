import torch
from utils.utils import to, gpu, num_gpus

class Trainer():
    """The base class for training models with data."""

    def __init__(self, max_epochs, gpus=0, gradient_clip_val=0):
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.gpus = [gpu(i) for i in range(min(gpus, num_gpus()))]

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        print(f"Train Start!\n------------------------------------")
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
        print(f"Train finish!")

    def fit_epoch(self):
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            # print(f"train_batch_idx {self.train_batch_idx}, finished")
            self.train_batch_idx += 1

            if self.train_batch_idx % 100 == 0:
                print(f"epoch: {self.epoch + 1:3d}/{self.max_epochs}, ppl: {torch.exp(loss):>3f}")

        if self.val_dataloader is None:
            return self.model.eval()

        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)
        # print(self.num_train_batches)

    def prepare_batch(self, batch):
        if self.gpus:
            batch = [to(a, self.gpus[0]) for a in batch]
        return batch

    def prepare_model(self, model):
        model.trainer = self
        # model.board.xlim = [0, self.max_epochs]
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model


    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm
