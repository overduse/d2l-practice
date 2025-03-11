# import torch
from torch import nn
# from torch.nn import functional as F
# from d2l import torch as d2l

from dataset_mt import TimeMachine
from model_scratch import RNNLMScratch
from train import Trainer

class RNN(nn.Module):
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens=64):
        super(RNN, self).__init__()
        # self.save_hyperparameters()
        self.num_inputs, self.num_hiddens = num_inputs, num_hiddens
        self.rnn = nn.RNN(num_inputs, num_hiddens)

    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)

class RNNLM(RNNLMScratch):
    """The RNN-based language model implemented with high-level APIs. """
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)

    def output_layer(self, rnn_outputs):
        return self.linear(rnn_outputs).swapaxes(0, 1)

if __name__ == '__main__':
    data = TimeMachine(batch_size=1024, num_steps=32)
    rnn = RNN(num_inputs=len(data.vocab), num_hiddens=32)
    model = RNNLM(rnn, vocab_size=len(data.vocab), lr=0.1)

    trainer = Trainer(max_epochs=200, gradient_clip_val=1)
    trainer.fit(model, data)

    pred = model.predict("it has ", 20, data.vocab)
    print(pred)
