import torch
from torch import nn
from torch.nn import functional as F

from utils.utils import reshape

class RNNScratch(nn.Module):
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.num_inputs, self.num_hiddens, self.sigma = num_inputs, num_hiddens, sigma

        self.W_xh = nn.Parameter(
            torch.randn(num_inputs, num_hiddens) * self.sigma)
        self.W_hh = nn.Parameter(
            torch.randn(num_hiddens, num_hiddens) * self.sigma)
        self.b_h = nn.Parameter(torch.zeros(num_hiddens))

    def forward(self, inputs, state=None):
        if state is None:
            # Initial state with shape: (batch_size, num_hiddens)
            state = torch.zeros((inputs.shape[1], self.num_hiddens),
                                device=inputs.device)
        else:
            state, = state
        outputs = []
        for X in inputs:    # Shape of inputs: (num_steps, batch_size, num_inputs)
            state = torch.tanh(torch.matmul(X, self.W_xh) +
                               torch.matmul(state, self.W_hh) + self.b_h)
            outputs.append(state)
        return outputs, state

    def init_rnn_state(self, batch_size, num_hiddens, device):
        return (torch.zeros((batch_size, num_hiddens), device= device),)


class RNNLMScratch(nn.Module):
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super(RNNLMScratch, self).__init__()
        self.rnn, self.vocab_size, self.lr = rnn, vocab_size, lr
        self.init_params()

    def init_params(self):
        self.W_hq = nn.Parameter(
            torch.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(torch.zeros(self.vocab_size))

    def one_hot(self, X):
        # Outputs shape: (num_steps, batch_size, vocab_size)
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)

    def output_layer(self, rnn_outputs):
        outputs = [torch.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
        return torch.stack(outputs, 1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state)
        return self.output_layer(rnn_outputs)

    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = reshape(Y, (-1,))
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        # self.plot('ppl', torch.exp(l), train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        # self.plot('ppl', torch.exp(l), train=False)

    def predict(self, prefix, num_preds, vocab, device=None):
        state, outputs = None, [vocab[prefix[0]]]
        for i in range(len(prefix) + num_preds - 1):
            X = torch.tensor([[outputs[-1]]], device=device)
            embs = self.one_hot(X)
            rnn_outputs, state = self.rnn(embs, state)
            if i < len(prefix) - 1: # warm-up period
                outputs.append(vocab[prefix[i + 1]])
            else:    # Predict num_preds steps
                Y = self.output_layer(rnn_outputs)
                outputs.append(int(Y.argmax(dim=2).reshape(1)))
        return ''.join([vocab.idx_to_token[i] for i in outputs])


# check_fns
def check_len(a, n):
    """Check the length of a list"""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'

def check_shape(a, shape):
    """Check the shape of a tensor"""
    assert a.shape == shape, \
        f'tensor\'s shape {a.shape} != expected shape {shape}'


if __name__ == '__main__':

    batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
    rnn = RNNScratch(num_inputs, num_hiddens)
    X = torch.ones((num_steps, batch_size, num_inputs))
    outputs, state = rnn(X)

    # check_fns
    check_len(outputs, num_steps)
    check_shape(outputs[0], (batch_size, num_hiddens))
    check_shape(state, (batch_size, num_hiddens))

    # check forward computation
    model = RNNLMScratch(rnn, num_inputs)
    outputs = model(torch.ones((batch_size, num_steps), dtype=torch.int64))
    check_shape(outputs, (batch_size, num_steps, num_inputs))

