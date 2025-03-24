import torch
from torch import nn
from .model import EncoderDecoder
from .encoder import Encoder
from .decoder import Decoder

def init_seq2seq(module):
    """Initialize weights for sequence-to-sequence learning."""
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])

class Seq2Seq(EncoderDecoder):
    """The RNN encoder-decoder for sequence-to-sequence learning."""
    def __init__(self, encoder, decoder, tgt_pad, lr):
        super(Seq2Seq, self).__init__(encoder, decoder)
        self.encoder, self.decoder = encoder, decoder
        self.tgt_pad, self.lr = tgt_pad, lr

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        # self.plot('ppl', torch.exp(l), train=True)
        return l

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        # self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)

    def configure_optimizers(self):
        # Adam optimizer is used here
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def loss(self, Y_hat, Y):
        l = super(Seq2Seq, self).loss(Y_hat, Y, averaged=False)
        mask = (Y.reshape(-1) != self.tgt_pad).type(torch.float32)
        return (l * mask).sum() / mask.sum()

# class Seq2SeqEncoder(Encoder):
#     """The RNN encoder for sequence-to-sequence learning."""
#     def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
#                  dropout:float =0):
#         super(Seq2SeqEncoder, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.rnn = GRU(embed_size, num_hiddens=num_hiddens, num_layers=num_layers, dropout=dropout)
#         self.apply(init_seq2seq)
#
#     def forward(self, X, *args):
#         # X shape: (batch_size, num_steps)
#         embs = self.embedding(X.t().type(torch.int64))
#         # embs shape: (num_steps, batch_size, embed_size)
#         outputs, state = self.rnn(embs)
#         # outputs shape: (num_steps, batch_size, num_hiddens)
#         # state shape: (num_layers, batch_size, num_hiddens)
#         return outputs, state
#
# class Seq2SeqDecoder(Decoder):
#     """The RNN decoder for sequence-to-sequence learning"""
#     def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
#                  dropout:float=0):
#         super(Seq2SeqDecoder, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.rnn = GRU(embed_size+num_hiddens, num_hiddens,
#                           num_layers, dropout)
#         self.dense = nn.LazyLinear(vocab_size)
#         self.apply(init_seq2seq)
#
#     def init_state(self, enc_all_outputs, *args):
#         return enc_all_outputs
#
#     def forward(self, X, state):
#         # X shape: (batch_size, num_steps)
#         # embs shape: (num_steps, batch_size, embed_size)
#         embs = self.embedding(X.t().type(torch.int32))
#         enc_output, hidden_state = state
#         # context shape: (batch_size, num_hiddens)
#         context = enc_output[-1]
#         # Broadcast context to (num_steps, batch_size, num_hiddens)
#         context = context.repeat(embs.shape[0], 1, 1)
#         # Concat at the feature dimension
#         embs_and_context = torch.cat((embs, context), -1)
#         outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
#         outputs = self.dense(outputs).swapaxes(0, 1)
#         # outputs shape: (batch_size, num_steps, vocab_size)
#         # hidden_state shape: (num_layers, batch_size, num_hiddens)
#         return outputs, [enc_output, hidden_state]
#
#
