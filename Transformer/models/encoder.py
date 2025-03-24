import torch, math
from torch import nn
from .attention import MultiHeadAttention
from .model import AddNorm, PositionWiseFFN, PositionalEncoding

class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, X, valid_lens):
        raise NotImplementedError

class TransformerEncoder(Encoder):
    """The Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        super(TransformerEncoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

class TransformerEncoderBlock(nn.Module):
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 user_bias=False):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, user_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

if __name__ == '__main__':
    from d2l import torch as d2l
    X = torch.ones((2, 100, 24))
    valid_lens = torch.tensor([3, 2])
    encoder_blk = TransformerEncoderBlock(24, 48, 8, 0.5)
    encoder_blk.eval()
    d2l.check_shape(encoder_blk(X, valid_lens), X.shape)

    encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
    d2l.check_shape(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens),
                    (2, 100, 24))

