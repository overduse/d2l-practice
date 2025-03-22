import torch
from models.model import MultiHeadAttention
from utils.utils import check_shape


if __name__ == '__main__':
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
    batch_size, num_queries, num_kvpairs = 2, 4, 6
    valid_lens = torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    check_shape(attention(X, Y, Y, valid_lens),
                    (batch_size, num_queries, num_hiddens))
