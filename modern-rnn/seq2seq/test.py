import torch
from models.enc_dec import Seq2SeqEncoder, Seq2SeqDecoder

from utils.utils import check_shape


if __name__ == '__main__':
    vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
    batch_size, num_steps = 4, 9
    encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
    X = torch.zeros((batch_size, num_steps))
    enc_outputs, enc_state = encoder(X)

    check_shape(enc_outputs, (num_steps, batch_size, num_hiddens))
    check_shape(enc_outputs, (num_steps, batch_size, num_hiddens))

    decoder = Seq2SeqDecoder(vocab_size, embed_size, num_hiddens, num_layers)
    state = decoder.init_state(encoder(X))
    dec_outputs, state = decoder(X, state)
    check_shape(dec_outputs, (batch_size, num_steps, vocab_size))
    check_shape(state[1], (num_layers, batch_size, num_hiddens))


