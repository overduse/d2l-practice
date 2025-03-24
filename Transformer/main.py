from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder
from models.train import Trainer
from models.seq2seq import Seq2Seq

from data.dataset_FraEng import MTFraEng

from d2l import torch as d2l

if __name__ == '__main__':
    data = MTFraEng(batch_size=256)
    num_hiddens, num_blks, dropout = 128, 2, 0.2
    ffn_num_hiddens, num_heads = 64, 4

    encoder = TransformerEncoder(
        len(data.src_vocab), num_hiddens, ffn_num_hiddens, num_heads,
        num_blks, dropout)
    decoder = TransformerDecoder(
        len(data.tgt_vocab), num_hiddens, ffn_num_hiddens, num_heads,
        num_blks, dropout)

    model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.001)
    trainer = Trainer(max_epochs=100)
    trainer.fit(model, data)

    model.to('cuda')
    model.eval()

    engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    preds, _ = model.predict_step(
        data.build(engs, fras), d2l.try_gpu(), data.num_steps)
    for en, fr, p in zip(engs, fras, preds):
        translation = []
        for token in data.tgt_vocab.to_tokens(p):
            if token == '<eos>':
                break
            translation.append(token)
        print(f'{en} => {translation}, bleu,'
              f'{d2l.bleu(" ".join(translation), fr, k=2):.3f}')
