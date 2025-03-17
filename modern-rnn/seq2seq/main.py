from models.train import Trainer

from models.enc_dec import Seq2SeqEncoder, Seq2SeqDecoder, Seq2Seq
from data.dataset_FraEng import MTFraEng

from utils.utils import try_gpu
from utils.metrics import bleu
# from torch.nn import Embedding


if __name__ == '__main__':
    data = MTFraEng(batch_size=128)

    embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
    encoder = Seq2SeqEncoder(
        len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(
        len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                    lr=0.005)
    # model.to("cuda")

    trainer = Trainer(max_epochs=200)
    trainer.fit(model, data)

    engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']


    # for model predict
    model.eval()
    model.to("cuda")

    preds, _ = model.predict_step( data.build(engs, fras), try_gpu(), data.num_steps)
    for en, fr, p in zip(engs, fras, preds):
        translation = []
        for token in data.tgt_vocab.to_tokens(p):
            if token == '<eos>':
                break
            translation.append(token)
        print(f'{en} => {translation}, bleu,'
              f'{bleu(" ".join(translation), fr, k=2):.3f}')
