from models.model import ViT
from models.train import Trainer
from data.dataset import FashionMNIST

# from d2l import torch as d2l

if __name__ == '__main__':

    img_size, patch_size = 96, 16
    num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
    emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.00005

    model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
                num_blks, emb_dropout, blk_dropout, lr)
    # trainer = Trainer(max_epochs=10)
    trainer = Trainer(max_epochs=20)
    data = FashionMNIST(batch_size=512, resize=(img_size, img_size))
    trainer.fit(model, data)
