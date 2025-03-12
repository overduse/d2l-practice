from data.dataset_mt import TimeMachine
from models.model import GRUScratch, RNNLMScratch, GRU
from models.train import Trainer
from utils.utils import try_gpu


if __name__ == '__main__':

    data = TimeMachine(batch_size=1024, num_steps=32)
    # gru = GRUScratch(num_inputs=len(data.vocab), num_hiddens=32)
    gru = GRU(num_inputs=len(data.vocab), num_hiddens=32)
    model = RNNLMScratch(gru, vocab_size=len(data.vocab), lr=0.1)
    trainer = Trainer(max_epochs=500, gpus=1, gradient_clip_val=1)
    trainer.fit(model, data)

    print(model.predict("it has ", 20, data.vocab, try_gpu()))
    print(model.predict("traveller", 20, data.vocab, try_gpu()))
