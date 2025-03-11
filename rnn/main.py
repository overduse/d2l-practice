from model_scratch import RNNScratch, RNNLMScratch

from dataset_mt import TimeMachine
from train import Trainer
from utils.utils import try_gpu

if __name__ == '__main__':

    data = TimeMachine(batch_size=1024, num_steps=32)
    rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=0.1)
    trainer = Trainer(max_epochs=200, gradient_clip_val=1, gpus=0)

    trainer.fit(model, data)

    model.to(try_gpu())
    print(model.predict('it has', 30, data.vocab, try_gpu()))
    print(model.predict('time machine', 30, data.vocab, try_gpu()))
    print(model.predict('time traveller', 30, data.vocab, try_gpu()))
