from data.dataset_mt import TimeMachine
from models.model import LSTMScratch, RNNLMScratch, LSTM
from models.train import Trainer
from utils.utils import try_gpu


if __name__ == '__main__':

    data = TimeMachine(batch_size=1024, num_steps=35)

    # lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=32)
    lstm = LSTM(num_inputs=len(data.vocab), num_hiddens=32)

    model = RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=0.1)
    trainer = Trainer(max_epochs=50, gpus=1, gradient_clip_val=1)
    trainer.fit(model, data)

    print(model.predict("it has ", 20, data.vocab, try_gpu()))
    print(model.predict("traveller", 20, data.vocab, try_gpu()))
