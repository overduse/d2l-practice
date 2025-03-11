import collections
import re
import torch

from d2l import torch as d2l

class TimeMachine:
    """The Time Machine dataset."""
    def __init__(self, batch_size=64, num_steps=10, num_train=10000, num_val=5000):
        self.root = '../data/'
        self.batch_size = batch_size
        self.num_train, self.num_val = num_train, num_val

        self._corpus, self.vocab = self.build(self._download())
        array = torch.tensor([self._corpus[i:i+num_steps+1]
                            for i in range(len(self._corpus)-num_steps)])
        self.X, self.Y = array[:,:-1], array[:,1:]

    def build(self, raw_text=None, vocab=None):
        if raw_text is None: raw_text=self._download()
        tokens = self._tokenize(self._preprocess(raw_text))
        if vocab is None: vocab = Vocab(tokens)
        assert tokens is not None, "type of tokens mustn't to be None"
        corpus = [vocab[token] for token in tokens]
        return corpus, vocab

    def _download(self) -> str:
        fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', self.root,
                              '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

    def _preprocess(self, text):
        return re.sub('[^A-Za-z]+', ' ', text).lower()

    def _tokenize(self, line, token='char'):
        '''split text into words or char'''
        # if token == 'word':
        #     return [line.split() for line in lines]
        # elif token == 'char':
        #     return list(line) for line in lines]
        if token == 'char':
            return list(line)
        else:
            print('Error: unknown token type: ', token)

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_dataloader(self, train):
        idx = slice(0, self.num_train) if train else slice(
            self.num_train, self.num_train + self.num_val)
        return self.get_tensorloader([self.X, self.Y], train, idx)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)


class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                        reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq ])))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']


if __name__ == '__main__':

    data = TimeMachine(batch_size=2)

    for X, Y in data.train_dataloader():
        print('X:', X)
        print('Y:', Y)
        break
