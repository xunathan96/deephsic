import random
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.datasets import PennTreebank as pennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

def yield_tokens_from_context(context):
    for tokens in context:
        yield tokens


class PennTreebank(Dataset):

    def __init__(self,
                 root: str = './data/penn_treebank',
                 size: int = None,
                 split: str = 'train',
                 train_val_test_split: str = '7:1:2',
                 shuffle: bool = False):
        super().__init__()

        self.data_iter = pennTreebank(root=root, split='train')
        self.data = self.parse_data()

        # compute train-val-test splits
        n = len(self.data)
        size = min(size, n) if size is not None else n
        splits = [int(m) if m.isdigit() else None for m in train_val_test_split.split(':')]
        # one split is inferred
        if splits.count(None) > 1:
            raise Exception(f"We can only infer a maximum of one split.")
        splits = [size-sum(filter(None, splits)) if m is None else m for m in splits]

        TRAIN_SPLIT = splits[0]/sum(splits)
        VAL_SPLIT = splits[1]/sum(splits)
        TEST_SPLIT = splits[2]/sum(splits)

        # deterministic shuffle of trajectories
        if shuffle:
            rng = random.Random(2024)
            rng.shuffle(self.data)

        # data splits
        split_idx = np.cumsum([int(split*size) for split in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)])
        if split == 'train':
            self.samples = self.data[:split_idx[0]]
        elif split == 'val':
            self.samples = self.data[split_idx[0]:split_idx[1]]
        elif split == 'test':
            self.samples = self.data[split_idx[1]:]     # TODO: ...

        # build vocabulary
        self.vocab = build_vocab_from_iterator(yield_tokens(self.data_iter), specials=["<unk>"])
        # self.vocab = build_vocab_from_iterator(yield_tokens_from_context(self.data[:split_idx[2]]), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.vocab_size = len(self.vocab)
        print('vocab_size:', self.vocab_size)


    def parse_data(self):
        context = list()
        for text in tqdm(list(self.data_iter)):
            tokens = tokenizer(text) #self.text_pipeline(text)
            # sliding window of size 5
            for i in range(4, len(tokens)):
                context.append(tokens[i-4:i+1])
        return context

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        ids =  self.vocab(tokens)
        return torch.tensor(ids[:2] + ids[3:]), torch.tensor([ids[2]])



def main():

    dataset = PennTreebank(root='data/penn_treebank', size=20000, split='train')
    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    data_iter = iter(dataloader)
    batch = next(data_iter)
    context, target = batch
    print('context:', context)
    print('target:', target)



if __name__ == '__main__':
    main()


