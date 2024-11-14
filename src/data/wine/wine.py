import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# https://github.com/cmu-phil/example-causal-datasets/tree/main/real/wine-quality/data


class Wine(Dataset):

    # fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, quality
    X_VAR = ['residual_sugar']
    Y_VAR = ['quality']
    # residual_sugar vs quality (0.03)
    # fixed_acidity vs quality (0.11)

    def __init__(self,
                 root: str,
                 split: str = 'train',
                 train_val_test_split: str = '7:1:2',):
        super().__init__()
        self.raw_df = self.preprocess(pd.read_csv(root, delimiter='\t'))
        self.data = self.raw_df[self.X_VAR + self.Y_VAR]

        # deterministic shuffle of trajectories
        self.data = self.data.sample(frac = 1, random_state=2024)

        # compute train-val-test splits
        size = 6497
        splits = [int(m) if m.isdigit() else None for m in train_val_test_split.split(':')]
        # one split is inferred
        if splits.count(None) > 1:
            raise Exception(f"We can only infer a maximum of one split.")
        splits = [size-sum(filter(None, splits)) if m is None else m for m in splits]

        TRAIN_SPLIT = splits[0]/sum(splits)
        VAL_SPLIT = splits[1]/sum(splits)
        TEST_SPLIT = splits[2]/sum(splits)

        # data splits
        split_idx = np.cumsum([int(split*size) for split in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)])
        if split == 'train':
            self.data = self.data.iloc[:split_idx[0]]
        elif split == 'val':
            self.data = self.data.iloc[split_idx[0]:split_idx[1]]
        elif split == 'test':
            self.data = self.data.iloc[split_idx[1]:]

    @staticmethod
    def preprocess(df: pd.DataFrame):
        df['quality'] = df['quality'].astype(float)
        # normalize tenure/monthlycharges to (0,1]
        def normalize(df: pd.DataFrame):
            return (df - df.mean()) / df.std()
        df['quality'] = normalize(df['quality'])
        df['residual_sugar'] = normalize(df['residual_sugar'])
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        x = sample[self.X_VAR].to_numpy()
        y = sample[self.Y_VAR].to_numpy()
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y







def main():
    dataset = Wine(root='./raw/winequality-red-white.mixed.maximum.2.txt',
                    split='train',
                    train_val_test_split='7:0:3')

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    batch = next(iter(dataloader))
    X,Y = batch
    print(X)
    print(Y)


if __name__=='__main__':
    main()
