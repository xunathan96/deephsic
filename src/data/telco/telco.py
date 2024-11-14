import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# https://www.kaggle.com/datasets/blastchar/telco-customer-churn


class Telco(Dataset):

    # tenure, MonthlyCharges, TotalCharges, Churn
    X_VAR = ['tenure']
    Y_VAR = ['MonthlyCharges']
    
    def __init__(self,
                 root: str,
                 split: str = 'train',
                 train_val_test_split: str = '7:1:2',):
        super().__init__()
        self.raw_df = self.preprocess(pd.read_csv(root, delimiter=','))
        self.data = self.raw_df[self.X_VAR + self.Y_VAR]

        # compute train-val-test splits
        size = 7043
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
        # normalize tenure/monthlycharges to (0,1]
        def normalize(df: pd.DataFrame):
            return (df - df.mean()) / df.std()
        df['tenure'] = normalize(df['tenure'])
        df['MonthlyCharges'] = normalize(df['MonthlyCharges'])
        # convert churn no/yes to 0/1
        df['Churn'] = df['Churn'].map({'Yes': 1.0, 'No': 0.0})
        df['gender'] = df['gender'].map({'Male': 1.0, 'Female': 0.0})
        df['PhoneService'] = df['PhoneService'].map({'Yes': 1.0, 'No': 0.0})
        df['StreamingTV'] = df['StreamingTV'].map({'Yes': 1.0, 'No': 0.0, 'No internet service': -1.0})
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
    dataset = Telco(root='./raw/churn.csv',
                    split='train',
                    train_val_test_split='7:0:3')

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    batch = next(iter(dataloader))
    X,Y = batch
    print(X)
    print(Y)


if __name__=='__main__':
    main()
