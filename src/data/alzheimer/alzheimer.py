import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset

class Alzheimer(Dataset):

    categories = ['BMI','Smoking','AlcoholConsumption','PhysicalActivity','DietQuality','SleepQuality',
                  'SystolicBP','DiastolicBP','CholesterolTotal','CholesterolLDL','CholesterolHDL',
                  'CholesterolTriglycerides','MMSE','FunctionalAssessment','ADL']
    # PatientID,Age,Gender,Ethnicity,EducationLevel,BMI,Smoking,AlcoholConsumption,PhysicalActivity,DietQuality,SleepQuality,
    # SystolicBP,DiastolicBP,CholesterolTotal,CholesterolLDL,CholesterolHDL,CholesterolTriglycerides,MMSE,FunctionalAssessment,ADL
    # binary:
    # FamilyHistoryAlzheimers,CardiovascularDisease,Diabetes,Depression,HeadInjury,Hypertension,
    # MemoryComplaints,BehavioralProblems,Confusion,Disorientation,PersonalityChanges,DifficultyCompletingTasks,Forgetfulness,Diagnosis,DoctorInCharge
    X_VAR = ['CholesterolLDL']
    Y_VAR = ['CholesterolHDL']
    # CholesterolLDL vs CholesterolHDL

    def __init__(self,
                 root: str,
                 split: str = 'train',
                 train_val_test_split: str = '7:1:2',):
        super().__init__()
        self.raw_df = self.preprocess(pd.read_csv(root, delimiter=','))
        self.data = self.raw_df[self.X_VAR + self.Y_VAR]

        # deterministic shuffle of trajectories
        self.data = self.data.sample(frac = 1, random_state=20242024)

        # compute train-val-test splits
        size = 2149
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
        def normalize(df: pd.DataFrame):
            return (df - df.mean()) / df.std()
        for key in Alzheimer.categories:
            df[key] = normalize(df[key])
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
    dataset = Alzheimer(root='./raw/alzheimers_disease_data.csv',
                      split='train',
                      train_val_test_split='7:0:3')

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    batch = next(iter(dataloader))
    X,Y = batch
    print(X)
    print(Y)


if __name__=='__main__':
    main()
