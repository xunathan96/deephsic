import random
import numpy as np
from torch.utils.data import Dataset
from typing import Any, Callable
from utils import dump, load

# TRAIN_SPLIT = 3000/10000
# VAL_SPLIT = 1000/10000
# TEST_SPLIT = 1 - TRAIN_SPLIT - VAL_SPLIT

class RatInABox(Dataset):

    X_VAR = ['grid']            # ['grid', 'boundary-vector', 'head-direction', 'velocity']
    Y_VAR = ['head_direction']  # ['pos', 'head_direction', 'vel', 'rot_vel']     true label

    def __init__(self,
                 root: str,
                 split: str = 'train',
                 train_val_test_split: str = '7:1:2',   # can be a ratio or (if one split is inferred) absolute sizes 
                 size: int = None,
                 window: str = 'full',  # past, future, present, full
                 transform: Callable[..., Any] | None = None,
                 transform_x: Callable[..., Any] | None = None,
                 transform_y: Callable[..., Any] | None = None,):
        super().__init__()
        self.window = window
        self.transform_x = transform_x or transform
        self.transform_y = transform_y or transform
        trajs = self.preprocess(load(root))
        n = len(trajs)

        # compute train-val-test splits
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
        rng = random.Random(2024)
        rng.shuffle(trajs)

        # data splits
        split_idx = np.cumsum([int(split*size) for split in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)])
        if split == 'train':
            self.samples = trajs[:split_idx[0]]
        elif split == 'val':
            self.samples = trajs[split_idx[0]:split_idx[1]]
        elif split == 'test':
            # self.samples = trajs[split_idx[1]:split_idx[2]]   # TODO: fix this later
            self.samples = trajs[split_idx[1]:]

    def preprocess(self, data):
        samples = []
        frames = window2slice(self.window)
        for traj in data:
            x = np.concatenate([traj[neurons]['firingrate'][frames] for neurons in self.X_VAR], axis=-1)    # (T,20,Dx)
            y = np.concatenate([traj['agent'][feature][frames] for feature in self.Y_VAR], axis=-1)         # (T,20,Dy)
            # y = np.concatenate([traj[neurons]['firingrate'][frames] for neurons in self.Y_VAR], axis=-1)    # (T,20,Dx)
            samples.append((x, y))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        trajx, trajy = self.samples[index]
        if self.transform_x is not None:
            trajx = self.transform_x(trajx)
        if self.transform_y is not None:
            trajy = self.transform_y(trajy)
        return trajx, trajy


def window2slice(window):
    if window == 'full':
        return slice(0, None)
    elif window == 'past':
        return slice(0, 12)
    elif window == 'future':
        return slice(12, None)
    elif window == 'present':
        return 12 #slice(12, 13)
    else:
        raise NotImplementedError()



def main():
    from data.transforms import NumpyToTensor

    riab = RatInABox(root='./raw/riab-10000.pkl',
                     split='test',
                     size=5000,
                     train_val_test_split='7:1:2',
                     transform_x=NumpyToTensor,
                     transform_y=NumpyToTensor)
    print(len(riab))
    # x, y = riab[23]
    # print(x.shape)
    # print(y.shape)


if __name__=='__main__':
    main()
