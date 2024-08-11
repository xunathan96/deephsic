import random
from typing import Any, Callable, Tuple
import numpy as np
from pathlib import Path
import torch.utils
from torchvision.datasets import ImageFolder


# https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data?select=train

class Emotion(ImageFolder):

    def __init__(self,
                 root: str = './archive',
                 size: int = None,
                 split: str = 'train',
                 train_val_test_split: str = '7:1:2',
                 transform: Callable[..., Any] | None = None,
                 target_transform: Callable[..., Any] | None = None):

        super().__init__(Path(root), transform, target_transform)

        # compute train-val-test splits
        n = len(self.samples)
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
        rng.shuffle(self.samples)

        # data splits
        split_idx = np.cumsum([int(split*size) for split in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)])
        if split == 'train':
            self.samples = self.samples[:split_idx[0]]
        elif split == 'val':
            self.samples = self.samples[split_idx[0]:split_idx[1]]
        elif split == 'test':
            self.samples = self.samples[split_idx[1]:]      # NOTE: safe to use split_idx[1]:split_idx[2] if we have size=None


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]  # (str, int)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, np.array(target, dtype=np.float32)[np.newaxis]


def main():
    
    import torchvision.transforms as tf

    dataset = Emotion(root='archive',
                      split='train',
                      train_val_test_split='1:0:0',
                      transform=tf.Compose([
                          tf.Grayscale(),
                          tf.Resize(32),
                          tf.ToTensor(),
                          tf.Normalize(0.5, 0.5)
                      ]))

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print('size:', len(dataset))

    for batch in dataloader:
        img, label = batch
        print('image:', img.shape)
        print('label:', label)
        return 1/0




if __name__=='__main__':
    main()
