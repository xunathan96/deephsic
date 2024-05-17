import numpy as np
from pathlib import Path
from torchvision.datasets import CIFAR10
from PIL import Image
from typing import Any, Callable, Tuple


class CIFAR10H(CIFAR10):

    def __init__(self,
                 root: str,
                 split: str = 'train',
                 train_val_test_split: str = '7:1:2',
                 size: int = None,
                 transform: Callable[..., Any] | None = None,
                 target_transform: Callable[..., Any] | None = None,
                 download: bool = False) -> None:
        super().__init__(root, False, transform, target_transform, download)
        self.soft_labels = np.load(Path(root)/"cifar10h-probs.npy")
        self.soft_labels = self.soft_labels.astype(np.float32)  # (10000, 10)
        self.targets = np.array(self.targets)
        n_samples = len(self.soft_labels)

        # compute train-val-test splits
        size = min(size, n_samples) if size is not None else n_samples
        splits = list(map(int, train_val_test_split.split(':')))
        TRAIN_SPLIT = splits[0]/sum(splits)
        VAL_SPLIT = splits[1]/sum(splits)
        TEST_SPLIT = splits[2]/sum(splits)

        # use consistent shuffle for data
        idx = np.arange(n_samples)
        np.random.seed(2024)
        np.random.shuffle(idx)
        self.data = self.data[idx]
        self.targets = self.targets[idx]
        self.soft_labels = self.soft_labels[idx]

        # split data
        n_splits = np.cumsum([int(split*size) for split in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)])
        if split == 'train':
            self.data = self.data[:n_splits[0]]
            self.targets = self.targets[:n_splits[0]]
            self.soft_labels = self.soft_labels[:n_splits[0]]
        elif split == 'val':
            self.data = self.data[n_splits[0]:n_splits[1]]
            self.targets = self.targets[n_splits[0]:n_splits[1]]
            self.soft_labels = self.soft_labels[n_splits[0]:n_splits[1]]
        elif split == 'test':
            self.data = self.data[n_splits[1]:n_splits[2]]
            self.targets = self.targets[n_splits[1]:n_splits[2]]
            self.soft_labels = self.soft_labels[n_splits[1]:n_splits[2]]


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, softlabel) where target is index of the target class, softlabel is the vector of class probs
        """
        img, target = self.data[index], self.soft_labels[index] # self.targets[index]
        target_id = self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target[[target_id]]





def main():
    cifar10h = CIFAR10H(root='./raw',
                        split='test',
                        # size=5000,
                        train_val_test_split='7:1:2',
                        download=True)

if __name__ == '__main__':
    main()
