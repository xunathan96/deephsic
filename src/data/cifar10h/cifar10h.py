import numpy as np
from pathlib import Path
from torchvision.datasets import CIFAR10
from PIL import Image
from typing import Any, Callable, Tuple

TRAIN_SPLIT = 7/10
VAL_SPLIT = 1/10
TEST_SPLIT = 2/10

class CIFAR10H(CIFAR10):

    def __init__(self,
                 root: str,
                 split: float = 'train',
                 transform: Callable[..., Any] | None = None,
                 target_transform: Callable[..., Any] | None = None,
                 download: bool = False) -> None:
        super().__init__(root, False, transform, target_transform, download)
        self.soft_labels = np.load(Path(root)/"cifar10h-probs.npy")
        self.soft_labels = self.soft_labels.astype(np.float32)  # (10000, 10)
        self.targets = np.array(self.targets)
        n_samples = len(self.soft_labels)

        n_train = int(TRAIN_SPLIT*n_samples)
        n_val = int(VAL_SPLIT*n_samples)
        n_test = n_samples - n_train - n_val

        # use consistent shuffle for data
        idx = np.arange(n_samples)
        np.random.seed(2024)
        np.random.shuffle(idx)
        self.data = self.data[idx]
        self.targets = self.targets[idx]
        self.soft_labels = self.soft_labels[idx]

        if split=='train':
            self.data = self.data[:n_train]
            self.targets = self.targets[:n_train]
            self.soft_labels = self.soft_labels[:n_train]
        elif split=='val':
            self.data = self.data[n_train:n_train + n_val]
            self.targets = self.targets[n_train:n_train + n_val]
            self.soft_labels = self.soft_labels[n_train:n_train + n_val]
        elif split=='test':
            self.data = self.data[n_train + n_val:]
            self.targets = self.targets[n_train + n_val:]
            self.soft_labels = self.soft_labels[n_train + n_val:]

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
    cifar10h = CIFAR10H(root='./raw', split='test', download=True)


if __name__ == '__main__':
    main()
