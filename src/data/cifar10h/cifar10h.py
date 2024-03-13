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
                 split: str = 'train',
                 transform: Callable[..., Any] | None = None,
                 target_transform: Callable[..., Any] | None = None,
                 download: bool = False) -> None:
        super().__init__(root, False, transform, target_transform, download)
        self.soft_labels = np.load(Path(root)/"cifar10h-probs.npy")
        self.soft_labels = self.soft_labels.astype(np.float32)  # (10000, 10)
        n_samples = len(self.soft_labels)
        n_train = int(TRAIN_SPLIT*n_samples)
        n_val = int(VAL_SPLIT*n_samples)
        n_test = int(TEST_SPLIT*n_samples)
        s_train = n_train
        s_val = s_train + n_val
        s_test = s_val + n_test

        if split=='train':
            self.data = self.data[:s_train]
            self.targets = self.targets[:s_train]
            self.soft_labels = self.soft_labels[:s_train]
        elif split=='val':
            self.data = self.data[s_train:s_val]
            self.targets = self.targets[s_train:s_val]
            self.soft_labels = self.soft_labels[s_train:s_val]
        elif split=='test':
            self.data = self.data[s_val:s_test]
            self.targets = self.targets[s_val:s_test]
            self.soft_labels = self.soft_labels[s_val:s_test]



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
