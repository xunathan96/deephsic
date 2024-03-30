from typing import Any, Callable
from pathlib import Path
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from imagenet_c import corrupt

__all__ = ['ImageNet', 'ImageNetSplit']

class ImageNet(ImageFolder):
    def __init__(self,
                 root: str,
                 split: str = 'train',
                 transform: Callable[..., Any] | None = None,
                 target_transform: Callable[..., Any] | None = None):
        super().__init__(Path(root)/split, transform, target_transform)


class ImageNetSplit(ImageFolder):
    def __init__(self,
                 root: str,
                 split: str = 'train',
                 transform: Callable[..., Any] | None = None,
                 target_transform: Callable[..., Any] | None = None):
        super().__init__(Path(root), transform, target_transform)

        print(self.imgs)

        if split == 'train':
            ...



def main():
    imagenet = ImageNet(root='data/cls-loc', split='val')
    print(len(imagenet))
    print(imagenet[11100])

if __name__=='__main__':
    main()

