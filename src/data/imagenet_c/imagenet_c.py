import numpy as np
from pathlib import Path
from typing import Any, Callable, Tuple
from torchvision.datasets import ImageFolder
__all__=['ImageNetC']


class ImageNetC(ImageFolder):
    def __init__(self,
                 root: str,
                 corruption: str,
                 split: str = 'train',
                 transform: Callable[..., Any] | None = None,
                 target_transform: Callable[..., Any] | None = None):
        super().__init__(Path(root)/corruption/split, transform, target_transform)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]  # (str, int)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, np.array(target, dtype=np.float32)[np.newaxis]
