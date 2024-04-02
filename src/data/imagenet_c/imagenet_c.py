from pathlib import Path
from typing import Any, Callable
from torchvision.datasets import ImageFolder
__all__=['ImageNetC']


class ImageNetC(ImageFolder):
    def __init__(self,
                 root: str,
                 split: str = 'train',
                 transform: Callable[..., Any] | None = None,
                 target_transform: Callable[..., Any] | None = None):
        super().__init__(Path(root)/'data'/split, transform, target_transform)


