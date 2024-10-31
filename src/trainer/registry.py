from .base import BaseTrainer
from .classifier import Classifier
from .hsic import HSICTrainer
from .pathwise import Pathwise
from .c2st import C2STTrainer
from .mmd import MMDTrainer
from .infonce import InfoNCETrainer
from .nwj import NWJTrainer
from .mi import MITrainer

class TrainerRegistry:
    
    def __init__(self):
        self.registry = dict()

    def __getitem__(self, key):
        return self.registry[key]

    def register(self, key, trainer):
        self.registry[key] = trainer

    def create(self, key, **kwds):
        trainer = self.registry.get(key)
        if not trainer:
            raise ValueError(key)
        return trainer(**kwds)


trainerRegistry = TrainerRegistry()
trainerRegistry.register('hsic', HSICTrainer)
trainerRegistry.register('c2st', C2STTrainer)
trainerRegistry.register('mmd', MMDTrainer)
trainerRegistry.register('infonce', InfoNCETrainer)
trainerRegistry.register('nwj', NWJTrainer)
trainerRegistry.register('mi', MITrainer)


def get(key) -> BaseTrainer:
    return trainerRegistry[key]






