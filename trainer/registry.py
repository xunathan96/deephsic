from .base import BaseTrainer
from .classifier import Classifier
from .hsic import HSIC
from .pathwise import Pathwise
from .c2st import C2ST
from .mmd import MMD


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
trainerRegistry.register('Classifier', Classifier)
trainerRegistry.register('HSIC', HSIC)
trainerRegistry.register('Pathwise', Pathwise)
trainerRegistry.register('C2ST', C2ST)
trainerRegistry.register('MMD', MMD)



def get(key) -> BaseTrainer:
    return trainerRegistry[key]






