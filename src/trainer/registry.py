from .base import BaseTrainer
from .classifier import Classifier
from .hsic import HSICTrainer
from .pathwise import Pathwise
from .c2st import C2STTrainer
from .mmd import MMDTrainer


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
trainerRegistry.register('HSIC', HSICTrainer)
trainerRegistry.register('Pathwise', Pathwise)
trainerRegistry.register('C2ST', C2STTrainer)
trainerRegistry.register('MMD', MMDTrainer)



def get(key) -> BaseTrainer:
    return trainerRegistry[key]






