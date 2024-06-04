from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
from typing import TypedDict
import numpy as np
import torch
import torch.nn as nn
import wandb
import inspect
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer

from config.config import Config
import utils.utils as utils

EARLY_STOP = 400    # interval after which apply early stopping
SAVE_INTERVAL = 1   # interval after which the model is saved


class BaseTrainer(ABC):

    def __init__(self, config: Config):
        self.cfg = config
        self.device = torch.device(config['device'])
        self.wandb = ('wandb' in self.cfg)

        # TODO: handle loading pretrained/checkpoint
        self._setup_wandb()
        self._setup_model()
        self._setup_dataloaders()
        self._setup_optimizers()

        self.is_train = ('train' in self.dataset) and ('train' in self.dataloader)
        self.is_validate = ('val' in self.dataset) and ('val' in self.dataloader)
        self.is_test = ('test' in self.dataset) and ('test' in self.dataloader)

    @classmethod
    def build(cls, config: Config):
        return cls(config)

    def _setup_model(self):
        self.model: nn.Module = self.cfg['model'].build()
        self.model.to(self.device)
        if self.wandb:
            wandb.watch(self.model)

    def _setup_dataloaders_depreciated(self):
        self.dataset = {'train': None, 'val': None, 'test': None}
        self.dataloader = {'train': None, 'val': None, 'test': None}
        if 'dataset' not in self.cfg: return
        self.dataset = dict()
        for mode in ['train', 'val', 'test']:
            self.dataset[mode] = None
            if (mode in self.cfg['dataset']) and (self.cfg['dataset'][mode] is not None):
                dataset = self.cfg['dataset'][mode]
                self.dataset[mode] = dataset.build()
        if 'dataloader' not in self.cfg: return
        self.dataloader = dict()
        for mode in ['train', 'val', 'test']:
            self.dataloader[mode] = None
            if (mode in self.cfg['dataloader']) and (self.cfg['dataloader'][mode] is not None) and (self.dataset[mode] is not None):
                loader = self.cfg['dataloader'][mode]
                self.dataloader[mode] = loader.build(dataset=self.dataset[mode])

    def _setup_dataloaders(self):
        self.dataset: DatasetDict = dict()
        self.dataloader: DataloaderDict = dict()
        # build datasets
        if 'dataset' not in self.cfg: return
        if 'train' in self.cfg['dataset']:
            self.dataset['train']: Dataset = self.cfg['dataset']['train'].build() # type: ignore
        if 'val' in self.cfg['dataset']:
            self.dataset['val']: Dataset = self.cfg['dataset']['val'].build() # type: ignore
        if 'test' in self.cfg['dataset']:
            self.dataset['test']: Dataset = self.cfg['dataset']['test'].build() # type: ignore
        # build dataloaders
        if 'dataloader' not in self.cfg: return
        if ('train' in self.cfg['dataloader']) and ('train' in self.dataset):
            self.dataloader['train']: DataLoader = self.cfg['dataloader']['train'].build(dataset=self.dataset['train']) # type: ignore
        if ('val' in self.cfg['dataloader']) and ('val' in self.dataset) and len(self.dataset['val']) > 0:
            self.dataloader['val']: DataLoader = self.cfg['dataloader']['val'].build(dataset=self.dataset['val']) # type: ignore
        if ('test' in self.cfg['dataloader']) and ('test' in self.dataset):
            self.dataloader['test']: DataLoader = self.cfg['dataloader']['test'].build(dataset=self.dataset['test']) # type: ignore


    def _setup_optimizers(self):
        self.optimizer = self.scheduler = self.criterion = None
        if 'optimizer' in self.cfg:
            self.optimizer: Optimizer = self.cfg['optimizer'].build(params=self.model.parameters())
        if 'scheduler' in self.cfg:
            self.scheduler: LRScheduler = self.cfg['scheduler'].build(optimizer=self.optimizer)
        if 'criterion' in self.cfg:
            self.criterion: nn.Module = self.cfg['criterion'].build()
            self.criterion = self.criterion.to(self.device)


    def _setup_wandb(self):
        if self.wandb:
            sig = list(inspect.signature(wandb.init).parameters)
            kwds = {k:v for k,v in self.cfg['wandb'].items() if k in sig}   # filter out extraneous args
            kwds['config'] = self._wandb_config()
            wandb.init(**kwds)

    def _wandb_config(self):
        return {
            'n-epochs': self.cfg['n_epochs'],
            'batch-size': self.cfg['dataloader']['train']['batch_size'],
            'lr': self.cfg['optimizer']['lr'],
            'architecture': self.cfg['model']['name'],
            'dataset': self.cfg['dataset']['name'],
        }


    def backprop(self, loss: torch.Tensor, optimizer: Optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    @abstractmethod
    def train_one_epoch(self, epoch: int, *args, **kwds):
        # train model for one epoch based on the training set.
        ...

    @abstractmethod
    def validation(self, epoch: int, *args, **kwds):
        # get loss based on the validation set.
        ...

    @abstractmethod
    def inference(self, *args, **kwds):
        # generate predicted samples based on the test set. 
        ...

    def infer(self, input: torch.Tensor, label=None):
        # run inference on a single input batch
        self.model.eval()
        return self.model(input)


    def train(self, epochs=0, *args, **kwds):
        if not self.is_train:
            raise Exception(f"Training error: no train data specified.")
        self.best_loss = np.inf
        self.best_epoch = -1
        stop = False
        for epoch in (pbar:=tqdm(range(epochs),
                                 bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                                 dynamic_ncols=True,
                                 position=0)):

            loss_train = self.train_one_epoch(epoch, *args, **kwds)
            if self.is_validate:
                pbar.set_description(f"Training: [best-epoch: {self.best_epoch+1}, best-loss: {self.best_loss:.3e}]")
                loss_val = self.validation(epoch, *args, **kwds)
                stop = self._early_stopping(epoch, loss_val)
                if stop: break
            else:
                pbar.set_description(f"Training: [epoch: {epoch+1}, loss: {loss_train:.3e}]")
                if (epoch+1) % SAVE_INTERVAL == 0:
                    # fp = Path(self.cfg['save_dir'])/f"epoch_{epoch+1}.pt"
                    fp = Path(self.cfg['save_dir'])/'best.pt'
                    self.save_checkpoint(fp, epoch, loss_train)

            if self.scheduler is not None:
                self.scheduler.step()
            if self.wandb:  # TODO: set a log interval
                wandb.log({
                    'epoch': epoch+1,
                    'loss_train': loss_train,
                    'loss_val': loss_val,
                }, step=epoch+1)

        if stop:
            print(f"Early Stopping: no improvement in validation error over the last {EARLY_STOP} epochs.")
            return False
        return True


    def eval(self, *args, **kwds):
        # run inference on the test set and return the computed metrics dictionary
        if not self.is_test:
            raise Exception(f"Evaluation error: no test data specified.")
        samples = self.inference(*args, **kwds)
        stats = self.compute_metrics(samples)
        return stats


    @abstractmethod
    def compute_metrics(self, samples):
        r"""returns a dictionary of metrics based on the given predictions and labels"""
        ...


    def _early_stopping(self, epoch, loss):
        stop = False
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            fp = Path(self.cfg['save_dir'])/'best.pt'
            self.save_checkpoint(fp, epoch, loss)
        elif epoch - self.best_epoch > EARLY_STOP:
            stop = True
        return stop


    def save_checkpoint(self, filepath: str | Path, epoch: int, loss: float):
        return utils.save_checkpoint(filepath,
                                     epoch,
                                     loss,
                                     self.model,
                                     self.optimizer,
                                     self.scheduler)

    def load_checkpoint(self, filepath: str | Path):
        return utils.load_checkpoint(filepath,
                                     self.model,
                                     self.optimizer,
                                     self.scheduler,
                                     self.device)


    def tsne(self):
        ...





# ==============================
#       Typing Structures
# ==============================

class DatasetDict(TypedDict):
    train: Dataset
    val: Dataset
    test: Dataset

class DataloaderDict(TypedDict):
    train: DataLoader
    val: DataLoader
    test: DataLoader

