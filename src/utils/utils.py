import numpy as np
import torch
import torch.nn as nn
import argparse
import random
import pandas as pd
from pathlib import Path


def seed_all(seed=None, harsh=False):
    if not seed:
        return 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if harsh:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def activation_registry(activation, *args, **kwds):
    return {
        None: nn.Identity,
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'Tanh': nn.Tanh,
        'GLU': nn.GLU,
        'ELU': nn.ELU,
        'Sigmoid': nn.Sigmoid,
        'Softmax': nn.Softmax,
    }[activation](*args, **kwds)


def save_checkpoint(filepath: str,
                    epoch: int,
                    loss: float,
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': loss,
    }
    fp = Path(filepath).with_suffix('.pt')
    fp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, fp)


def load_checkpoint(filepath: str,
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                    device: torch.device = torch.device('cpu')):
    checkpoint = torch.load(Path(filepath), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)    # NOTE: is this necessary?
    if (optimizer is not None) and (checkpoint['optimizer_state_dict'] is not None):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if (scheduler is not None) and (checkpoint['scheduler_state_dict'] is not None):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def save_checkpoint_multi(filepath: str,
                          epoch: int,
                          loss: float,
                          modelList: list[nn.Module],
                          optimizerList: list[torch.optim.Optimizer],
                          schedulerList: list[torch.optim.lr_scheduler.LRScheduler],):
    checkpoint = {
        'epoch': epoch,
        'models_state_dict': [model.state_dict() for model in modelList if model is not None],
        'optimizers_state_dict': [opt.state_dict() for opt in optimizerList if opt is not None],
        'schedulers_state_dict': [schdr.state_dict() for schdr in schedulerList if schdr is not None],
        'loss': loss,
    }
    fp = Path(filepath).with_suffix('.pt')
    fp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, fp)


def load_checkpoint_multi(filepath: str,
                          modelList: list[nn.Module],
                          optimizerList: list[torch.optim.Optimizer] = [],
                          schedulerList: list[torch.optim.lr_scheduler.LRScheduler] = [],
                          device: torch.device = torch.device('cpu')):
    checkpoint = torch.load(Path(filepath), map_location=device)
    for model, state_dict in zip(modelList, checkpoint['models_state_dict']):   # TODO: make sure load is in-place !
        if model is not None: model.load_state_dict(state_dict)
    for optimizer, state_dict in zip(optimizerList, checkpoint['optimizers_state_dict']):
        if optimizer is not None: optimizer.load_state_dict(state_dict)
    for scheduler, state_dict in zip(schedulerList, checkpoint['schedulers_state_dict']):
        if scheduler is not None: scheduler.load_state_dict(state_dict)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss






class Tabular:
    def __init__(self, filepath: str = None):
        self.df = pd.DataFrame()    
        if filepath is None:
            self.fp = None; self.header = None
        else:
            self.fp = Path(filepath)
            if self.fp.suffix == '.csv':
                self.col = pd.read_csv(self.fp, nrows=0).columns if self.fp.exists() else None   # get header (to retain order)
            elif self.fp.suffix == '.xlsx':
                self.col = pd.read_excel(self.fp, nrows=0).columns if self.fp.exists() else None
            else:
                raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{self.df}"

    def read_csv(self, filepath: str = None):
        if filepath is None: filepath = self.fp
        self.df = pd.read_csv(Path(filepath))

    def read_excel(self, filepath: str = None):
        if filepath is None: filepath = self.fp
        self.df = pd.read_excel(Path(filepath))

    def to_csv(self,
               filepath: str = None,
               mode: str = 'a'):
        fp = Path(filepath) if filepath is not None else self.fp
        header = False if (mode == 'a' and fp.exists()) else True
        fp.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(fp, index=False, mode=mode, header=header)

    def to_excel(self,
                 filepath: str = None,
                 mode: str = 'a'):
        fp = Path(filepath) if filepath is not None else self.fp
        header = False if (mode == 'a' and fp.exists()) else True
        fp.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_excel(fp, index=False, mode=mode, header=header)

    def append(self, mapping: dict):
        if self.col is None:
            row = pd.DataFrame(data=mapping, index=[len(self.df.index)])
        else:
            row = pd.DataFrame(data=mapping, columns=self.col, index=[len(self.df.index)])
        self.df = pd.concat([self.df if not self.df.empty else None,    # to prevent warning
                             row])



class NestArgs(argparse._SubParsersAction):
    def __call__(self,
                 parser: argparse.ArgumentParser,
                 namespace: argparse.Namespace,
                 values: str | list[str] | None,
                 option_string: str | None = None) -> None:
        parser_name = values[0]
        arg_strings = values[1:]

        # set the parser name if requested
        if self.dest is not argparse.SUPPRESS:
            setattr(namespace, self.dest, parser_name)

        # select the parser
        try:
            parser = self._name_parser_map[parser_name]
        except KeyError:
            args = {'parser_name': parser_name,
                    'choices': ', '.join(self._name_parser_map)}
            msg = argparse._('unknown parser %(parser_name)r (choices: %(choices)s)') % args
            raise argparse.ArgumentError(self, msg)

        # parse all the remaining options into the namespace
        # store any unrecognized options on the object, so that the top
        # level parser can decide what to do with them

        # In case this subparser defines new defaults, we parse them
        # in a new namespace object and then update the original
        # namespace for the relevant parts.
        subnamespace, arg_strings = parser.parse_known_args(arg_strings, None)
        setattr(namespace, parser_name, subnamespace)   # NOTE: nest subnamespace
        # for key, value in vars(subnamespace).items():
        #     setattr(namespace, key, value)

        if arg_strings:
            vars(namespace[parser_name]).setdefault(argparse._UNRECOGNIZED_ARGS_ATTR, [])
            getattr(namespace[parser_name], argparse._UNRECOGNIZED_ARGS_ATTR).extend(arg_strings)

