import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from typing import TypedDict
from pathlib import Path

import metrics
import utils.utils as utils
from .base import BaseTrainer
from kernel import BaseKernel

EARLY_STOP = 400        # interval after which apply early stopping
SAVE_INTERVAL = 100     # interval after which the model is saved
RUNNING_PER_EPOCH = 1   # number of running statistics computed per epoch


class HSICBaseTrainer(BaseTrainer):

    def _setup_model(self):
        self.tied = True
        self.model: KernelDict = dict()
        self.model['k']: BaseKernel = self.cfg['model']['k'].build()    # type: ignore
        self.model['l']: BaseKernel = self.model['k']                   # type: ignore
        if self.cfg['model']['tied'] == False:
            self.tied = False
            self.model['l'] = self.cfg['model']['l'].build()
        self.model['k'].to(self.device)
        self.model['l'].to(self.device)
        if self.wandb:
            wandb.watch((self.model['k'], self.model['l']))

    def _setup_optimizers(self):
        self.optimizer = self.scheduler = self.criterion = None
        if 'optimizer' in self.cfg:
            params = set(self.model['k'].parameters()) | set(self.model['l'].parameters())   # NOTE: be careful here
            self.optimizer: Optimizer = self.cfg['optimizer'].build(params=params)
        if 'scheduler' in self.cfg:
            self.scheduler: LRScheduler = self.cfg['scheduler'].build(optimizer=self.optimizer)
        if 'criterion' in self.cfg:
            self.criterion: nn.Module = self.cfg['criterion'].build()
            self.criterion = self.criterion.to(self.device)

    def save_checkpoint(self, filepath: str | Path, epoch: int, loss: float):
        return utils.save_checkpoint(filepath,
                                     epoch,
                                     loss,
                                     model=self.model['k'] if self.tied else (self.model['k'], self.model['l']),
                                     optimizer=self.optimizer,
                                     scheduler=self.scheduler)

    def load_checkpoint(self, filepath: str | Path):
        return utils.load_checkpoint(filepath,
                                     model=self.model['k'] if self.tied else (self.model['k'], self.model['l']),
                                     optimizer=self.optimizer,
                                     scheduler=self.scheduler,
                                     device=self.device)

    def _wandb_config(self):
        return None


class HSICTrainer(HSICBaseTrainer):

    def train_one_epoch(self, epoch: int):
        self.model['k'].train()
        self.model['l'].train()
        losses = list()
        running_loss = 0
        RUNNING_INTERVAL = len(self.dataloader['train'])//RUNNING_PER_EPOCH
        for i, batch in enumerate(pbar:=tqdm(self.dataloader['train'],
                                             bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                                             dynamic_ncols=True,
                                             leave=False)):
            X = batch[0].to(self.device)    # (B,Dx)
            Y = batch[1].to(self.device)    # (B,Dy)

            loss = self.criterion(self.model['k'], self.model['l'], X, Y)
            self.backprop(loss, self.optimizer)

            losses.append(loss.item())
            running_loss += loss.item()
            if (i+1)%RUNNING_INTERVAL==0:
                pbar.set_description(f"[{epoch+1}, {i+1:4d}]    loss: {running_loss/RUNNING_INTERVAL:.2e}")
                running_loss = 0

        return sum(losses)/len(losses)


    @torch.no_grad
    def validation(self, epoch: int):
        self.model['k'].eval()
        self.model['l'].eval()
        losses = list()
        running_loss = 0
        RUNNING_INTERVAL = len(self.dataloader['val'])//RUNNING_PER_EPOCH
        for i, batch in enumerate(pbar:=tqdm(self.dataloader['val'],
                                             bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                                             dynamic_ncols=True,
                                             leave=False)):
            X = batch[0].to(self.device)    # (B,Dx)
            Y = batch[1].to(self.device)    # (B,Dy)
            loss = self.criterion(self.model['k'], self.model['l'], X, Y)

            losses.append(loss.item())
            running_loss += loss.item()
            if (i+1)%RUNNING_INTERVAL==0:
                pbar.set_description(f"[{epoch+1}, {i+1:4d}]    loss: {running_loss/RUNNING_INTERVAL:.2e}")
                running_loss = 0

        return sum(losses)/len(losses)


    @torch.no_grad
    def inference(self,
                  n_tests: int = 100,
                  n_permutations: int = 500):
        self.model['k'].eval()
        self.model['l'].eval()
        samples = list()
        test_iter = iter(self.dataloader['test'])
        for i in (pbar:=tqdm(range(n_tests),
                             bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                             dynamic_ncols=True,
                             leave=False)):
            try:
                batch = next(test_iter)
            except StopIteration:
                test_iter = iter(self.dataloader['test'])
                batch = next(test_iter)

            X = batch[0].to(self.device)    # (B,Dx)
            Y = batch[1].to(self.device)    # (B,Dy)
            hsic, var, p_value, r = metrics.hsic.permutation_test(self.model['k'], self.model['l'],
                                                                  X, Y,
                                                                  compute_var=False,
                                                                  n_permutations=n_permutations)
            samples.append((hsic, var, p_value, r))
            pbar.set_description(f"[{i+1}/{n_tests}] hsic: {hsic}, p-value: {p_value:.4f}")
        return samples


    def compute_metrics(self,
                        samples: list,
                        significance: float = 0.05):
        hsic_arr, var_arr, p_value_arr, thresh_arr = zip(*samples)
        hsic_arr = np.array(hsic_arr)
        var_arr = np.array(var_arr)
        p_value_arr = np.array(p_value_arr)
        thresh_arr = np.array(thresh_arr)
        stats = dict()
        stats['hsic'] = hsic_arr.mean()
        stats['var'] = var_arr.mean() if None not in var_arr else None
        stats['p-value'] = p_value_arr.mean()
        stats['thresh'] = thresh_arr.mean()
        stats['power'] = (p_value_arr < significance).mean()
        return stats


    def eval(self, n_samples=None):
        # run inference on the test set and return the computed metrics dictionary
        if not self.is_test:
            raise Exception(f"Evaluation error: no test data specified.")
        if n_samples is not None:
            self.dataloader['test'] = self.cfg['dataloader']['test'].build(
                dataset=self.dataset['test'],
                batch_size=n_samples)
        samples = self.inference(n_tests=100, n_permutations=500)
        stats = self.compute_metrics(samples, significance=0.05)
        return stats



    # ==============================
    #         DEBUGGING
    # ==============================

    def weighted_power_grid(self):
        import numpy as np
        from kernel import WeightedGaussian

        scale_k = [1]*5
        scale_l = [1]*5
        grid = np.empty((7,7))
        weights = [0, 0.1, 0.5, 1., 2., 5., 10.]
        with tqdm(total=7*7,
                  bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  dynamic_ncols=True) as pbar:
            for i, wk in enumerate(weights):
                for j, wl in enumerate(weights):
                    pbar.set_description(f"empirical power @ {i},{j}")
                    scale_k[0] = wk
                    scale_l[0] = wl
                    k = WeightedGaussian(scale_k, device=self.device)
                    l = WeightedGaussian(scale_l, device=self.device)
                    #hsic, var = metrics.hsic.hsic(k, l, X, Y, compute_var=True)
                    #power_approx = (hsic/(var.sqrt()+1.e-4)).item()
                    empirical_power,_ = self.empirical_power(k,l,n_tests=100)
                    grid[i][j] = empirical_power
                    pbar.update(1)

        with np.printoptions(precision=5, suppress=True):
            print(grid)












# ==============================
#       HELPER FUNCTIONS
# ==============================

def marginals(joint: torch.Tensor):
    # split joint samples into marginals X,Y
    dim = joint.shape[-1]
    mask = torch.zeros(dim, dtype=torch.bool, device=joint.device)
    mask[dim//2+1:] = True
    mask[1] = True
    return joint[:,~mask], joint[:,mask]

class KernelDict(TypedDict):
    k: BaseKernel
    l: BaseKernel

