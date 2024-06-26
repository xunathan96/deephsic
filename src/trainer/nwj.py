import numpy as np
import torch
from tqdm import tqdm
from .base import BaseTrainer
import metrics

RUNNING_INTERVAL = 24

class NWJTrainer(BaseTrainer):

    def train_one_epoch(self, epoch: int):
        self.model.train()
        losses = list()
        running_loss = 0
        for i, batch in enumerate(pbar:=tqdm(self.dataloader['train'],
                                             bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                                             dynamic_ncols=True,
                                             leave=False)):
            X = batch[0].to(self.device)    # (N, *, Dx)
            Y = batch[1].to(self.device)    # (N, *, Dy)

            loss = self.criterion(self.model, X, Y)
            self.backprop(loss, self.optimizer)

            losses.append(loss.item())
            running_loss += loss.item()
            if (i+1)%RUNNING_INTERVAL==0:
                pbar.set_description(f"[{epoch+1}, {i+1:4d}]    loss: {running_loss/RUNNING_INTERVAL:.4f}")
                running_loss = 0

        return sum(losses)/len(losses)


    @torch.no_grad
    def validation(self, epoch: int):
        self.model.eval()
        losses = list()
        running_loss = 0
        for i, batch in enumerate(pbar:=tqdm(self.dataloader['val'],
                                             bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                                             dynamic_ncols=True,
                                             leave=False)):
            X = batch[0].to(self.device)    # (B,Dx)
            Y = batch[1].to(self.device)    # (B,Dy)
            loss = self.criterion(self.model, X, Y)

            losses.append(loss.item())
            running_loss += loss.item()
            if (i+1)%RUNNING_INTERVAL==0:
                pbar.set_description(f"[{epoch+1}, {i+1:4d}]    loss: {running_loss/RUNNING_INTERVAL:.2e}")
                running_loss = 0

        return sum(losses)/len(losses)


    @torch.no_grad
    def inference(self,
                  n_tests: int = 100,
                  n_permutations: int = 500,):
        self.model.eval()
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

            X = batch[0].to(self.device)    # (N, *, Dx)
            Y = batch[1].to(self.device)    # (N, *, Dy)
            nwj, p_value = metrics.nwj.permutation_test(self.model,
                                                        X, Y,
                                                        n_permutations=n_permutations)
            samples.append((nwj, p_value))
            pbar.set_description(f"[{i+1}/{n_tests}] NWJ: {nwj}, p-value: {p_value:.4f}")
        return samples


    def compute_metrics(self,
                        samples: list,
                        significance: float = 0.05):
        stat, p_values = zip(*samples)
        stat = np.array(stat)
        p_values = np.array(p_values)
        stats = dict()
        stats['NWJ'] = stat.mean()
        stats['p-value'] = p_values.mean()
        stats['power'] = (p_values<significance).mean()
        return stats


    def eval(self,
             n_samples = None,
             n_tests = 100,
             n_permutations = 500,
             **kwds):
        # run inference on the test set and return the computed metrics dictionary
        if not self.is_test:
            raise Exception(f"Evaluation error: no test data specified.")
        if n_samples is not None:
            self.dataloader['test'] = self.cfg['dataloader']['test'].build(
                dataset=self.dataset['test'],
                batch_size=n_samples)
        samples = self.inference(n_tests, n_permutations)
        stats = self.compute_metrics(samples, significance=0.05)
        return stats


    def type1_error(self,
                    n_samples = None,
                    n_tests: int = 100,
                    n_permutations: int = 500,
                    significance: float = 0.05):
        if not self.is_test:
            raise Exception(f"Evaluation error: no test data specified.")
        if n_samples is not None:
            self.dataloader['test'] = self.cfg['dataloader']['test'].build(
                dataset=self.dataset['test'],
                batch_size=n_samples)

        self.model.eval()
        n_reject = 0
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

            X = batch[0].to(self.device)    # (N, *, Dx)
            Y = batch[1].to(self.device)    # (N, *, Dy)
            Y_shuffle = Y[torch.randperm(Y.shape[0], device=self.device)]   # null distribution
            acc, p_value = metrics.nwj.nwj(self.model,
                                           X, Y_shuffle,
                                           n_permutations = n_permutations)
            if p_value < significance:
                n_reject += 1
            pbar.set_description(f"[{i+1}/{n_tests}] stat: {acc}, p-value: {p_value:.4f}")
        return {'type1-error': n_reject / n_tests}




# ==============================
#       HELPER FUNCTIONS
# ==============================

