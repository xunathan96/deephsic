import numpy as np
import torch
from tqdm import tqdm
from .base import BaseTrainer
import metrics

RUNNING_INTERVAL = 24

class C2STTrainer(BaseTrainer):

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
            X, Y, t = compile_samples(X,Y)  # (2N, *, Dx), (2N, *, Dy), and (2N,)

            logits = self.model(X, Y)       # (2N, 1)
            logits = torch.flatten(logits, start_dim=-2)
            loss = self.criterion(logits, t)
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
            X = batch[0].to(self.device)    # (N, *, Dx)
            Y = batch[1].to(self.device)    # (N, *, Dy)
            X, Y, t = compile_samples(X,Y)  # (2N, *, Dx), (2N, *, Dy), and (2N,)

            logits = self.model(X, Y)       # (2N, 1)
            logits = torch.flatten(logits, start_dim=-2)
            loss = self.criterion(logits, t)

            losses.append(loss.item())
            running_loss += loss.item()
            if (i+1)%RUNNING_INTERVAL==0:
                pbar.set_description(f"[{epoch+1}, {i+1:4d}]    loss: {running_loss/RUNNING_INTERVAL:.2e}")
                running_loss = 0

        return sum(losses)/len(losses)
    

    @torch.no_grad
    def inference(self,
                  n_tests: int = 100,
                  n_permutations: int = 500,
                  significance: float = 0.05,
                  statistic: str = 'logit'):
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
            acc, p_value = metrics.c2st.permutation_test(self.model,
                                                         X, Y,
                                                         statistic=statistic,
                                                         n_permutations=n_permutations)
            samples.append((acc, p_value))
            pbar.set_description(f"[{i+1}/{n_tests}] accuracy: {acc}, p-value: {p_value:.4f}")
        return samples


    def compute_metrics(self,
                        samples: list,
                        significance: float = 0.05):
        accs, p_values = zip(*samples)
        accs = np.array(accs)
        p_values = np.array(p_values)
        stats = dict()
        stats['statistic'] = accs.mean()
        stats['p-value'] = p_values.mean()
        stats['power'] = (p_values<significance).mean()
        return stats


    def eval(self, n_samples=None, statistic='logit'):
        # run inference on the test set and return the computed metrics dictionary
        if not self.is_test:
            raise Exception(f"Evaluation error: no test data specified.")
        if n_samples is not None:
            self.dataloader['test'] = self.cfg['dataloader']['test'].build(
                dataset=self.dataset['test'],
                batch_size=n_samples)
        samples = self.inference(n_tests=100, n_permutations=500, statistic=statistic)
        stats = self.compute_metrics(samples, significance=0.05)
        return stats




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

def compile_samples_depreciated(X, Y, test='independence'):
    # prepare the data samples based on the type of hypothesis test
    if (m:=X.shape[0]) != (n:=Y.shape[0]):
        raise Exception(f"Error: expected X and Y to have equal number of samples but got {m} and {n} samples.")
    device = X.device

    if test=='independence':
        # compile samples from null and alternate hypotheses
        Y_prime = Y[torch.randperm(n, device=device)]
        Z_alt = torch.cat((X,Y), dim=-1)            # alternate: Pxy
        Z_null = torch.cat((X,Y_prime), dim=-1)     # null: Px*Py
        Z = torch.cat((Z_null,Z_alt), dim=0)        # (2N, 2D)
        # create label vector
        t = torch.zeros(2*n, device=device)
        t[n:] = 1
        # shuffle samples
        shuffle_idx = torch.randperm(2*n, device=device)
        Z = Z[shuffle_idx]
        t = t[shuffle_idx]

    elif test=='two-sample':
        raise NotImplementedError()

    return Z, t


def compile_samples(X, Y, test='independence'):
    # prepare the data samples based on the type of hypothesis test
    if (m:=X.shape[0]) != (n:=Y.shape[0]):
        raise Exception(f"Error: expected X and Y to have equal number of samples but got {m} and {n} samples.")
    device = X.device

    if test=='independence':
        # compile samples from null (Px*Py) and alternate (Pxy) hypotheses
        Y_prime = Y[torch.randperm(n, device=device)]
        X_null_alt = torch.cat((X, X), dim=0)        # (2N, *, Dx)
        Y_null_alt = torch.cat((Y_prime, Y), dim=0)  # (2N, *, Dy)
        # create label vector
        t = torch.zeros(2*n, device=device)
        t[n:] = 1
        # shuffle samples
        shuffle_idx = torch.randperm(2*n, device=device)
        X = X_null_alt[shuffle_idx]
        Y = Y_null_alt[shuffle_idx]
        t = t[shuffle_idx]

    elif test=='two-sample':
        raise NotImplementedError()

    return X, Y, t

