import numpy as np
import torch
from tqdm import tqdm
from .base import BaseTrainer
import metrics

RUNNING_PER_EPOCH = 5   # number of running statistics computed per epoch

class MMDTrainer(BaseTrainer):

    def train_one_epoch(self, epoch: int):
        self.model.train()
        losses = list()
        running_loss = 0
        RUNNING_INTERVAL = max(len(self.dataloader['train'])//RUNNING_PER_EPOCH, 1)
        for i, batch in enumerate(pbar:=tqdm(self.dataloader['train'],
                                             bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                                             dynamic_ncols=True,
                                             leave=False)):
            X = batch[0].to(self.device)    # (N,Dx)
            Y = batch[1].to(self.device)    # (N,Dy)
            Z_null, Z_alt = compile_samples(X,Y, test='independence')   # (N, 2D)

            loss = self.criterion(self.model, Z_null, Z_alt)
            self.backprop(loss, self.optimizer)

            losses.append(loss.item())
            running_loss += loss.item()
            if (i+1)%RUNNING_INTERVAL==0:
                pbar.set_description(f"[{epoch+1}, {i+1:4d}]    loss: {running_loss/RUNNING_INTERVAL:.2e}")
                running_loss = 0

        return sum(losses)/len(losses)


    @torch.no_grad
    def validation(self, epoch: int, *args, **kwds):
        self.model.eval()
        losses = list()
        running_loss = 0
        RUNNING_INTERVAL = max(len(self.dataloader['val'])//RUNNING_PER_EPOCH, 1)
        for i, batch in enumerate(pbar:=tqdm(self.dataloader['val'],
                                             bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                                             dynamic_ncols=True,
                                             leave=False)):
            X = batch[0].to(self.device)    # (N,Dx)
            Y = batch[1].to(self.device)    # (N,Dy)
            Z_null, Z_alt = compile_samples(X,Y, test='independence')   # (N, 2D) and (N, 2D)
            loss = self.criterion(self.model, Z_null, Z_alt)

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
                  permutation_test: str = 'independence'):
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

            X = batch[0].to(self.device)    # (N,Dx)
            Y = batch[1].to(self.device)    # (N,Dy)
            mmd2, var, p_value, r = metrics.mmd.permutation_test(self.model,
                                                                 X, Y,
                                                                 compute_var=False,
                                                                 n_permutations=n_permutations,
                                                                 test=permutation_test)

            # NOTE: treating it like a two-sample test causes issues
            # Z_null, Z_alt = compile_samples(X,Y, test='independence')   # (N, 2D)
            # mmd2, var, p_value, r = metrics.mmd.permutation_test(self.model,
            #                                                      Z_null, Z_alt,
            #                                                      compute_var=False,
            #                                                      n_permutations=n_permutations,
            #                                                      test='two-sample')

            samples.append((mmd2, var, p_value, r))
            pbar.set_description(f"[{i+1}/{n_tests}] mmd2: {mmd2}, p-value: {p_value:.4f}")
        return samples


    def compute_metrics(self,
                        samples: list,
                        significance: float = 0.05):
        mmd2_arr, var_arr, p_value_arr, thresh_arr = zip(*samples)
        mmd2_arr = np.array(mmd2_arr)
        var_arr = np.array(var_arr)
        p_value_arr = np.array(p_value_arr)
        thresh_arr = np.array(thresh_arr)
        stats = dict()
        stats['mmd2'] = mmd2_arr.mean()
        stats['var'] = var_arr.mean() if None not in var_arr else None
        stats['p-value'] = p_value_arr.mean()
        stats['thresh'] = thresh_arr.mean()
        stats['power'] = (p_value_arr < significance).mean()
        return stats


    def eval(self,
             n_samples=None,
             n_tests=100,
             n_permutations=500,
             permutation_test='independence'):
        # run inference on the test set and return the computed metrics dictionary
        if not self.is_test:
            raise Exception(f"Evaluation error: no test data specified.")
        if n_samples is not None:
            self.dataloader['test'] = self.cfg['dataloader']['test'].build(
                dataset=self.dataset['test'],
                batch_size=n_samples)
        samples = self.inference(n_tests, n_permutations, permutation_test=permutation_test)
        stats = self.compute_metrics(samples, significance=0.05)
        return stats


    def type1_error(self,
                    n_samples = None,
                    permutation_test: str = 'independence',
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
            mmd2, var, p_value, r = metrics.mmd.permutation_test(self.model,
                                                                 X, Y_shuffle,
                                                                 compute_var=False,
                                                                 n_permutations=n_permutations,
                                                                 test=permutation_test)
            if p_value < significance:
                n_reject += 1
            pbar.set_description(f"[{i+1}/{n_tests}] mmd2: {mmd2}, p-value: {p_value:.4f}")
        return {'type1-error': n_reject / n_tests}



# ==============================
#       HELPER FUNCTIONS
# ==============================

def compile_samples(X, Y, test='two-sample'):
    # prepare the data samples based on the type of hypothesis test
    # X: (N, D)
    # Y: (N, D)
    if (m:=X.shape[0]) != (n:=Y.shape[0]):
        raise Exception(f"Error: expected X and Y to have equal number of samples but got {m} and {n} samples.")
    device = X.device

    if test=='two-sample':
        return X, Y

    elif test=='independence':
        # compile samples from null and alternate hypotheses
        # need to split the data since if we don't then the samples Z_null and Z_alt aren't iid.
        X_null, X_alt = X[:n//2], X[n//2:]
        Y_null, Y_alt = Y[:n//2], Y[n//2:]
        Y_null = Y_null[torch.randperm(n//2, device=device)]
        Z_null = (X_null, Y_null)   # null: Px*Py
        Z_alt = (X_alt, Y_alt)      # alternate: Pxy
        return Z_null, Z_alt

    elif test=='independence_depreciated':
        # compile samples from null and alternate hypotheses
        Y_shuff = Y[torch.randperm(n, device=device)]
        Z_alt = (X,Y)           # alternate: Pxy
        Z_null = (X,Y_shuff)    # null: Px*Py
        return Z_null, Z_alt

    else:
        raise NotImplementedError()

