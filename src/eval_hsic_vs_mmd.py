import argparse
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from tqdm import tqdm
import torch

from utils import utils
from data.toy import HDGM, Sinusoid
from kernel import Gaussian, GaussianJoint
import metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu',
                        action='store_true',
                        help='use cpu during experiement.')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='the gpu core to use during experiment.')
    parser.add_argument('--method',
                        type=str,
                        choices=['mmd', 'hsic'])
    parser.add_argument('--dataset',
                        type=str,
                        help='dataset to run tests on.')
    parser.add_argument('--save-dir',
                        type=str,
                        default=default_save_dir(),
                        help='directory to save experiment results/logs.')
    parser.add_argument('--n-samples',
                        type=int,
                        default=100,
                        help='number of samples used to compute the test statistic (default 100).')
    parser.add_argument('--n-shuffles',
                        type=int,
                        default=1)
    parser.add_argument('--n-tests',
                        type=int,
                        default=100,
                        help='number of permutation tests used to calculate empirical power.')
    parser.add_argument('--n-permutations',
                        type=int,
                        default=500,
                        help='number of permutations per permutation test.')
    parser.add_argument('--num-workers',
                        type=int,
                        default=0,
                        help='number of dataloader workers.')
    return parser.parse_args()

def default_save_dir():
    dt = datetime.now()
    dtstr = dt.strftime('%Y%m%d-%H%M%S')
    return f'exp/{dtstr}/'

def dataset(name):
    if name == 'Sinusoid.1f.1d':
        return Sinusoid(size=1000000,
                        frequency=1,
                        dim=1,
                        split='test',
                        train_val_test_split='0:0:10')
    elif name == 'Sinusoid.2f.1d':
        return Sinusoid(size=1000000,
                        frequency=2,
                        dim=1,
                        split='test',
                        train_val_test_split='0:0:10')
    elif name == 'Sinusoid.2f.4d':
        return Sinusoid(size=1000000,
                        frequency=2,
                        dim=4,
                        split='test',
                        train_val_test_split='0:0:10')

MEDIAN = 2.

def eval_hsic(dataset: Dataset,
              n_samples: int,
              n_tests: int = 100,
              n_permutations: int = 500,
              significance: float = 0.05,
              num_workers: int = 0,
              device: torch.device = torch.device('cpu')):

    dataloader = DataLoader(dataset,
                            batch_size=n_samples,
                            shuffle=True,
                            num_workers=num_workers,
                            drop_last=True)
    k = Gaussian(bandwidth=MEDIAN).to(device)
    l = Gaussian(bandwidth=MEDIAN).to(device)

    stats = defaultdict(list)
    n_reject = 0
    test_iter = iter(dataloader)
    for i in (pbar:=tqdm(range(n_tests),
                         bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                         dynamic_ncols=True,
                         leave=False)):
        try:
            batch = next(test_iter)
        except StopIteration:
            test_iter = iter(dataloader)
            batch = next(test_iter)

        X = batch[0].to(device)
        Y = batch[1].to(device)
        # median_x = median_heuristic(X)
        # median_y = median_heuristic(Y)
        # k.bandwidth = median_x
        # l.bandwidth = median_y
        # print(median_x)
        # print(median_y)
        # return 1/0

        hsic, var, p_value, r = metrics.hsic.permutation_test(k, l,
                                                              X, Y,
                                                              compute_var=False,
                                                              n_permutations=n_permutations,
                                                              significance=significance)
        if p_value < significance:
            n_reject += 1   # reject null hypothesis
        pbar.set_description(f"[{i+1}/{n_tests}] n_reject: {n_reject}, hsic: {hsic}, p-value: {p_value:.4f}")
        stats['stat'].append(hsic)
        stats['p-value'].append(p_value)
        stats['thresh'].append(r)

    avg = lambda x: sum(x)/len(x)
    stats['stat'] = avg(stats['stat'])
    stats['var'] = None
    stats['p-value'] = avg(stats['p-value'])
    stats['thresh'] = avg(stats['thresh'])
    stats['power'] = n_reject/n_tests
    return stats


def eval_mmd(dataset: Dataset,
             n_samples: int,
             n_tests: int = 100,
             n_shuffles: int = 1,
             n_permutations: int = 500,
             significance: float = 0.05,
             num_workers: int = 0,
             device: torch.device = torch.device('cpu')):

    dataloader = DataLoader(dataset,
                            batch_size=n_samples,
                            shuffle=True,
                            num_workers=num_workers,
                            drop_last=True)
    k = GaussianJoint(bandwidth=MEDIAN).to(device)

    stats = defaultdict(list)
    n_reject = 0
    test_iter = iter(dataloader)
    for i in (pbar:=tqdm(range(n_tests),
                         bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                         dynamic_ncols=True,
                         leave=False)):
        try:
            batch = next(test_iter)
        except StopIteration:
            test_iter = iter(dataloader)
            batch = next(test_iter)

        X = batch[0].to(device)
        Y = batch[1].to(device)
        mmd, var, p_value, r = metrics.mmd.permutation_test_pairs(k,
                                                            X, Y,
                                                            compute_var=False,
                                                            n_shuffles=n_shuffles,
                                                            n_permutations=n_permutations,
                                                            significance=significance,)
        if p_value < significance:
            n_reject += 1   # reject null hypothesis
        pbar.set_description(f"[{i+1}/{n_tests}] n_reject: {n_reject}, mmd: {mmd}, p-value: {p_value:.4f}")
        stats['stat'].append(mmd)
        stats['p-value'].append(p_value)
        stats['thresh'].append(r)

    avg = lambda x: sum(x)/len(x)
    stats['stat'] = avg(stats['stat'])
    stats['var'] = None
    stats['p-value'] = avg(stats['p-value'])
    stats['thresh'] = avg(stats['thresh'])
    stats['power'] = n_reject/n_tests
    return stats




def main(args):
    utils.seed_all(0)
    testset = dataset(args.dataset)

    if args.method == 'hsic':
        stats = eval_hsic(dataset=testset,
                        n_samples=args.n_samples,
                        n_tests=args.n_tests,
                        n_permutations=args.n_permutations,
                        num_workers=args.num_workers)

    elif args.method == 'mmd':
        stats = eval_mmd(dataset=testset,
                         n_samples=args.n_samples,
                         n_tests=args.n_tests,
                         n_shuffles=args.n_shuffles,
                         n_permutations=args.n_permutations,
                         num_workers=args.num_workers)

    print(dict(stats))

    # save evaluation metrics
    table = utils.Tabular(f"{args.save_dir}/stats-hsic-vs-mmd.csv")
    row = {
        'dataset': f"{args.dataset}-{args.n_samples}",
        'method': args.method,
        'model': 'median',
        'n_samples': args.n_samples,
        'n_shuffles': args.n_shuffles,
        **stats
    }
    table.append(row)
    table.to_csv()


if __name__=='__main__':
    main(parse_args())





def pDist2(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    r"""compute all paired (squared) distances between samples of X and Y
    X: (Nx, D) torch.Tensor
    Y: (Ny: D) torch.Tensor
    returns matrix of paired distances of size (Nx, Ny)"""
    xyT = X @ Y.mT                      # (Nx, Ny) pairwise inner products <x_i, y_j>
    x_norm2 = torch.sum(X**2, dim=-1)   # (Nx,)
    y_norm2 = torch.sum(Y**2, dim=-1)   # (Ny,)
    x_norm2 = x_norm2.unsqueeze(-1)     # (Nx, 1)
    Dxy = x_norm2 - 2*xyT + y_norm2     # (Nx, Ny) pairwise distances |x_i - y_j|^2
    Dxy[Dxy<0] = 0                      # TODO: clamp to stable values
    return Dxy

