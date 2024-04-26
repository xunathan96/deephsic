import argparse
from collections import defaultdict
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import torch
from torchvision import transforms

from utils import utils
from data.toy import HDGM
from data.cifar10h import CIFAR10H
from data.imagenet_c import ImageNetC
from kernel import Gaussian, median_heuristic
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
    parser.add_argument('--dataset',
                        type=str,
                        choices=['HDGM-4', 'HDGM-8', 'HDGM-10', 'HDGM-20', 'HDGM-30', 'HDGM-40', 'HDGM-50', 'Cifar10h', 'ImageNet-GN-ZB-F'],
                        help='dataset to run tests on.')
    parser.add_argument('--save-dir',
                        type=str,
                        default=default_save_dir(),
                        help='directory to save experiment results/logs.')
    parser.add_argument('--n-samples',
                        type=int,
                        default=100,
                        help='number of samples used to compute the test statistic (default 100).')
    return parser.parse_args()

def default_save_dir():
    dt = datetime.now()
    dtstr = dt.strftime('%Y%m%d-%H%M%S')
    return f'exp/{dtstr}/'



def eval_hsic_median(dataloader: DataLoader,
                     n_tests: int = 100,
                     n_permutations: int = 500,
                     significance: float = 0.05,
                     device: torch.device = torch.device('cpu')):
    k = Gaussian().to(device)
    l = Gaussian().to(device)
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
        X = torch.flatten(X, start_dim=1)
        k.bandwidth = median_heuristic(X)
        l.bandwidth = median_heuristic(Y)

        # Dxx = pDist2(X, X)    # (Nx, Ny)
        # mahalanobis = -0.5*Dxx/(k.bandwidth**2)
        # kxx = k(X,X)
        # lyy = l(Y,Y)

        # print('k.bandwidth:', k.bandwidth)
        # print('l.bandwidth:', l.bandwidth)
        # print('Dxx:', Dxx)
        # print('Dxx.max():', Dxx.max())  # for image x this max is 1730 !!
        # print('Dxx.min():', Dxx.min())  # 0
        # print('mahalanobis.max():', mahalanobis.max())
        # print('mahalanobis.min():', mahalanobis.min())
        # print('kxx.max():', kxx.max())
        # print('kxx.min():', kxx.min())

        # return 1/0

        hsic, var, p_value, r = metrics.hsic.permutation_test(k, l,
                                                              X, Y,
                                                              compute_var=False,
                                                              n_permutations=n_permutations,
                                                              significance=significance)
        if p_value < significance:
            n_reject += 1   # reject null hypothesis
        pbar.set_description(f"[{i+1}/{n_tests}] hsic: {hsic}, p-value: {p_value:.4f}")
        stats['hsic'].append(hsic)
        stats['p-value'].append(p_value)
        stats['thresh'].append(r)

    avg = lambda x: sum(x)/len(x)
    stats['hsic'] = avg(stats['hsic'])
    stats['var'] = None
    stats['p-value'] = avg(stats['p-value'])
    stats['thresh'] = avg(stats['thresh'])
    stats['power'] = n_reject/n_tests
    return stats


def dataset(name):
    if name == 'HDGM-4':
        return HDGM(dim=4, size=10000)
    elif name == 'HDGM-8':
        return HDGM(dim=8, size=10000)
    elif name == 'HDGM-10':
        return HDGM(dim=10, size=10000)
    elif name == 'HDGM-20':
        return HDGM(dim=20, size=10000)
    elif name == 'HDGM-30':
        return HDGM(dim=30, size=10000)
    elif name == 'HDGM-40':
        return HDGM(dim=40, size=10000)
    elif name == 'HDGM-50':
        return HDGM(dim=50, size=10000)
    elif name == 'Cifar10h':
        return CIFAR10H(root='data/cifar10h/raw',
                        split='test',
                        download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
                        ]))
    elif name == 'ImageNet-GN-ZB-F':
        return ImageNetC(root='data/imagenet_c',
                         corruption='gn_zb_f',
                         split='test',
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                         ]))



def pDist2(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    r"""compute all paired (squared) distances between samples of X and Y
    X: (Nx, D) torch.Tensor
    Y: (Ny: D) torch.Tensor
    returns matrix of paired distances of size (Nx, Ny)"""
    xyT = X @ Y.T                       # (Nx, Ny) pairwise inner products <x_i, y_j>
    x_norm2 = torch.sum(X**2, dim=-1)   # (Nx,)
    y_norm2 = torch.sum(Y**2, dim=-1)   # (Ny,)
    x_norm2 = x_norm2.unsqueeze(-1)     # (Nx, 1)
    Dxy = x_norm2 - 2*xyT + y_norm2     # (Nx, Ny) pairwise distances |x_i - y_j|^2
    Dxy[Dxy<0] = 0                      # TODO: clamp to stable values
    return Dxy


def main(args):
    utils.seed_all(0)
    testset = dataset(args.dataset)
    testloader = DataLoader(testset,
                            batch_size=args.n_samples,
                            shuffle=True,
                            drop_last=True)

    stats = eval_hsic_median(dataloader=testloader, n_tests=100)
    print(dict(stats))

    # save evaluation metrics
    table = utils.Tabular(f"{args.save_dir}/stats-hsic.csv")
    row = {
        'dataset': args.dataset,
        'kernel': 'median',
        'n-samples': args.n_samples,
        **stats
    }
    table.append(row)
    table.to_csv()


if __name__=='__main__':
    main(parse_args())




