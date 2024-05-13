import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

import metrics
from config.config import Config
from utils import utils
from kernel import Kernel, Gaussian, median_heuristic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        help='path to the experiment config file.')
    parser.add_argument('--cpu',
                        action='store_true',
                        help='use cpu during experiement.')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='the gpu core to use during experiment.')
    parser.add_argument('--save-dir',
                        type=str,
                        default=default_save_dir(),
                        help='directory to save experiment results/logs.')
    return parser.parse_args()

def default_save_dir():
    dt = datetime.now()
    dtstr = dt.strftime('%Y%m%d-%H%M%S')
    return f'exp/{dtstr}/'



def main(args):
    cfg = Config(file=args.config,
                 device=f'cuda:{args.gpu}' if not args.cpu else 'cpu',
                 save_dir=args.save_dir)
    utils.seed_all(cfg['seed'])
    device = torch.device(cfg['device'])
    dataset = cfg['dataset']['test'].build()
    dataloader = cfg['dataloader']['test'].build(dataset=dataset)
    stats = eval_median_heuristic(dataloader, device=device)
    print('median_heuristic:', stats)

    # save evaluation metrics
    table = utils.Tabular(f"{args.save_dir}/stats.csv")
    row = {
        'dataset': f"HDGM-{cfg['dataset']['test']['dim']}",
        'kernel': "median",
        'n_samples': cfg['dataloader']['test']['batch_size'],
        **stats
    }
    table.append(row)
    table.to_csv()

    # save config
    sf = Path(cfg['save_dir'])/"settings"/Path(args.config).name
    cfg.save(sf)





# ==============================
#       HYPOTHESIS TESTS
# ==============================

def eval_test_power(dataloader: DataLoader,
                k: Kernel,
                l: Kernel,
                n_permutations: int = 500,
                significance: float = 0.05,
                device: torch.device = torch.device('cpu')):
    r"""evaluate the empirical test power on the given dataloader using hsic with kernels k and l"""
    stats = defaultdict(list)
    n_tests = len(dataloader)
    n_reject = 0
    for i, batch in enumerate(pbar:=tqdm(dataloader,
                                         bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                                         dynamic_ncols=True,
                                         leave=False)):
        joint = batch.to(device)    # (B, D)
        X,Y = marginals(joint)
        hsic, var, p_value, r = metrics.hsic.permutation_test(k, l,
                                                              X, Y,
                                                              compute_var=False,
                                                              n_permutations=n_permutations,
                                                              significance=significance)
        if p_value < significance:
            n_reject += 1   # reject null hypothesis
        pbar.set_description(f"[{i+1}/{n_tests}] hsic: {hsic}, p-value: {p_value:.4f}")
        stats['hsic'].append(hsic.item())
        stats['p-value'].append(p_value)
        stats['thresh'].append(r)

    avg = lambda x: sum(x)/len(x)
    stats['hsic'] = avg(stats['hsic'])
    stats['var'] = None
    stats['p-value'] = avg(stats['p-value'])
    stats['thresh'] = avg(stats['thresh'])
    stats['power'] = n_reject/n_tests
    return stats


def eval_median_heuristic(dataloader: DataLoader,
                          n_permutations: int = 500,
                          significance: float = 0.05,
                          device: torch.device = torch.device('cpu')):
    r"""evaluate the empirical test power on the given dataloader using hsic with the median heuristic"""
    k = Gaussian().to(device)
    l = Gaussian().to(device)

    stats = defaultdict(list)
    n_tests = len(dataloader)
    n_reject = 0
    for i, batch in enumerate(pbar:=tqdm(dataloader,
                                         bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                                         dynamic_ncols=True,
                                         leave=False)):
        joint = batch.to(device)    # (B, D)
        X,Y = marginals(joint)
        k.bandwidth = median_heuristic(X)
        l.bandwidth = median_heuristic(Y)
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



if __name__=='__main__':
    main(parse_args())




