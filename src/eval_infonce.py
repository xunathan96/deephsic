import argparse
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import metrics
from config.config import Config
from trainer import registry
from utils import utils
from kernel import Kernel

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
    parser.add_argument('--num-workers',
                        type=int,
                        default=0,
                        help='number of workers for the dataloader.')
    parser.add_argument('--save-dir',
                        type=str,
                        default=default_save_dir(),
                        help='directory to save experiment results/logs.')
    parser.add_argument('--pretrained-path',
                        type=str,
                        help='filepath from which to load the checkpoint.')
    parser.add_argument('--n-samples',
                        type=int,
                        default=100,
                        help='number of samples used to compute the test statistic (default 100).')
    parser.add_argument('--n-tests',
                        type=int,
                        default=100,
                        help='number of permutation tests used to calculate empirical power.')
    parser.add_argument('--n-permutations',
                        type=int,
                        default=500,
                        help='number of permutations per permutation test.')
    return parser.parse_args()

def default_save_dir():
    dt = datetime.now()
    dtstr = dt.strftime('%Y%m%d-%H%M%S')
    return f'exp/{dtstr}/'



def main(args):
    cfg = Config(yaml_path=args.config,
                 device=f'cuda:{args.gpu}' if not args.cpu else 'cpu',
                 save_dir=args.save_dir)
    cfg['dataloader']['test']['num_workers'] = args.num_workers
    utils.seed_all(cfg['seed'])
    pipeline = registry.get('InfoNCE').build(cfg)
    pipeline.load_checkpoint(args.pretrained_path)
    stats = pipeline.eval(n_samples=args.n_samples,
                          n_tests=args.n_tests,
                          n_permutations=args.n_permutations)
    print(stats)

    # save evaluation metrics
    table = utils.Tabular(f"{args.save_dir}/stats-infonce.csv")
    row = {
        'dataset': cfg['dataset']['name'],
        'function': cfg['model']['name'],
        'n-samples': args.n_samples,
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

def print_params(model):
    print('feature_kernel_bandwidth:', model.feature_kernel.bandwidth)
    print('smoothing_kernel_bandwidth', model.smoothing_kernel.bandwidth)
    print('deepkernel_eps:', model.eps)
    print(model.featurizer.net[1].linear.weight)

def are_params_equal(model_1: torch.nn.Module, model_2: torch.nn.Module):
    for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def compare_models(model_1: torch.nn.Module, model_2: torch.nn.Module):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        #print('Models match perfectly! :)')
        return True
    return False


if __name__=='__main__':
    main(parse_args())




