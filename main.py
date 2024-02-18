import argparse
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import metrics
from config.config import Config
from trainer import registry
from utils import utils
from kernel import Kernel, Gaussian, median_heuristic

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='config/exp/train.dhsic.blobHD.adam.1e-4.yml',
                        help='path to the experiment config file.')
    parser.add_argument('--cpu',
                        action='store_true',
                        help='use cpu during experiement.')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='the gpu core to use during experiment.')
    parser.add_argument('--n-epochs',
                        type=int,
                        default=10,
                        help='number of epochs to use during training')
    parser.add_argument('--save-dir',
                        type=str,
                        default=default_save_dir(),
                        help='directory to save experiment results/logs.')
    parser.add_argument('--checkpoint-fp',
                        type=str,
                        help='filepath from which to load the checkpoint.')
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

    # save config
    sf = Path(cfg['save_dir'])/Path(args.config).name
    cfg.save(sf)

    #Classifier = registry.get('Classifier').build(cfg)
    #Classifier.train(epochs=args.n_epochs)
    #metrics = Classifier.eval()
    #print(metrics)

    dHsic = registry.get('HSIC').build(cfg)
    #dHsic.train(epochs=args.n_epochs)
    dHsic.load(args.checkpoint_fp)
    stats = dHsic.eval()
    print(stats)

    """
    stats = eval_median_heuristic(dataloader=dHsic.dataloader['test'],
                                  device=dHsic.device)
    stats = eval_test_power(dataloader=dHsic.dataloader['test'],
                            k=Gaussian(),
                            l=Gaussian(),
                            device=dHsic.device)
    """



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
    stats = defaultdict(list)
    n_tests = len(dataloader)
    n_reject = 0
    for i, batch in enumerate(pbar:=tqdm(dataloader,
                                         bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                                         dynamic_ncols=True,
                                         leave=False)):
        joint = batch.to(device)    # (B, D)
        X,Y = marginals(joint)
        median_dist_X = median_heuristic(X)
        median_dist_Y = median_heuristic(Y)
        k = Gaussian(bandwidth=median_dist_X)
        l = Gaussian(bandwidth=median_dist_Y)
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








def test():
    from utils.yaml.parser import parse_yaml, dump_yaml

    cfg = parse_yaml(file='config/exp/train.cnn0.mnist.adam.1e-4.yml')
    print('CFG:', cfg)

    x = torch.rand(10, requires_grad=True)
    opt = cfg['optimizer'].build(params=[x])
    print('OPT:', opt)

    tf = cfg['dataset']['transform']
    print(tf.build())

    dataset = cfg['dataset']
    print(dataset.build())

    loss = cfg['criterion']
    print(loss.build())

    scheduler = cfg['scheduler']
    print(scheduler.build(optimizer=opt))

    model = cfg['model']
    print(model.build())

    dump_yaml(cfg, file='test.yml')






if __name__=='__main__':
    main(parse_args())




