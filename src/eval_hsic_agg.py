import argparse
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from agginc import agginc, human_readable_dict  # jax version
import jax
import matplotlib.pyplot as plt
from torchvision import transforms

from utils import utils
from data.toy import HDGM, Sinusoid
from data.imagenet_c import ImageNetC
from data.cifar10h import CIFAR10H
from data.riab import RatInABox
from data.transforms import NumpyToTensor

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
                        choices=['HDGM-4', 'HDGM-8', 'HDGM-10', 'HDGM-20', 'HDGM-30', 'HDGM-40', 'HDGM-50',
                                 'Cifar10h', 'ImageNet-GN-ZB-F', 'RatInABox', 'Sinusoid'],
                        help='dataset to run tests on.')
    parser.add_argument('--save-dir',
                        type=str,
                        default=default_save_dir(),
                        help='directory to save experiment results/logs.')
    parser.add_argument('--n-samples',
                        type=int,
                        default=100,
                        help='number of samples used to compute the test statistic (default 100).')
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
    if name == 'HDGM-4':
        return HDGM(dim=4, size=10000)
    elif name == 'HDGM-8':
        return HDGM(dim=8, size=10000)
    elif name == 'HDGM-10':
        return HDGM(dim=10, size=10000)
    elif name == 'HDGM-20':
        return HDGM(dim=20, size=100000)
    elif name == 'HDGM-30':
        return HDGM(dim=30, size=100000)
    elif name == 'HDGM-40':
        return HDGM(dim=40, size=100000)
    elif name == 'HDGM-50':
        return HDGM(dim=50, size=100000)
    elif name == 'Cifar10h':
        return CIFAR10H(root='data/cifar10h/raw',
                        split='test',
                        train_val_test_split='0:0:10',
                        download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
                        ]))
    elif name == 'ImageNet-GN-ZB-F':
        return ImageNetC(root='data/imagenet_c/112x112/',
                         corruption='gn_zb_f',
                         split='test',
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                         ]))
    elif name == 'RatInABox':
        return RatInABox(root='data/riab/raw/riab-20000-T.pkl',
                         split='test',
                         train_val_test_split='0:0:10',
                         window='present',
                         transform=NumpyToTensor())
    elif name == 'Sinusoid':
        return Sinusoid(size=1000000,
                        frequency=4,
                        dim=2,
                        split='test',
                        train_val_test_split='0:0:10')


def eval_hsic_agg(dataloader: DataLoader,
                  n_tests: int = 100):
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

        X = batch[0].numpy()
        Y = batch[1].numpy()
        X = X.reshape((X.shape[0], -1))
        Y = Y.reshape((X.shape[0], -1))

        # plt.scatter(X[:,0], Y[:,1], s=2)
        # plt.axis('equal')
        # plt.show()
        # return 1/0
        # R = min(X.shape[0], Y.shape[0]) - 1     # use full U-statistics
        reject = agginc("hsic", X, Y)
        n_reject += reject
        pbar.set_description(f"[{i+1}/{n_tests}] n_reject: {n_reject}")

    stats = dict()
    stats['hsic'] = None
    stats['var'] = None
    stats['p-value'] = None
    stats['thresh'] = None
    stats['power'] = n_reject.item()/n_tests
    return stats





def main(args):
    utils.seed_all(0)
    testset = dataset(args.dataset)
    testloader = DataLoader(testset,
                            batch_size=args.n_samples,
                            shuffle=True,
                            drop_last=True,
                            num_workers=args.num_workers,)

    stats = eval_hsic_agg(dataloader=testloader,
                          n_tests=100)
    print(stats)

    # save evaluation metrics
    table = utils.Tabular(f"{args.save_dir}/stats-hsic.csv")
    row = {
        'dataset': f"{args.dataset}-{args.n_samples}",
        'method': 'hsic',
        'model': 'agg',
        'n_samples': args.n_samples,
        **stats
    }
    table.append(row)
    table.to_csv()


if __name__=='__main__':
    main(parse_args())




