import argparse
from datetime import datetime
from pathlib import Path

from config.config import Config
from trainer import registry
from utils import utils

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
    parser.add_argument('--n-epochs',
                        type=int,
                        default=100,
                        help='number of epochs to use during training')
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

    # save config
    sf = Path(cfg['save_dir'])/Path(args.config).name
    cfg.save(sf)

    dHsic = registry.get('HSIC').build(cfg)
    dHsic.train(epochs=args.n_epochs)
    stats = dHsic.eval()
    print(stats)

    # save evaluation results
    table = utils.Tabular(f"{args.save_dir}/eval.csv")
    table.append(stats)
    table.to_csv()




def bandwidth(args):
    cfg = Config(file=args.config,
                 device=f'cuda:{args.gpu}' if not args.cpu else 'cpu',
                 save_dir=args.save_dir)
    utils.seed_all(cfg['seed'])
    dHsic = registry.get('HSIC').build(cfg)

    import torch
    from kernel import Gaussian
    k = Gaussian()
    l = Gaussian(trainable=False)
    k = k.to(cfg['device'])
    l = l.to(cfg['device'])

    X = torch.tensor([
        [1.,1.],
        [2.,0.],
        [3.,-1.]
    ]).to(cfg['device'])
    Y = torch.tensor([
        [-1.,1.],
        [-2.,0.],
        [-3.,-1.]
    ]).to(cfg['device'])

    Kxy = k(X,Y)
    Lxy = l(X,Y)
    print(Kxy)
    print(Lxy)

    # save config
    sf = Path(cfg['save_dir'])/Path(args.config).name
    cfg.save(sf)



if __name__=='__main__':
    main(parse_args())
    #bandwidth(parse_args())




