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

    deepMMD = registry.get('MMD').build(cfg)
    deepMMD.train(epochs=args.n_epochs)
    stats = deepMMD.eval()
    print(stats)

    # save evaluation results
    table = utils.Tabular(f"{args.save_dir}/eval.csv")
    table.append(stats)
    table.to_csv()


if __name__=='__main__':
    main(parse_args())
