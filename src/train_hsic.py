import argparse
from datetime import datetime
from pathlib import Path

from config.config import Config
from trainer import registry
from utils import utils
from utils.wandb import add_wandb_args

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
    parser.add_argument('--n-epochs',
                        type=int,
                        default=100,
                        help='number of epochs to use during training')
    parser.add_argument('--save-dir',
                        type=str,
                        default=default_save_dir(),
                        help='directory to save experiment results/logs.')
    # add wandb subparser
    subparsers = parser.add_subparsers(action=utils.NestArgs)
    parser_wandb = subparsers.add_parser('wandb')
    add_wandb_args(parser_wandb)
    return parser.parse_args()

def default_save_dir():
    dt = datetime.now()
    dtstr = dt.strftime('%Y%m%d-%H%M%S')
    return f'exp/{dtstr}/'



def main(args):
    cfg = Config(yaml_path=args.config,
                 device=f'cuda:{args.gpu}' if not args.cpu else 'cpu',
                 save_dir=args.save_dir,
                 n_epochs=args.n_epochs,)
    if 'wandb' in args: cfg.set('wandb', vars(args.wandb))
    for split in ['train', 'val', 'test']:
        cfg['dataloader'][split]['num_workers'] = args.num_workers
    utils.seed_all(cfg['seed'])

    # save config
    sf = Path(cfg['save_dir'])/Path(args.config).name
    cfg.save(sf)

    pipeline = registry.get('HSIC').build(cfg)
    pipeline.train(epochs=args.n_epochs)
    stats = pipeline.eval()
    print(stats)

    # save evaluation results
    table = utils.Tabular(f"{args.save_dir}/eval.csv")
    table.append(stats)
    table.to_csv()





if __name__=='__main__':
    main(parse_args())




