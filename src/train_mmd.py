import argparse
from datetime import datetime
from pathlib import Path

from config.config import Config
from trainer import registry
from utils import utils
from utils.wandb import add_wandb_args
from utils.yaml import parse_yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-config',
                        type=str,
                        help='path to the experiment config file.')
    parser.add_argument('--data-config',
                        type=str,
                        help='path to the dataset config file.')
    parser.add_argument('--model-config',
                        type=str,
                        help='path to the model config file.')
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
    parser.add_argument('--n-tests',
                        type=int,
                        default=100,
                        help='number of permutation tests used to calculate empirical power.')
    parser.add_argument('--n-permutations',
                        type=int,
                        default=500,
                        help='number of permutations per permutation test.')
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
    cfg = Config(yaml_path = args.train_config,
                 device = f'cuda:{args.gpu}' if not args.cpu else 'cpu',
                 save_dir = args.save_dir,
                 n_epochs = args.n_epochs,)
    cfg.update(dataset=parse_yaml(args.data_config)) if args.data_config else None
    cfg.update(model=parse_yaml(args.model_config)) if args.model_config else None
    if 'wandb' in args: cfg.set('wandb', vars(args.wandb))
    for split in ['train', 'val', 'test']:
        cfg['dataloader'][split]['num_workers'] = args.num_workers
    utils.seed_all(cfg['seed'])

    # save config
    save_config = '--'.join([Path(pth).stem for pth in (args.train_config, args.data_config, args.model_config) if pth is not None])
    sf = Path(cfg['save_dir'])/Path(save_config).with_suffix('.yml')
    cfg.save(sf)

    pipeline = registry.get('MMD').build(cfg)
    pipeline.train(epochs=args.n_epochs)
    return 1

    stats = pipeline.eval(n_tests=args.n_tests, n_permutations=args.n_permutations)
    print(stats)

    # save evaluation results
    table = utils.Tabular(f"{args.save_dir}/eval.csv")
    table.append(stats)
    table.to_csv()


if __name__=='__main__':
    main(parse_args())
