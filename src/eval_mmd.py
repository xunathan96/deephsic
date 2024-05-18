import argparse
from datetime import datetime
from pathlib import Path

from config.config import Config
from trainer import registry
from utils import utils
from utils.yaml import parse_yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-config',
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
    parser.add_argument('--permutation-test',
                        type=str,
                        choices=['two-sample', 'independence', 'split-independence'],
                        default='independence',
                        help='the type of permutation test to use.')
    return parser.parse_args()

def default_save_dir():
    dt = datetime.now()
    dtstr = dt.strftime('%Y%m%d-%H%M%S')
    return f'exp/{dtstr}/'



def main(args):
    cfg = Config(yaml_path = args.eval_config,
                 device = f'cuda:{args.gpu}' if not args.cpu else 'cpu',
                 save_dir = args.save_dir)
    cfg.update(dataset=parse_yaml(args.data_config)) if args.data_config else None
    cfg.update(model=parse_yaml(args.model_config)) if args.model_config else None
    cfg['dataloader']['test']['num_workers'] = args.num_workers
    utils.seed_all(cfg['seed'])
    pipeline = registry.get('MMD').build(cfg)
    pipeline.load_checkpoint(args.pretrained_path)
    stats = pipeline.eval(n_samples=args.n_samples,
                          n_tests=args.n_tests,
                          n_permutations=args.n_permutations,
                          permutation_test=args.permutation_test)
    print(stats)

    # save evaluation metrics
    table = utils.Tabular(f"{args.save_dir}/stats-mmd.csv")
    row = {
        'dataset': cfg['dataset']['name'],
        'kernel': cfg['model']['name'],
        'permutation_test': args.permutation_test,
        'n_samples': args.n_samples,
        **stats
    }
    table.append(row)
    table.to_csv()

    # save config
    save_config = '--'.join([Path(pth).stem for pth in (args.eval_config, args.data_config, args.model_config) if pth is not None])
    sf = Path(cfg['save_dir'])/"settings"/Path(save_config).with_suffix('.yml')
    cfg.save(sf)



if __name__=='__main__':
    main(parse_args())




