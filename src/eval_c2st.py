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
    parser.add_argument('--statistic',
                        type=str,
                        choices=['accuracy', 'logit'],
                        default='logit',
                        help='test statistic to use.')
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
    pipeline = registry.get('C2ST').build(cfg)
    pipeline.load_checkpoint(args.pretrained_path)
    stats = pipeline.eval(n_samples=args.n_samples,
                          n_tests=args.n_tests,
                          n_permutations=args.n_permutations,
                          statistic=args.statistic)
    print(stats)

    # save evaluation metrics
    table = utils.Tabular(f"{args.save_dir}/stats-c2st.csv")
    row = {
        'dataset': cfg['dataset']['name'],
        'classifier': cfg['model']['name'],
        'type': args.statistic,
        'n-samples': args.n_samples,
        **stats
    }
    table.append(row)
    table.to_csv()

    # save config
    sf = Path(cfg['save_dir'])/"settings"/Path(args.config).name
    cfg.save(sf)



if __name__=='__main__':
    main(parse_args())




