import argparse
import pickle
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
    parser.add_argument('--pretrained-path',
                        type=str,
                        help='filepath from which to load the checkpoint.')
    parser.add_argument('--n-samples',
                        type=int,
                        default=100,
                        help='number of samples used to compute the test statistic (default 100).')
    return parser.parse_args()

def default_save_dir():
    dt = datetime.now()
    dtstr = dt.strftime('%Y%m%d-%H%M%S')
    return f'exp/{dtstr}/'



def main(args):
    cfg = Config(yaml_path = args.eval_config,
                 device = f'cuda:{args.gpu}' if not args.cpu else 'cpu',)
    cfg.update(dataset=parse_yaml(args.data_config)) if args.data_config else None
    cfg.update(model=parse_yaml(args.model_config)) if args.model_config else None
    cfg['dataloader']['test']['num_workers'] = args.num_workers
    cfg['dataloader']['test']['batch_size'] = args.n_samples
    cfg['dataloader']['test']['shuffle'] = False
    utils.seed_all(cfg['seed'])

    pipeline = registry.get(cfg['method']).build(cfg)
    pipeline.load_checkpoint(args.pretrained_path)
    gram = pipeline.gram()

    with open(f"gram_{cfg['method']}_{cfg['dataset']['name']}_n={args.n_samples}.pkl", 'wb') as file: 
        pickle.dump(gram, file) 



if __name__=='__main__':
    main(parse_args())




