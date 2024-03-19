import argparse
__all__ = ['add_wandb_args']

def add_wandb_args(parser: argparse.ArgumentParser):
    # parser.add_argument('--wandb',
    #                     action='store_true',
    #                     help='log to wandb.')
    parser.add_argument('--project',
                        type=str,
                        default=None,
                        help='wandb project name.')
    parser.add_argument('--name',
                        type=str,
                        default=None,
                        help='run display name.')
    parser.add_argument('--mode',
                        type=str,
                        default='online',
                        help='wandb mode.')
    parser.add_argument('--sweep',
                        action='store_true',
                        help='use wandb sweep.')
    return parser

