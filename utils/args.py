from argparse import ArgumentParser, ArgumentTypeError
import sys
import yaml


def parse_args(parser: ArgumentParser):
    args = sys.argv[1:]
    if args.count('--config') <= 0:
        return parser.parse_args()

    conf = args.index('--config')
    if conf >= 0 and conf + 1 < len(args):
        config_file = args[conf + 1]
    else:
        return parser.parse_args()

    # load yaml file and override specified args
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file {config_file} is empty")

    config_args = []
    for k, v in config.items():
        config_args.append(f'--{k}')
        for v_i in str(v).split():
            config_args.append(v_i)

    return parser.parse_args(args=config_args + args)


def str2bool(v):
    # https://stackoverflow.com/a/43357954/4417954
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')
