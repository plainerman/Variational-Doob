from argparse import ArgumentParser
import sys
import yaml


def parse_args(parser: ArgumentParser):
    # args = parser.parse_args()
    args = sys.argv[1:]
    conf = args.index('--config')
    if conf >= 0 and conf + 1 < len(args):
        config_file = args[conf + 1]
    else:
        return parser.parse_args()

    # load yaml file and override specified args
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    cli_args = {}
    # convert args to dictionary
    i = 0
    while i < len(args):
        if args[i].startswith('--'):
            cli_args[args[i].replace('--', '')] = args[i + 1]
            i += 2
        else:
            i += 1

    config |= cli_args

    config_args = []
    for k, v in config.items():
        args.append(f'--{k}')
        args.append(str(v))

    if config is None:
        raise ValueError(f"Config file {config_file} is empty")

    return parser.parse_args(args=config_args)
