from argparse import ArgumentParser


def parse_args(parser: ArgumentParser, strict=True):
    args = parser.parse_args()
    if args.config is not None:
        import yaml
        # load yaml file and override specified args

        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Config file {args.config} is empty")

        parser.set_defaults(**config)

        if strict:
            for key in config.keys():
                if not hasattr(args, key):
                    raise ValueError(f"Unknown argument: '{key}' specified in {args.config}")

        args = parser.parse_args()

    return args
