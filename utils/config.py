import yaml
from dotmap import DotMap


def get_config(path: str):
    with open(path) as f:
        cfg = yaml.safe_load(f)
        cfg = DotMap(cfg)
        return cfg
