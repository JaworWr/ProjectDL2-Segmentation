import yaml
from dotmap import DotMap


def get_config(path: str):
    with open(path) as f:
        cfg_text = f.read()
        cfg = yaml.safe_load(cfg_text)
        cfg = DotMap(cfg)
        cfg._text = cfg_text
        return cfg
