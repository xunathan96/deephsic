from pathlib import Path
from utils.yaml.parser import parse_yaml, dump_yaml


class Config:
    
    def __init__(self, file: str, **kwds):
        self.yaml_cfg = parse_yaml(Path(file))
        self.__dict__.update(kwds)

        if shared:=set(self.yaml_cfg).intersection(kwds):
            raise Warning(f"The keys {shared} exists in both the YAML file and given keyword arguments.")

    def __repr__(self):
        return self.yaml_cfg.__repr__() # for debugging

    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif key in self.yaml_cfg:
            return self.yaml_cfg[key]
        else:
            raise KeyError(f'{key}')

    def __contains__(self, item):
        if item in self.__dict__:
            return True
        elif item in self.yaml_cfg:
            return True
        else:
            return False

    def save(self, filepath: str):
        return dump_yaml(self.yaml_cfg, file=Path(filepath))


