import warnings
from pathlib import Path
from utils.yaml.parser import parse_yaml, dump_yaml

class Config:
    def __init__(self,
                 yaml_path: str,
                 **kwds):
        yaml_dict = parse_yaml(Path(yaml_path))
        self.yaml_path = yaml_path
        self.__dict__.update(yaml_dict|kwds)
        if shared:=set(yaml_dict).intersection(kwds):
            raise Warning(f"The given keys {shared} overrides those found in {yaml_path}.")

    def __repr__(self):
        return self.__dict__.__repr__()

    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise KeyError(f'{key}')

    def __contains__(self, item):
        if item in self.__dict__:
            return True
        else:
            return False

    def get(self, key, default=None):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            return default

    def set(self, key, value):
        return setattr(self, key, value)

    def update(self, **kwds):
        if shared:=set(self.__dict__).intersection(kwds):
            warnings.warn(f"The given keys {shared} are overridden.")
        self.__dict__.update(kwds)

    def merge(self, config: "Config"):
        raise NotImplementedError()

    def save(self, filepath: str):
        return dump_yaml(self.__dict__, file=Path(filepath))
