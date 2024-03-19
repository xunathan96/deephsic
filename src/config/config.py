from pathlib import Path
from utils.yaml.parser import parse_yaml, dump_yaml


class Config_depreciated:    
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

    def save(self, filepath: str):
        return dump_yaml(self.__dict__, file=Path(filepath))
