import yaml
from pathlib import Path
from .registry import yamlRegistry
__all__ = ['parse_yaml', 'dump_yaml']

def parse_yaml(file: str):
    Loader: yaml.SafeLoader = yaml.SafeLoader
    for cls in yamlRegistry.values():
        Loader.add_constructor(cls.yaml_tag, constructor=cls.from_yaml)

    Include.add(Path(file)) # keep track of all imports
    Loader.add_constructor(Include.yaml_tag, constructor=Include.from_yaml)

    with open(Path(file), 'r') as stream:
        data = yaml.load(stream, Loader=Loader)

    Include.remove(Path(file))
    return data


def dump_yaml(data, file: str):
    Dumper: yaml.SafeDumper = yaml.SafeDumper
    for cls in yamlRegistry.values():
        Dumper.add_representer(cls, representer=cls.to_yaml)

    file = Path(file).with_suffix('.yml')
    file.parent.mkdir(parents=True, exist_ok=True)

    with open(file, 'w') as stream:
        yaml_str = yaml.dump(data, Dumper=Dumper)
        stream.write(yaml_str)



class Include:
    yaml_tag = u'!include'
    imported = set()

    @classmethod
    def from_yaml(cls, loader: yaml.loader, node: yaml.Node):
        if isinstance(node, yaml.nodes.ScalarNode):
            file = loader.construct_scalar(node)
            ext = Path(file).suffix
            if ext in ('.yml', '.yaml'):
                return parse_yaml(file)
            else:
                NotImplementedError()
        else:
            raise Exception(f'{cls.yaml_tag} specifies a ScalarNode but found {node.__class__.__name__} instead.')

    @classmethod
    def add(cls, file: str):
        file = Path(file)
        if file in cls.imported:
            raise Exception(f'Error [circular import]: {file} has already been imported.')
        else:
            cls.imported.add(file)

    @classmethod
    def remove(cls, file: str):
        cls.imported.remove(Path(file))
