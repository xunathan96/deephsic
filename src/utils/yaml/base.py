import yaml
import inspect
__all__ = ['BaseMappingObject', 'BaseSequenceObject', 'BaseScalarObject']


class BaseMappingObject:
    r"""This abstract class is an interface between YAML mapping nodes and python objects.
    The node described by this mapping object has an associated yaml tag."""
    yaml_tag: str
    Blueprint: object
    #Recipe: object

    def __init__(self, params: dict):
        self.params: dict = params

    def __len__(self):
        return 1

    def __repr__(self):
        return f"{self.Blueprint.__name__}Builder({self.params})"

    def __getitem__(self, key):
        return self.params[key]

    def __contains__(self, key):
        if key in self.params:
            return True
        else:
            return False

    def build(self, *args, **kwds):
        built_params = build_recursive(self.params, *args, **kwds)  # build paramater objects recursively
        sig = list(inspect.signature(self.Blueprint).parameters)
        kwds = {k:v for k,v in (built_params|kwds).items() if k in sig} # filter out unnecessary arguments
        return self.Blueprint(*args, **kwds)

    @classmethod
    def from_yaml(cls, loader: yaml.loader, node: yaml.Node):
        r"""Constructs an object instance from the given yaml node."""
        if isinstance(node, yaml.nodes.MappingNode):
            params = loader.construct_mapping(node, deep=True)
            self = cls(params)
        else:
            raise Exception(f'{cls.yaml_tag} specifies a MappingNode but found {node.__class__.__name__} instead.')
        return self

    @classmethod
    def to_yaml(cls, dumper: yaml.dumper, self):
        r"""Maps the given object instance back to a yaml node."""
        return dumper.represent_mapping(cls.yaml_tag, mapping=self.params)



class BaseSequenceObject:
    r"""This abstract class is an interface between YAML sequence nodes and python objects.
    The node described by this sequence object has an associated yaml tag."""
    yaml_tag: str
    Blueprint: object

    def __init__(self, seq: list):
        self.seq: list = seq

    def __len__(self):
        return len(self.seq)

    def __repr__(self):
        return f"{self.Blueprint.__name__}Builder({self.seq})"

    def __getitem__(self, idx):
        return self.seq[idx]

    def build(self, unpack=False, *args, **kwds):
        built_seq = build_recursive(self.seq, *args, **kwds)  # build paramater objects recursively
        # built blueprint based on packed or unpacked sequence of nodes
        if not unpack: built_seq = [built_seq]
        args = tuple(built_seq + list(args))
        # filter out unnecessary arguments
        sig = list(inspect.signature(self.Blueprint).parameters)
        kwds = {k:v for k,v in kwds.items() if k in sig}
        return self.Blueprint(*args, **kwds)

    @classmethod
    def from_yaml(cls, loader: yaml.loader, node: yaml.Node):
        r"""Constructs an object instance from the given yaml node."""
        if isinstance(node, yaml.nodes.SequenceNode):
            seq = loader.construct_sequence(node, deep=True)
            self = cls(seq)
        else:
            raise Exception(f'{cls.yaml_tag} specifies a SequenceNode but found {node.__class__.__name__} instead.')
        return self

    @classmethod
    def to_yaml(cls, dumper: yaml.dumper, self):
        r"""Maps the given object instance back to a yaml node."""
        return dumper.represent_sequence(cls.yaml_tag, sequence=self.seq)



class BaseScalarObject:
    r"""This abstract class is an interface between YAML sca;ar nodes and python objects.
    The node described by this scalar object has an associated yaml tag."""
    yaml_tag: str
    Blueprint: object

    def __init__(self, value):
        self.value = value

    def __len__(self):
        return 1

    def __repr__(self):
        return f"{self.Blueprint.__name__}Builder({self.value})"

    def build(self, *args, **kwds):
        if self.value != '':
            return self.Blueprint(self.value, *args, **kwds)
        else:
            return self.Blueprint(*args, **kwds)

    @classmethod
    def from_yaml(cls, loader: yaml.loader, node: yaml.Node):
        r"""Constructs an object instance from the given yaml node."""
        if isinstance(node, yaml.nodes.ScalarNode):
            value = loader.construct_scalar(node)
            self = cls(value)
        else:
            raise Exception(f'{cls.yaml_tag} specifies a ScalarNode but found {node.__class__.__name__} instead.')
        return self

    @classmethod
    def to_yaml(cls, dumper: yaml.dumper, self):
        r"""Maps the given object instance back to a yaml node."""
        return dumper.represent_scalar(cls.yaml_tag, value=self.value)





# ==============================
#       HELPER FUNCTIONS
# ==============================

def build_recursive(root, *args, **kwds):
    r"""recursively build all yaml objects starting from the given root object"""
    if isinstance(root, BaseMappingObject | BaseSequenceObject | BaseScalarObject):
        return root.build(*args, **kwds)
    elif isinstance(root, dict):
        node = dict()
        for k,v in root.items():
            node[k] = build_recursive(v, *args, **kwds)
        return node
    elif isinstance(root, list):
        node = list()
        for v in root:
            node.append(build_recursive(v, *args, **kwds))
        return node
    else:
        return root

