from .base import *
from utils.enums import YamlNodeType

import torch.nn as nn
import torchvision
import optim, data, model, kernel, mcmc

__all__ = ['yamlRegistry', 'YamlRegistry']

class YamlRegistry:
    r"""Registry class that stores all YAML custom tags and their mapped classes"""

    def __init__(self):
        self.registry = dict()
        self.stack = []

    def __repr__(self):
        return f"{self.registry}"
    
    def values(self):
        return self.registry.values()

    def register(self, yaml_tag, nodetype: YamlNodeType):
        self.stack.append((yaml_tag, nodetype))
        return self

    def map_to(self, recipe):
        curr_tag, nodetype = self.stack.pop()
        if nodetype == YamlNodeType.MAPPING:
            Base = BaseMappingObject
        elif nodetype == YamlNodeType.SEQUENCE:
            Base = BaseSequenceObject
        elif nodetype == YamlNodeType.SCALAR:
            Base = BaseScalarObject

        class YamlObject(Base):
            yaml_tag = curr_tag
            Blueprint = recipe
        self.registry[curr_tag] = YamlObject
        return self


yamlRegistry = YamlRegistry()

# data
yamlRegistry.register(yaml_tag=u'!torch.DataLoader',        nodetype=YamlNodeType.MAPPING).map_to(data.dataloader.DataLoader)
yamlRegistry.register(yaml_tag=u'!dataset.MNIST',           nodetype=YamlNodeType.MAPPING).map_to(torchvision.datasets.MNIST)
yamlRegistry.register(yaml_tag=u'!dataset.Gaussian2D',      nodetype=YamlNodeType.MAPPING).map_to(data.toy.Gaussian2D)
yamlRegistry.register(yaml_tag=u'!dataset.Blob2ST',         nodetype=YamlNodeType.MAPPING).map_to(data.toy.Blob2ST)
yamlRegistry.register(yaml_tag=u'!dataset.HDGM',            nodetype=YamlNodeType.MAPPING).map_to(data.toy.HDGM)
yamlRegistry.register(yaml_tag=u'!dataset.CIFAR10H',        nodetype=YamlNodeType.MAPPING).map_to(data.cifar10h.CIFAR10H)
yamlRegistry.register(yaml_tag=u'!dataset.ImageNetC',       nodetype=YamlNodeType.MAPPING).map_to(data.imagenet_c.ImageNetC)
yamlRegistry.register(yaml_tag=u'!dataset.RatInABox',       nodetype=YamlNodeType.MAPPING).map_to(data.riab.RatInABox)
# yamlRegistry.register(yaml_tag=u'!dataset.PennTreebank',    nodetype=YamlNodeType.MAPPING).map_to(data.penn_treebank.PennTreebank)
# yamlRegistry.register(yaml_tag=u'!dataset.Emotion',         nodetype=YamlNodeType.MAPPING).map_to(data.emotion.Emotion)
yamlRegistry.register(yaml_tag=u'!dataset.Sinusoid',        nodetype=YamlNodeType.MAPPING).map_to(data.toy.Sinusoid)

# transforms
yamlRegistry.register(yaml_tag=u'!transform.CenterCrop',    nodetype=YamlNodeType.MAPPING).map_to(data.transforms.CenterCrop)
yamlRegistry.register(yaml_tag=u'!transform.Resize',        nodetype=YamlNodeType.MAPPING).map_to(data.transforms.Resize)
yamlRegistry.register(yaml_tag=u'!transform.Normalize',     nodetype=YamlNodeType.MAPPING).map_to(data.transforms.Normalize)
yamlRegistry.register(yaml_tag=u'!transform.ToTensor',      nodetype=YamlNodeType.SCALAR).map_to(data.transforms.ToTensor)
yamlRegistry.register(yaml_tag=u'!transform.Compose',       nodetype=YamlNodeType.SCALAR).map_to(data.transforms.Compose)
yamlRegistry.register(yaml_tag=u'!transform.NumpyToTensor', nodetype=YamlNodeType.SCALAR).map_to(data.transforms.NumpyToTensor)
yamlRegistry.register(yaml_tag=u'!transform.Grayscale',     nodetype=YamlNodeType.SCALAR).map_to(data.transforms.Grayscale)
yamlRegistry.register(yaml_tag=u'!transform.GaussianBlur',  nodetype=YamlNodeType.MAPPING).map_to(data.transforms.GaussianBlur)

# optimizers
yamlRegistry.register(yaml_tag=u'!optim.Adam',      nodetype=YamlNodeType.MAPPING).map_to(optim.optimizer.Adam)
yamlRegistry.register(yaml_tag=u'!optim.AdamW',     nodetype=YamlNodeType.MAPPING).map_to(optim.optimizer.AdamW)
yamlRegistry.register(yaml_tag=u'!optim.SGD',       nodetype=YamlNodeType.MAPPING).map_to(optim.optimizer.SGD)
yamlRegistry.register(yaml_tag=u'!optim.Adagrad',   nodetype=YamlNodeType.MAPPING).map_to(optim.optimizer.Adagrad)

# schedulers
yamlRegistry.register(yaml_tag=u'!scheduler.LinearLR',                      nodetype=YamlNodeType.MAPPING).map_to(optim.scheduler.LinearLR)
yamlRegistry.register(yaml_tag=u'!scheduler.CosineAnnealingLR',             nodetype=YamlNodeType.MAPPING).map_to(optim.scheduler.CosineAnnealingLR)
yamlRegistry.register(yaml_tag=u'!scheduler.CosineAnnealingWarmRestarts',   nodetype=YamlNodeType.MAPPING).map_to(optim.scheduler.CosineAnnealingWarmRestarts)
yamlRegistry.register(yaml_tag=u'!scheduler.SequentialLR',                  nodetype=YamlNodeType.MAPPING).map_to(optim.scheduler.SequentialLR)

# criteria
yamlRegistry.register(yaml_tag=u'!criterion.CrossEntropyLoss',      nodetype=YamlNodeType.MAPPING).map_to(optim.criterion.CrossEntropyLoss)
yamlRegistry.register(yaml_tag=u'!criterion.BCEWithLogitsLoss',     nodetype=YamlNodeType.MAPPING).map_to(optim.criterion.BCEWithLogitsLoss)
yamlRegistry.register(yaml_tag=u'!criterion.MSELoss',               nodetype=YamlNodeType.MAPPING).map_to(optim.criterion.MSELoss)
yamlRegistry.register(yaml_tag=u'!criterion.HSIC',                  nodetype=YamlNodeType.SCALAR).map_to(optim.criterion.HSIC)
yamlRegistry.register(yaml_tag=u'!criterion.MMDTestPower',          nodetype=YamlNodeType.MAPPING).map_to(optim.criterion.MMDTestPower)
yamlRegistry.register(yaml_tag=u'!criterion.HSICTestPower',         nodetype=YamlNodeType.MAPPING).map_to(optim.criterion.HSICTestPower)
yamlRegistry.register(yaml_tag=u'!criterion.MutualInfoLowerBound',  nodetype=YamlNodeType.MAPPING).map_to(optim.criterion.MutualInfoLowerBound)
yamlRegistry.register(yaml_tag=u'!criterion.MITestPower',           nodetype=YamlNodeType.MAPPING).map_to(optim.criterion.MITestPower)

# models
yamlRegistry.register(yaml_tag=u'!nn.ModuleList',               nodetype=YamlNodeType.SCALAR).map_to(nn.ModuleList)
yamlRegistry.register(yaml_tag=u'!nn.Sequential',               nodetype=YamlNodeType.SEQUENCE).map_to(nn.Sequential)
yamlRegistry.register(yaml_tag=u'!nn.Embedding',                nodetype=YamlNodeType.MAPPING).map_to(nn.Embedding)
yamlRegistry.register(yaml_tag=u'!nn.Flatten',                  nodetype=YamlNodeType.MAPPING).map_to(nn.Flatten)       # TODO: old code used SCALAR
yamlRegistry.register(yaml_tag=u'!model.Identity',              nodetype=YamlNodeType.SCALAR).map_to(model.base.Identity)
yamlRegistry.register(yaml_tag=u'!model.Neck',                  nodetype=YamlNodeType.MAPPING).map_to(model.base.Neck)
yamlRegistry.register(yaml_tag=u'!model.FeedForward',           nodetype=YamlNodeType.MAPPING).map_to(model.mlp.FeedForward)
yamlRegistry.register(yaml_tag=u'!model.ConvNet',               nodetype=YamlNodeType.MAPPING).map_to(model.cnn.ConvNet)
yamlRegistry.register(yaml_tag=u'!model.ResNet',                nodetype=YamlNodeType.MAPPING).map_to(model.resnet.ResNet)
yamlRegistry.register(yaml_tag=u'!model.Gaussian',              nodetype=YamlNodeType.MAPPING).map_to(model.distribution.Gaussian)
yamlRegistry.register(yaml_tag=u'!model.Dirichlet',             nodetype=YamlNodeType.MAPPING).map_to(model.distribution.Dirichlet)
yamlRegistry.register(yaml_tag=u'!model.TransformerEncoder',    nodetype=YamlNodeType.MAPPING).map_to(model.transformer.TransformerEncoder)
yamlRegistry.register(yaml_tag=u'!model.SelfAttention',         nodetype=YamlNodeType.MAPPING).map_to(model.attention.SelfAttention)

# kernels
yamlRegistry.register(yaml_tag=u'!kernel.Gaussian',         nodetype=YamlNodeType.MAPPING).map_to(kernel.Gaussian)
yamlRegistry.register(yaml_tag=u'!kernel.Linear',           nodetype=YamlNodeType.MAPPING).map_to(kernel.Linear)
yamlRegistry.register(yaml_tag=u'!kernel.DeepKernel',       nodetype=YamlNodeType.MAPPING).map_to(kernel.DeepKernel)
yamlRegistry.register(yaml_tag=u'!kernel.WeightedGaussian', nodetype=YamlNodeType.MAPPING).map_to(kernel.WeightedGaussian)

# mcmc
yamlRegistry.register(yaml_tag=u'!mcmc.MALA',   nodetype=YamlNodeType.MAPPING).map_to(mcmc.MALA)

