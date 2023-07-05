import torch
from .generator import Generator
from .discriminator import Discriminator
from torch import nn


def choose_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return 'cpu'


def init_weights(net, scaling=0.02):
    def init_func(m):
        class_name = m.__class__.__name__
        if hasattr(m, 'weight') and (class_name.find('Conv')) is True:
            torch.nn.init.normal_(m.weight.data, 0.0, scaling)
        elif class_name.find('BatchNorm2d') is True:
            torch.nn.init.normal_(m.weight.data, 1.0, scaling)
            torch.nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def make_network(device, pretrain=False, model_path=None):
    generator = Generator(device, 3, 3).to(device)
    discriminator = Discriminator(6).to(device)
    if not pretrain:
        init_weights(generator, scaling=0.02)
        init_weights(discriminator, scaling=0.02)
    else:
        assert model_path is not None, \
            'no path is provided for pretrained models'
        loader = torch.load(model_path)
        generator.load_state_dict(loader['gen'])
        discriminator.load_state_dict(loader['disc'])
    return generator, discriminator


def combine_param(generator: nn.Module, discriminator: nn.Module):
    g_state = generator.state_dict()
    d_state = discriminator.state_dict()
    return {'gen': g_state, 'disc': d_state}
