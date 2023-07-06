from torch.utils.data import DataLoader
import importlib


def make_dataloader(cfg):
    assert cfg.dataset_module is not None, \
        'must specify dataset module path'
    dataset = importlib.import_module('.' + cfg.dataset_module, package='lib.dataset').make_dataset(cfg)
    return  DataLoader(dataset, cfg.batch_size, shuffle=True)
