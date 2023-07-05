from yacs.config import CfgNode as CN
import os


cfg = CN()

# global
cfg.exp_name = 'test001'
cfg.epochs = 200
cfg.lr = 2e4
cfg.batch_size = 32
cfg.eval_iter = 1
cfg.save_iter = 1

# path and directory
cfg.dataset_dir = './archive/train_data'
cfg.save_dir = './result'
cfg.pretrained = False
cfg.model_dir = None

# params
cfg.lambd = 100


if cfg.pretrained:
    assert cfg.model_dir is not None, \
        'if use pretrained model, model_dir cannot be empty'
if not os.path.exists(cfg.save_dir):
    os.mkdir(cfg.save_dir) 
