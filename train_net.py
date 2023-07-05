from config import cfg
from lib.utils import choose_device
from lib.trainer import Trainer
        

def main():
    device = choose_device()
    
    trainer = Trainer(cfg, device, begin_iter=0)
    
    for epoch in range(cfg.epochs):
        trainer.train()
        if epoch % cfg.eval_iter == 0:
            trainer.eval()

        if epoch % cfg.save_iter == 0:
            trainer.save_model(cfg.save_dir)
        

if __name__ == '__main__':
    main()
