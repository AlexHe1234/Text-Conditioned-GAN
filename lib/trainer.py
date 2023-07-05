import torch
from .utils import make_network, combine_param
from .dataset import make_dataloader
from .loss import GeneratorLoss, DiscriminatorLoss
import os
import torchvision
from torch.utils.tensorboard import SummaryWriter


class Trainer(object):
    def __init__(self, cfg, device, begin_iter=0):
        self.dataloader = make_dataloader(cfg)
        self.gen, self.disc = make_network(device, cfg.pretrained, cfg.model_dir)
        self.g_loss = GeneratorLoss(cfg.lambd)
        self.d_loss = DiscriminatorLoss()
        self.g_optimizer = torch.optim.Adam(self.gen.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.disc.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
        self.device = device
        self.iter = begin_iter
        self.writer_input = SummaryWriter(f"./runs/{exp_name}/input image")
        self.writer_real = SummaryWriter(f"./runs/{exp_name}/real image")
        self.writer_generate = SummaryWriter(f"./runs/{exp_name}/generated image")
    
    def train(self):
        self.gen.train()
        self.disc.train()
        for _, img in enumerate(self.dataloader):
            input_image, target_image = img[0][0], img[0][1]
            input_img = input_image.to(self.device)
            target_img = target_image.to(self.device)
            
            real_target = torch.ones(input_img.shape[0], 1, 30, 30, requires_grad=True).to(self.device)
            fake_target = torch.zeros(input_img.shape[0], 1, 30, 30, requires_grad=True).to(self.device)

            generated_image = self.gen(input_img)

            disc_input_fake = torch.cat((input_img, generated_image), 1)
            D_fake = self.disc(disc_input_fake)
            D_fake_loss = self.d_loss(D_fake, fake_target)
            disc_input_real = torch.cat((input_img, target_img), 1)
            D_real = self.disc(disc_input_real)
            D_real_loss = self.d_loss(D_real, real_target)
            D_total_loss = (D_real_loss + D_fake_loss) / 2

            self.d_optimizer.zero_grad()
            D_total_loss.backward(retain_graph=True)
            self.d_optimizer.step()

            discriminator_output = self.disc(disc_input_fake)
            G_loss = self.g_loss(generated_image, target_img, discriminator_output, real_target)

            self.g_optimizer.zero_grad()
            G_loss.backward(retain_graph=True)
            self.g_optimizer.step()
            self.iter += 1
            
    def save_model(self, savedir):
        state_dict = combine_param(self.gen, self.disc)
        torch.save(state_dict, os.path.join('../', savedir, f'model_{self.iter}.pth'))

    # TODO implemented this and call log_tensorboard
    @torch.no_grad()
    def evaluate(self):
        pass
    
    @torch.no_grad()
    def log_tensorboard(self, input_image, target_image, generated_image, step):
        tb_input = input_image.reshape(-1, 3, 256, 256)
        tb_real = target_image.reshape(-1, 3, 256, 256)
        tb_generate = generated_image.reshape(-1, 3, 256, 256)
        img_grid_input = torchvision.utils.make_grid(tb_input, normalize=True)
        img_grid_real = torchvision.utils.make_grid(tb_real, normalize=True)
        img_grid_generate = torchvision.utils.make_grid(tb_generate, normalize=True)
        self.write_input.add_image("Input Images", img_grid_input, global_step=step)
        self.write_real.add_image("Real(Target) Images", img_grid_real, global_step=step)
        self.write_generate.add_image("Generated Images", img_grid_generate, global_step=step)
        