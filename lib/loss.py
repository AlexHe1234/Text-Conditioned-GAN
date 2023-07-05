import torch
from torch import nn


class GeneratorLoss(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.adversarial_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        self.lambd = lambd
    
    def forward(self, generated_img, target_img, discriminator_output, real_target):
        gen_loss = self.adversarial_loss(discriminator_output, real_target)
        l1_1 = self.l1_loss(generated_img, target_img)
        gen_total_loss = gen_loss + self.lambd * l1_1
        return gen_total_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.adversarial_loss = nn.BCELoss()
        
    def forward(self, output, fake_target):
        return self.adversarial_loss(output, fake_target)
