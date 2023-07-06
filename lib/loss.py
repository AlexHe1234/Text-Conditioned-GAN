from torch import nn
import clip
import torch


# class ClipModel(nn.Module):
#     def __init__(self, device):
#         super().__init__()
#         self.device = device
#         self.model, self.preprocess = clip.load('ViT-B/32', device)

#     @torch.no_grad()
#     def forward(self, x, c):
#         x = self.preprocess(x).unsqueeze(0).to(self.device)
#         return self.model(x, c)


class GeneratorLoss(nn.Module):
    def __init__(self, lambd, clip_weight, device):
        super().__init__()
        self.adversarial_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        self.lambd = lambd
        self.clip_weight = clip_weight
        # self.clip_model = ClipModel(device)
    
    def forward(self, generated_img, target_img, discriminator_output, real_target, c):
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
