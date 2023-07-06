import clip
import torch
from torch import nn
import numpy as np


# class ClipTextNet(nn.Module):
#     def __init__(self, device):
#         super(ClipTextNet, self).__init__()
#         self.model, self.preprocess = clip.load('ViT-B/32', device)
#         self.linear = nn.Linear(512, 512)
#         self.bn = nn.BatchNorm1d(512)

#     def forward(self, x):
#         with torch.no_grad():
#             features = self.model.encode_te


class EmbedLayer(nn.Module):
    def __init__(self, in_channel, mid_channel=512):
        super().__init__()
        self.layer1 = nn.Linear(in_channel, mid_channel)
        self.layer2 = nn.Linear(mid_channel, mid_channel)

    def forward(self, x):
        x = self.layer1(x)
        return self.layer2(x)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        down_conv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                              stride=2, padding=1, bias=False)
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = norm_layer(inner_nc)
        up_relu = nn.ReLU(True)
        up_norm = norm_layer(outer_nc)
        if outermost:
            up_conv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [down_conv]
            up = [up_relu, up_conv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            up_conv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [down_relu, down_conv]
            up = [up_relu, up_conv, up_norm]
            model = down + up
        else:
            up_conv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [down_relu, down_conv, down_norm]
            up = [up_relu, up_conv, up_norm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetInnerBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if input_nc is None:
            input_nc = outer_nc
        down_conv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                              stride=2, padding=1, bias=False)
        down_relu = nn.LeakyReLU(0.2, True)
        up_relu = nn.ReLU(True)
        up_norm = norm_layer(outer_nc)
        up_conv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
        down = [down_relu, down_conv]
        up = [up_relu, up_conv, up_norm]
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)
        self.condition = None

    def forward(self, x):
        down = self.down(x)
        down = down + self.condition.reshape(down.shape)
        return torch.cat([x, self.up(down)], 1)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, nf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        # defining the innermost block of layers
        # Conv2d/ConvTranspose2d -> BatchNorm -> Activation
        self.unet_block_inner = UnetInnerBlock(nf*8, nf*8, input_nc=None, norm_layer=norm_layer)
        # defining intermediate blocks
        unet_block = UnetSkipConnectionBlock(nf*8, nf*8, input_nc=None, submodule=self.unet_block_inner,
                                             norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf*8, nf*8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf*8, nf*8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_dropout=use_dropout)

        unet_block = UnetSkipConnectionBlock(nf*4, nf*8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nf*2, nf*4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nf, nf*2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)

        self.model = UnetSkipConnectionBlock(output_nc, nf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True, norm_layer=norm_layer)

    def forward(self, input, c):
        self.unet_block_inner.condition = c
        return self.model(input)


class Generator(nn.Module):
    def __init__(self, device, input_nc, output_nc, classes=6, nf=64):
        super().__init__()
        self.device = device
        self.condition_model = EmbedLayer(classes, 512)
        self.unet = UnetGenerator(input_nc, output_nc, nf)
        
    def forward(self, x, c: torch.Tensor):
        # with torch.no_grad():
        #     conditions = [clip.tokenize(string).to(self.device) for string in strings]
        # conditions = [self.condition_model(condition)[0][None, ...] for condition in conditions]
        # conditions = torch.cat(conditions, dim=0)[..., None, None]
        c = c[..., None, None]
        out = self.unet(x, c)
        exit()
        return out, c


if __name__ == '__main__':
    G = Generator(torch.device('cpu'))
    x = torch.ones((2, 3, 256, 256)).to('cpu')
    G(x, "feeling sexy")
