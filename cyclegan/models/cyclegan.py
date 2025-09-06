import torch
import torch.nn as nn
from models.generator import ResNetGenerator
from models.discriminator import PatchDiscriminator

class CycleGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.G_AB = ResNetGenerator()
        self.G_BA = ResNetGenerator()
        self.D_A = PatchDiscriminator()
        self.D_B = PatchDiscriminator()

    def set_requires_grad(self, nets, requires_grad=False):
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad
