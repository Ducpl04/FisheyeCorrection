import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        ch = 64
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.InstanceNorm2d(ch * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.InstanceNorm2d(ch * 4), nn.LeakyReLU(0.2),
            nn.Conv2d(ch * 4, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.net(x)
