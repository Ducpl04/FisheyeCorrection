import torch.nn as nn

def ResBlock(ch):
    return nn.Sequential(
        nn.Conv2d(ch, ch, 3, 1, 1),
        nn.InstanceNorm2d(ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch, ch, 3, 1, 1),
        nn.InstanceNorm2d(ch)
    )

class ResNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, n_blocks=6):
        super().__init__()
        model = [
            nn.Conv2d(in_ch, 64, 7, 1, 3), nn.InstanceNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True),
        ]
        for _ in range(n_blocks):
            model += [ResBlock(256)]
        model += [
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.InstanceNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.InstanceNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, out_ch, 7, 1, 3), nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
