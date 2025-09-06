import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        def down_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = down_block(in_channels, 64)
        self.enc2 = down_block(64, 128)
        self.enc3 = down_block(128, 256)
        self.enc4 = down_block(256, 512)

        self.dec1 = up_block(512, 256)
        self.dec2 = up_block(512, 128)
        self.dec3 = up_block(256, 64)   
        self.final = nn.ConvTranspose2d(128, out_channels, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        e1 = self.enc1(x)   # (B, 64, 128, 128)
        e2 = self.enc2(e1)  # (B, 128, 64, 64)
        e3 = self.enc3(e2)  # (B, 256, 32, 32)
        e4 = self.enc4(e3)  # (B, 512, 16, 16)

        d1 = self.dec1(e4)                              # (B, 256, 32, 32)
        d2 = self.dec2(torch.cat([d1, e3], dim=1))      # (B, 128, 64, 64)
        d3 = self.dec3(torch.cat([d2, e2], dim=1))      # (B, 64, 128, 128)
        out = self.final(torch.cat([d3, e1], dim=1))    # (B, 3, 256, 256)
        return self.tanh(out)
