import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 0), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 0), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super(PatchDiscriminator, self).__init__()
        
        def conv_layer(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            conv_layer(in_channels, ndf, normalize=False), 
            conv_layer(ndf, ndf * 2),                       
            conv_layer(ndf * 2, ndf * 4),                 
            conv_layer(ndf * 4, ndf * 8),               
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1) 
        )

    def forward(self, x):
        return self.model(x)