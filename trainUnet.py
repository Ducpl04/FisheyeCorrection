import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils, models
from models.generator import UNetGenerator
from models.discriminator import Discriminator
import os
import cv2
import numpy as np
import pytorch_msssim  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 50
lr = 2e-4
batch_size = 16

resume = True
checkpoint_epoch = 40
start_epoch = 0

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.resize = resize
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        if self.resize:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = nn.functional.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        return self.criterion(x_vgg, y_vgg)

criterion_percep = VGGPerceptualLoss().to(device)
criterion_ssim = pytorch_msssim.SSIM(data_range=1.0, size_average=True, channel=3)

class FisheyeDataset(torch.utils.data.Dataset):
    def __init__(self, fisheye_folder, equirectangular_folder):
        self.fisheye_paths = [os.path.join(fisheye_folder, f) for f in os.listdir(fisheye_folder)]
        self.equirectangular_paths = [os.path.join(equirectangular_folder, f) for f in os.listdir(equirectangular_folder)]
        self.fisheye_paths.sort()
        self.equirectangular_paths.sort()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        fisheye = cv2.imread(self.fisheye_paths[idx])
        fisheye = cv2.cvtColor(fisheye, cv2.COLOR_BGR2RGB)

        equirectangular = cv2.imread(self.equirectangular_paths[idx])
        equirectangular = cv2.cvtColor(equirectangular, cv2.COLOR_BGR2RGB)

        return self.transform(fisheye), self.transform(equirectangular)

    def __len__(self):
        return len(self.fisheye_paths)

fisheye_folder = "datasets/train_FE_openCV"
equirectangular_folder = "datasets/train"

dataset = FisheyeDataset(fisheye_folder, equirectangular_folder)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

G = UNetGenerator().to(device)
D = Discriminator().to(device)

criterion_adv = nn.BCELoss()
criterion_recon = nn.L1Loss()
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

if resume:
    G.load_state_dict(torch.load(f"checkpoint/G_epoch_{checkpoint_epoch}.pth"))
    D.load_state_dict(torch.load(f"checkpoint/D_epoch_{checkpoint_epoch}.pth"))
    opt_G.load_state_dict(torch.load(f"checkpoint/opt_G_epoch_{checkpoint_epoch}.pth"))
    opt_D.load_state_dict(torch.load(f"checkpoint/opt_D_epoch_{checkpoint_epoch}.pth"))
    start_epoch = checkpoint_epoch
    print(f"Resumed from epoch {checkpoint_epoch}")

print(torch.cuda.is_available())
print(device)

#Training
for epoch in range(start_epoch, epochs):
    G.train()
    D.train()

    for i, (fisheye, gt_eq) in enumerate(loader):
        fisheye, gt_eq = fisheye.to(device), gt_eq.to(device)

        D.zero_grad()
        real_pred = D(gt_eq)
        fake_eq = G(fisheye)
        fake_pred = D(fake_eq.detach())

        real_loss = criterion_adv(real_pred, torch.ones_like(real_pred, device=device))
        fake_loss = criterion_adv(fake_pred, torch.zeros_like(fake_pred, device=device))
        loss_D = (real_loss + fake_loss) / 2
        loss_D.backward()
        opt_D.step()

        G.zero_grad()
        fake_pred = D(fake_eq)
        loss_G_adv = criterion_adv(fake_pred, torch.ones_like(fake_pred, device=device))
        loss_G_recon = criterion_recon(fake_eq, gt_eq)
        loss_G_percep = criterion_percep(fake_eq, gt_eq)
        loss_G_ssim = 1.0 - criterion_ssim(fake_eq, gt_eq)

        loss_G = 1.0 * loss_G_adv + 0.5 * loss_G_recon + 5.0 * loss_G_percep + 1.0 * loss_G_ssim
        loss_G.backward()
        opt_G.step()

        if i % 100 == 0:
            print(f"[{epoch}/{epochs}] [{i}/{len(loader)}] D_loss: {loss_D.item():.4f}, G_loss: {loss_G.item():.4f}")

    print(f"Epoch {epoch}/{epochs} completed: D_loss = {loss_D.item():.4f}, G_loss = {loss_G.item():.4f}")

    if epoch % 5 == 0:
        os.makedirs("output", exist_ok=True)
        utils.save_image(fake_eq, f"output/fake_{epoch}.png", normalize=True)

    if epoch % 10 == 0:
        os.makedirs("checkpoint", exist_ok=True)
        torch.save(G.state_dict(), f"checkpoint/G_epoch_{epoch}.pth")
        torch.save(D.state_dict(), f"checkpoint/D_epoch_{epoch}.pth")
        torch.save(opt_G.state_dict(), f"checkpoint/opt_G_epoch_{epoch}.pth")
        torch.save(opt_D.state_dict(), f"checkpoint/opt_D_epoch_{epoch}.pth")
