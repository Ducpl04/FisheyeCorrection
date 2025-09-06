import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from models.cyclegan import CycleGAN
from datasets.unpaired_dataset import UnpairedImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CycleGAN().to(device)

adv_loss = nn.MSELoss()
cycle_loss = nn.L1Loss()

optim_G = torch.optim.Adam(
    list(model.G_AB.parameters()) + list(model.G_BA.parameters()), lr=2e-4, betas=(0.5, 0.999)
)
optim_D = torch.optim.Adam(
    list(model.D_A.parameters()) + list(model.D_B.parameters()), lr=2e-4, betas=(0.5, 0.999)
)

dataset = UnpairedImageDataset(
    root_A="datasets/train_FE_openCV",  # Domain A: fisheye
    root_B="datasets/train",  # Domain B: pinhole
    img_size=256
)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

#Training
epochs = 100
for epoch in range(epochs):
    for i, data in enumerate(loader):
        real_A = data['A'].to(device, non_blocking=True)
        real_B = data['B'].to(device, non_blocking=True)

        fake_B = model.G_AB(real_A)
        fake_A = model.G_BA(real_B)

        rec_A = model.G_BA(fake_B)
        rec_B = model.G_AB(fake_A)

        loss_G = adv_loss(model.D_B(fake_B), torch.ones_like(model.D_B(fake_B))) + \
                 adv_loss(model.D_A(fake_A), torch.ones_like(model.D_A(fake_A))) + \
                 0.5 * cycle_loss(rec_A, real_A) + \
                 1 * cycle_loss(rec_B, real_B)

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        for D, real, fake in [(model.D_A, real_A, fake_A), (model.D_B, real_B, fake_B)]:
            pred_real = D(real)
            pred_fake = D(fake.detach())
            loss_D = 0.5 * (adv_loss(pred_real, torch.ones_like(pred_real)) +
                            adv_loss(pred_fake, torch.zeros_like(pred_fake)))
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        if i % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(loader)}] Loss_G: {loss_G.item():.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.G_AB.state_dict(), f"checkpoints/G_AB_epoch{epoch+1}.pth")
    torch.save(model.G_BA.state_dict(), f"checkpoints/G_BA_epoch{epoch+1}.pth")
    torch.save(model.D_A.state_dict(), f"checkpoints/D_A_epoch{epoch+1}.pth")
    torch.save(model.D_B.state_dict(), f"checkpoints/D_B_epoch{epoch+1}.pth")
