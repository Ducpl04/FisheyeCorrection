import torch
import os
import cv2
import numpy as np
from torchvision import transforms, utils
from models.generator import UNetGenerator
from models.generator import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16

class FisheyeTestDataset(torch.utils.data.Dataset):
    def __init__(self, fisheye_folder):
        self.fisheye_paths = sorted([os.path.join(fisheye_folder, f) for f in os.listdir(fisheye_folder)])
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        fisheye_path = self.fisheye_paths[idx]
        fisheye = cv2.imread(fisheye_path)
        fisheye = cv2.cvtColor(fisheye, cv2.COLOR_BGR2RGB)
        return self.transform(fisheye), fisheye_path

    def __len__(self):
        return len(self.fisheye_paths)

test_folder = "datasets/test_FE_openCV"
dataset = FisheyeTestDataset(test_folder)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

G = Generator().to(device)
checkpoint_path = "checkpoint/G_epoch_40.pth"
G.load_state_dict(torch.load(checkpoint_path))
G.eval()

with torch.no_grad():
    for i, (fisheye, fisheye_paths) in enumerate(test_loader):
        fisheye = fisheye.to(device)
        fake_eq = G(fisheye)

        for idx, path in enumerate(fisheye_paths):
            fake_eq_image = fake_eq[idx].cpu().detach()
            filename = os.path.basename(path)
            utils.save_image(fake_eq_image, f"output/{filename}", normalize=True)

        if i % 10 == 0:
            print(f"Processed {i * batch_size}/{len(dataset)} samples.")
