import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class UnpairedImageDataset(Dataset):
    def __init__(self, root_A, root_B, img_size=256):
        self.paths_A = sorted([os.path.join(root_A, f) for f in os.listdir(root_A)])
        self.paths_B = sorted([os.path.join(root_B, f) for f in os.listdir(root_B)])
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return max(len(self.paths_A), len(self.paths_B))

    def __getitem__(self, idx):
        img_A = Image.open(self.paths_A[idx % len(self.paths_A)]).convert("RGB")
        img_B = Image.open(random.choice(self.paths_B)).convert("RGB")
        return {
            'A': self.transform(img_A),
            'B': self.transform(img_B)
        }
