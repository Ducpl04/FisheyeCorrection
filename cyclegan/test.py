import os
import torch
from torchvision import transforms
from PIL import Image
from models.generator import ResNetGenerator

# Cấu hình
input_dir = "datasets/test_FE_openCV"
output_dir = "CycleGAN_output"
os.makedirs(output_dir, exist_ok=True)

img_size = 256
model_path = "checkpoints/G_AB_epoch100.pth"  # Đường dẫn tới checkpoint tốt nhất

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetGenerator().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

# Duyệt qua tất cả ảnh trong thư mục input
image_names = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

for name in image_names:
    img_path = os.path.join(input_dir, name)
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)

    out_img = out.squeeze().cpu().clamp(0, 1)
    out_img_pil = transforms.ToPILImage()(out_img)
    out_img_pil.save(os.path.join(output_dir, name))

    print(f"Đã xử lý {name}")
