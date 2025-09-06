import os
import cv2
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
import lpips
from sewar.full_ref import uqi, vifp
from skimage.metrics import structural_similarity as ssim
from pytorch_fid import fid_score

lpips_model = lpips.LPIPS(net='alex')
device = torch.device('cpu')


def compute_mse(imageA, imageB):
    err = np.mean((imageA - imageB) ** 2)
    return err

def compute_psnr(imageA, imageB):
    mse = compute_mse(imageA, imageB)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def compute_ssim(imageA, imageB):
    return ssim(imageA, imageB, data_range=imageB.max() - imageB.min())

def compute_ncc(imageA, imageB):
    numerator = np.sum((imageA - np.mean(imageA)) * (imageB - np.mean(imageB)))
    denominator = np.sqrt(np.sum((imageA - np.mean(imageA))**2) * np.sum((imageB - np.mean(imageB))**2))
    return numerator / denominator if denominator != 0 else 0

def compute_lpips(imgA, imgB):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])
    imgA = t(Image.fromarray(imgA)).unsqueeze(0).to(device)
    imgB = t(Image.fromarray(imgB)).unsqueeze(0).to(device)

    with torch.no_grad():
        d = lpips_model(imgA, imgB)
    return d.item()

def evaluate_two_images(image1_path, image2_path):
    original = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    processed = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    if original is None or processed is None:
        print("❌ Không thể đọc một hoặc cả hai ảnh!")
        return
    
    if original.shape != processed.shape:
        print("⚠ Kích thước ảnh không khớp — đang resize ảnh processed để khớp.")
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    mse_value = compute_mse(original, processed)
    psnr_value = compute_psnr(original, processed)
    ssim_value = compute_ssim(original, processed)
    ncc_value = compute_ncc(original, processed)
    
    print("📊 Kết quả so sánh hai ảnh:")
    print(f"MSE: {mse_value:.4f}")
    print(f"PSNR: {psnr_value:.4f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"NCC: {ncc_value:.4f}")
    


# Đánh giá toàn bộ folder
def evaluate_image_folder(folder_original, folder_processed, output_csv="results.csv"):
    files = [f for f in os.listdir(folder_original) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    psnr_list, ssim_list, mse_list, ncc_list = [], [], [], []
    lpips_list, uqi_list, vif_list = [], [], []

    os.makedirs("temp_fid/real", exist_ok=True)
    os.makedirs("temp_fid/fake", exist_ok=True)

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "MSE", "PSNR", "SSIM", "NCC", "LPIPS", "UQI", "VIF"])

        for filename in tqdm(files, desc="🔍 Đang đánh giá ảnh"):
            path_orig = os.path.join(folder_original, filename)
            path_proc = os.path.join(folder_processed, filename)

            if not os.path.exists(path_proc):
                print(f"⚠ Không tìm thấy ảnh: {filename}")
                continue

            imgA = cv2.imread(path_orig)
            imgB = cv2.imread(path_proc)

            if imgA is None or imgB is None:
                print(f"❌ Không đọc được ảnh: {filename}")
                continue

            if imgA.shape != imgB.shape:
                imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))

            grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

            mse_val = compute_mse(grayA, grayB)
            psnr_val = compute_psnr(grayA, grayB)
            ssim_val = compute_ssim(grayA, grayB)
            ncc_val = compute_ncc(grayA, grayB)
            lpips_val = compute_lpips(imgA, imgB)
            uqi_val = uqi(grayA, grayB)
            vif_val = vifp(grayA, grayB)

            writer.writerow([filename, mse_val, psnr_val, ssim_val, ncc_val, lpips_val, uqi_val, vif_val])

            mse_list.append(mse_val)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            ncc_list.append(ncc_val)
            lpips_list.append(lpips_val)
            uqi_list.append(uqi_val)
            vif_list.append(vif_val)

            cv2.imwrite(os.path.join("temp_fid/real", filename), imgA)
            cv2.imwrite(os.path.join("temp_fid/fake", filename), imgB)

    print(f"Avg PSNR : {np.mean(psnr_list):.2f} dB")
    print(f"Avg SSIM : {np.mean(ssim_list):.4f}")
    print(f"Avg MSE  : {np.mean(mse_list):.2f}")
    print(f"Avg NCC  : {np.mean(ncc_list):.4f}")
    print(f"Avg LPIPS: {np.mean(lpips_list):.4f}")
    print(f"Avg UQI  : {np.mean(uqi_list):.4f}")
    print(f"Avg VIF  : {np.mean(vif_list):.4f}")
    
evaluate_image_folder(
    folder_original = r"C:\Users\phaml\Downloads\test\datasets\test",
    folder_processed = r"C:\Users\phaml\Downloads\test\output",
    output_csv="FEGAN1.csv"
)

