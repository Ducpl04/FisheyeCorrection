import cv2
import numpy as np
import os

def apply_fisheye_opencv(image, k1=0.5, k2=-0.3, k3=0.0, k4=0.0):
    h, w = image.shape[:2]
    K = np.array([[w/2, 0, w/2],
                  [0, h/2, h/2],
                  [0, 0, 1]], dtype=np.float64)
    D = np.array([[k1], [k2], [k3], [k4]]) 

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
    fisheye_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return fisheye_img

input_folder = 'datasets/New folder'
output_folder = 'datasets/129'

os.makedirs(output_folder, exist_ok=True)
k1, k2, k3, k4 = 0.5, -0.4, 0.0, 0.0
image_files = [f for f in os.listdir(input_folder)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for filename in image_files:
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    img = cv2.imread(input_path)
    fisheye_img = apply_fisheye_opencv(img, k1, k2, k3, k4)
    cv2.imwrite(output_path, fisheye_img)