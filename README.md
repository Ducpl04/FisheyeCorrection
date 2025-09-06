# Fisheye Image Rectification using Deep Learning

##  Overview
This project focuses on **fisheye image rectification** using deep learning techniques.  
The goal is to transform distorted fisheye images into rectified pinhole-like images for computer vision tasks such as recognition, detection, and scene understanding.

##  Implemented Methods
We explored multiple deep learning models for fisheye correction:
- **U-Net**: Encoder–decoder architecture for supervised image-to-image translation.
- **CycleGAN**: Unsupervised domain adaptation between fisheye and rectified domains.
- **FE-GAN**: GAN-based framework with perceptual loss for improved visual quality.

##  Results
- Models successfully restored structural details from fisheye images.
- GAN-based models (CycleGAN, FE-GAN) produced more realistic corrections.
- U-Net achieved sharper edge recovery with paired supervision.
- Performance evaluated with PSNR, SSIM, and LPIPS.

##  Future Work
- Improve generalization to real-world fisheye images with varied distortion parameters.
- Extend dataset with real fisheye–rectified pairs.
- Explore hybrid approaches combining geometry-based calibration with deep learning.

##  Authors
Pham Le Quy Duc

