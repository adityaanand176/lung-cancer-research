# Lung Cancer Research

This repository contains code, experiments, and results for a research project focused on lung cancer detection and analysis using deep learning techniques. The project is being conducted under the guidance of [Pawan Kumar Singh](https://scholar.google.co.in/citations?user=LctgJHoAAAAJ&hl=en).

## Project Overview

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection and accurate diagnosis are crucial for improving patient outcomes. In this project, we leverage state-of-the-art vision transformer models, specifically attention-based architectures, to analyse medical imaging data for lung cancer research.

## Supervisor

- **Dr. Pawan Kumar Singh**  
  [Google Scholar Profile](https://scholar.google.co.in/citations?user=LctgJHoAAAAJ&hl=en)

## Methods

We utilise the **ViT-base-patch-16-224** attention-based model for image analysis. Vision Transformers (ViT) have shown promising results in various computer vision tasks, and we apply this architecture to our lung cancer dataset to evaluate its effectiveness.

In addition, we now experiment with **ResNet-18**, a convolutional neural network architecture, to compare and benchmark traditional CNN-based models against transformers for medical imaging.

### Key Techniques Used

- **Image Preprocessing Pipeline**:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Bicubic Interpolation (Resizing to 224x224)
  - Gaussian Blurring
  - Otsu Thresholding
  - Erosion and Dilation
  - Normalization (using `AutoImageProcessor` stats for DeiT)

- **Model Architectures**:
  - Vision Transformer (DeiT Base Distilled Patch16)
  - ResNet-18 with modified classification head

- **Training Enhancements**:
  - Stratified 5-Fold Cross-Validation
  - Class Weighting for Imbalanced Data
  - Learning Rate Scheduling (`CosineAnnealingLR`)
  - Early Stopping with Patience Tracking
  - Label Smoothing Regularization
  - DataLoader with multiprocessing (`num_workers`, `pin_memory`)

- **Optimizers Used**:
  - `AdamW` for DeiT and ResNet

- **Metrics**:
  - Accuracy
  - Classification Report
  - Confusion Matrix

## Results

### ViT-base-patch-16-224 based Results

Classification report obtained using the ViT-base-patch-16-224 attention-based model:

![image](https://github.com/user-attachments/assets/5581bf9c-5d57-4026-bba6-cdd248e768d7)


Highest current accuracy achieved using ViT: **99.14%**

### DeiT-based Results
Classification report obtained using the DeiT-based model:
<img width="1000" alt="image" src="https://github.com/user-attachments/assets/f3a95fef-0bd7-4ff5-8681-a237f134e399" />


## Acknowledgements

- This research is conducted under the supervision of Dr. Pawan Kumar Singh.
- Thanks to all contributors and collaborators.

## License

This project is for academic and research purposes only.
