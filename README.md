# Lung Cancer Research

This repository contains code, experiments, and results for a research project focused on lung cancer detection and analysis using deep learning techniques. The project is being conducted under the guidance of [Pawan Kumar Singh](https://scholar.google.co.in/citations?user=LctgJHoAAAAJ&hl=en).

## Project Overview

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection and accurate diagnosis are crucial for improving patient outcomes. In this project, we leverage state-of-the-art vision transformer models, specifically attention-based architectures, to analyse medical imaging data for lung cancer research.

## Supervisor

- **Dr. Pawan Kumar Singh**  
  [Google Scholar Profile](https://scholar.google.co.in/citations?user=LctgJHoAAAAJ&hl=en)

## Methods

We utilise the **facebook/deit-base-distilled-patch16-224** attention-based model for image analysis. Vision Transformers (ViT) have shown promising results in various computer vision tasks, and we apply this architecture to our lung cancer dataset to evaluate its effectiveness.

### Key Techniques Used

- **Image Preprocessing Pipeline**:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Bicubic Interpolation (Resizing to 224×224)
  - Gaussian Blurring
  - Otsu's Thresholding
  - Morphological Transformations (Erosion + Dilation)
  - Grayscale Normalization using `AutoImageProcessor` from HuggingFace

- **Model Architectures**:
  - [`facebook/deit-base-distilled-patch16-224`](https://huggingface.co/facebook/deit-base-distilled-patch16-224)

- **Training Enhancements**:
  - Stratified 5-Fold Cross-Validation
  - Learning Rate Scheduling (`CosineAnnealingLR`)
  - Early Stopping with Patience Tracking
  - Reduce LROnPlateau
  - Label Smoothing Regularization
  - DataLoader with multiprocessing (`num_workers`, `pin_memory`)

- **Optimizers Used**:
  - AdamW

- **Metrics**:
  - Accuracy
  - Classification Report
  - ROC Curve and AUC (per class)
  - Confusion Matrix

## Data Efficient Image Transformer(DeiT) Results

### iQOTH-NCCD Dataset
Classification report obtained using the DeiT-based model:
<img width="1000" alt="image" src="https://github.com/user-attachments/assets/f3a95fef-0bd7-4ff5-8681-a237f134e399" />

Highest current accuracy achieved using ViT: **99.54%**

### LIDC-IDRI Dataset

Classification report obtained using the DeiT-based model:

Confusion Matrix obtained using the DeiT-based model:



## Acknowledgements

- This research is conducted under the supervision of Dr. Pawan Kumar Singh.

## License

This project is for academic and research purposes only.
