# Eye Disease Classification Using Deep Learning

> Multi-class fundus image classification across 4 eye diseases using 5 pretrained CNN architectures with GradCAM++ interpretability analysis.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [GradCAM++ Analysis](#gradcam-analysis)
- [Key Findings](#key-findings)
- [How to Run](#how-to-run)
- [Requirements](#requirements)

---

## Overview

Early detection of eye diseases is critical for preventing irreversible vision loss. This project builds and evaluates a deep learning pipeline for automated classification of retinal fundus images into four categories: **cataract**, **diabetic retinopathy**, **glaucoma**, and **normal**.

Five state-of-the-art CNN architectures are benchmarked under identical conditions. Beyond accuracy metrics, the project uses **GradCAM++** to produce visual explanations of model decisions — verifying that the model attends to clinically relevant anatomical regions rather than image artefacts.

---

## Dataset

**Source:** [Eye Diseases Classification — Kaggle](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)

| Class | Images |
|-------|--------|
| Cataract | 1,038 |
| Diabetic retinopathy | 1,098 |
| Glaucoma | 1,007 |
| Normal | 1,074 |
| **Total** | **4,217** |

The dataset is well-balanced across classes. All images are colour fundus photographs.

**Preprocessing applied:**
- Black border removal using contour detection (OpenCV)
- Resize to 224 × 224 pixels
- Train / validation / test split: 70% / 15% / 15% (stratified by class, seeded for reproducibility)

---

## Project Structure

```
├── train.py                  # Full training pipeline
├── gradcam_analysis.py    # GradCAM++ interpretability analysis
├── output/
│   ├── models/               # Saved .keras model checkpoints
│   ├── plots/                # Training curves, confusion matrices, ROC curves
│   ├── gradcam/           # GradCAM++ outputs
│   │   ├── *_gradcam_grid.png
│   │   ├── *_failure_cases.png
│   │   ├── *_method_comparison.png
│   │   ├── *_optic_disc.png
│   │   └── *_confidence.pdf
│   ├── results_summary.csv   # All model metrics
│   └── class_indices.json
```

---

## Methodology

### Architecture

Each model follows the same transfer learning structure:

```
Pretrained Base (ImageNet weights, frozen)
        ↓
GlobalAveragePooling2D
        ↓
Dropout(0.3)
        ↓
Dense(256, ReLU)
        ↓
Dense(4, Softmax)
```

### Models Evaluated

| Model | Parameters | ImageNet Top-1 |
|-------|-----------|----------------|
| ResNet50V2 | 25.6M | 80.0% |
| Xception | 22.9M | 79.0% |
| InceptionV3 | 23.9M | 77.9% |
| DenseNet121 | 8.1M | 75.0% |
| NASNetMobile | 5.3M | 74.4% |

### Training Setup

| Parameter | Value |
|-----------|-------|
| Image size | 224 × 224 |
| Batch size | 16 |
| Initial learning rate | 1e-4 |
| Optimiser | Adam |
| Max epochs | 50 |
| Early stopping patience | 10 (monitors val AUC) |
| LR reduction | ReduceLROnPlateau (factor 0.3, patience 5) |
| Distribution | MirroredStrategy (2× GPU) |
| Checkpoint metric | val AUC (maximise) |

### Data Augmentation (training only)

- Random rotation ±20°
- Random zoom ±20%
- Width / height shift ±10%
- Brightness range [0.8, 1.2]
- Horizontal flip

---

## Results

### Model Comparison

| Model | Test Accuracy | Test AUC (OvR) | Macro F1 | Weighted F1 | Best Epoch |
|-------|:---:|:---:|:---:|:---:|:---:|
| **ResNet50V2** | **81.73%** | **0.9550** | **0.8182** | **0.8184** | 23 |
| Xception | 80.63% | 0.9502 | 0.8075 | 0.8079 | 20 |
| InceptionV3 | 77.95% | 0.9427 | 0.7804 | 0.7803 | 17 |
| DenseNet121 | 75.91% | 0.9552 | 0.7575 | 0.7552 | 34 |
| NASNetMobile | 75.75% | 0.9342 | 0.7593 | 0.7585 | 12 |

**ResNet50V2** is the best overall model, achieving **81.73% accuracy** and **0.955 AUC** on the held-out test set.

> Note: DenseNet121 achieves a high AUC (0.955) but poor accuracy (75.9%) due to a recall collapse on diabetic retinopathy (47.3%) — it rarely predicts that class, suggesting it would benefit from class weighting.

### ResNet50V2 — Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| Cataract | 0.8788 | 0.9295 | 0.9034 | 156 |
| Diabetic retinopathy | 0.9348 | 0.7818 | 0.8515 | 165 |
| Glaucoma | 0.7847 | 0.7434 | 0.7635 | 152 |
| Normal | 0.7021 | 0.8148 | 0.7543 | 162 |
| **Macro avg** | **0.8251** | **0.8174** | **0.8182** | **635** |

**Cataract** is the easiest class (F1: 0.903, mean confidence: 0.86).  
**Glaucoma** and **normal** are the hardest (F1: 0.76, 0.75) — consistent with their visual similarity and the clinical difficulty of early-stage glaucoma.

---

## GradCAM++ Analysis

GradCAM++ (Gradient-weighted Class Activation Mapping++) was applied to the best model (ResNet50V2) to produce visual explanations of predictions. Unlike standard GradCAM, GradCAM++ uses second-order gradients to produce **sharper, more spatially precise** activation maps.

### Diabetic Retinopathy — Clinically Meaningful Activations

The model correctly attends to retinal haemorrhages and vessel lesion patterns — the primary clinical indicators of diabetic retinopathy. This is the strongest interpretability result in the project.

### Glaucoma — Optic Disc Analysis

Glaucoma diagnosis depends on the cup-to-disc ratio of the optic disc. A dedicated optic disc crop analysis was performed: the brightest region of the fundus image is detected and cropped, then GradCAM++ is applied to the crop independently. In most cases the model activates near the disc region, though inconsistency across samples explains the lower F1 score.

### Model Confidence by Class (ResNet50V2)

| Class | All samples | Correct predictions only |
|-------|:-----------:|:------------------------:|
| Cataract | 0.86 | 0.91 |
| Diabetic retinopathy | 0.71 | 0.84 |
| Glaucoma | 0.62 | 0.75 |
| Normal | 0.63 | 0.72 |

When the model is **correct**, it is confident across all classes. The lower overall averages for glaucoma and normal are driven by confused cases — the model is not systematically uncertain, it fails on specific hard cases.

### Failure Case Analysis

The 6 most confident wrong predictions were analysed. Key observations:

- **Glaucoma → Cataract (98% confidence):** The heatmap fires on peripheral retina rather than the optic disc. The model never located the relevant anatomical feature.
- **Normal → Glaucoma (95% confidence):** The heatmap correctly focuses on the optic disc — suggesting the normal image contains a disc morphology similar to early-stage glaucoma. This may be a genuine labelling ambiguity.
- **Cataract misclassifications (92%):** Diffuse, unfocused heatmaps — the model was attending to image-level texture rather than the lens, indicating these were low-quality or atypical images.

---

## Key Findings

1. **ResNet50V2 is the best model** for this task under frozen transfer learning. Its skip connections make it robust to the relatively small dataset size (~3K training images).

2. **Diabetic retinopathy heatmaps are clinically valid.** The model attends to haemorrhage and vessel lesion patterns without being trained on any clinical annotations — a result of learning discriminative features from fundus image labels alone.

3. **Glaucoma is the hardest class** across all five models (lowest F1 in every case). This is consistent with clinical literature — glaucoma shares visual characteristics with normal retinas, especially in early stages. Dedicated optic disc preprocessing or a two-stage model (detection + classification) would likely improve this.

4. **DenseNet121 recall collapses on diabetic retinopathy.** Despite a high AUC, DenseNet121 predicts diabetic retinopathy with only 47.3% recall. Class weighting during training would likely fix this.

5. **Training/validation AUC gap < 0.01 for all models.** No overfitting observed — the frozen base + dropout + early stopping strategy was effective.

6. **GradCAM++ produces meaningfully sharper maps than standard GradCAM**, especially for diabetic retinopathy lesion localisation and glaucoma optic disc focus.

---

## How to Run

### 1. Clone the repo and install dependencies

```bash
git clone https://github.com/navidnayyem/Eye-Disease-Classification-Using-Deep-Learning
cd Eye-Disease-Classification-Using-Deep-Learning
pip install -r requirements.txt
```

### 2. Download the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification) and place it at:

```
dataset/
├── cataract/
├── diabetic_retinopathy/
├── glaucoma/
└── normal/
```

### 3. Update paths in config

Edit the `CONFIG` dictionary at the top of `train.py` to point to your dataset and output directories.

### 4. Train all models

```bash
python train.py
```

Training runs all 5 models sequentially. With 2× GPU it takes approximately 3–4 hours total.

### 5. Run GradCAM++ analysis

```bash
python gradcam_analysis.py
```

Outputs are saved to `output/gradcam/`.

---

## Requirements

```
tensorflow>=2.10
opencv-python
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install all with:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn
```

> Tested on Python 3.10, TensorFlow 2.12, CUDA 11.8 with 2× NVIDIA GPUs.

---

## References

- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
- Chattopadhyay et al. (2018). *Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks.* WACV.
- Dataset: [Eye Diseases Classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification) — Kaggle, Guna Venkat Doddi.
