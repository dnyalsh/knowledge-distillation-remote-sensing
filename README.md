# 🛰 Knowledge Distillation for Remote Sensing Image Classification

MSc thesis project — University of Genoa (Supervisor: Prof. Gabriele Moser)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-success)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-99.05%25-brightgreen)

---

## 🏆 Results

### Teacher Model (RegNetY-032)

| Dataset | Classes | Teacher Accuracy |
|---|---|---|
| UC-Merced | 21 | **94.10%** |
| AID | 30 | **95.74%** |
| NWPU-RESISC45 | 45 | **94.27%** |
| Optimal-31 | 31 | **95.70%** |

### Student Models after Knowledge Distillation (DKD)

| Dataset | Teacher (RegNetY-032) | Student CNN (MobileNetV2) | Student ViT (DeiT) |
|---|---|---|---|
| UC-Merced | 94.10% | **99.05%** | 98.57% |
| AID | 95.74% | **96.80%** | 95.90% |
| NWPU-RESISC45 | 94.27% | **96.00%** | 95.43% |
| Optimal-31 | 95.70% | **98.11%** | 97.62% |

> DKD outperformed classical KD in **all 8** dataset × model combinations, with gains up to **+2.76%** on NWPU-RESISC45.
> Remarkably, student models **surpass the teacher** on all 4 datasets — the core strength of DKD.

---

## 🔍 Overview

Aerial and satellite image classification is critical for land cover mapping, disaster monitoring, and urban planning. Large deep learning models achieve high accuracy but are too computationally heavy for edge deployment.

This project applies **Knowledge Distillation (KD)** to compress large models into lightweight, deployable alternatives while preserving — and in fact **exceeding** — teacher accuracy.

### Architecture

```
Teacher Model
RegNetY-032 (large, high-accuracy)
        │
        ▼  Decoupled Knowledge Distillation (DKD)
   ┌────┴────┐
   │         │
MobileNetV2  DeiT
(CNN student) (ViT student)
```

### Key innovation: Decoupled Knowledge Distillation (DKD)

Classical KD transfers knowledge via soft label mimicking. **DKD** decouples the distillation loss into:
- **TCKD** — Target Class Knowledge Distillation
- **NCKD** — Non-Target Class Knowledge Distillation

This allows independent weighting of each component, resulting in consistently better student performance.

---

## 📦 Datasets

| Dataset | Classes | Images | Description |
|---|---|---|---|
| UC-Merced | 21 | 2,100 | US land use aerial imagery |
| AID | 30 | 10,000 | Aerial image dataset |
| NWPU-RESISC45 | 45 | 31,500 | Remote sensing scene classification |
| Optimal-31 | 31 | 1,860 | Optical remote sensing dataset |

---

## 🗂 Repository structure

```
knowledge-distillation-remote-sensing/
├── detection/
│   └── train_net.py              ← COCO object detection training
├── tools/
│   ├── train.py                  ← Main KD training script
│   ├── eval.py                   ← Model evaluation
│   └── train_teacher.py          ← Teacher model training
├── README.md
├── requirements.txt
└── setup.py
```

---

## ⚙️ Setup & usage

```bash
git clone https://github.com/dnyalsh/knowledge-distillation-remote-sensing.git
cd knowledge-distillation-remote-sensing
pip install -r requirements.txt
python setup.py develop

# Train teacher model
python tools/train_teacher.py --train_dir /data/AID/train --val_dir /data/AID/val --num_classes 30 --output Teacher_AID.pt

# Run KD training
python tools/train.py --cfg configs/dkd_config.yaml
```

---

## 🛠 Tech stack

| Tool | Purpose |
|---|---|
| PyTorch | Model training & distillation |
| torchvision | CNN architectures (MobileNetV2, RegNetY) |
| timm | Vision Transformer (DeiT) + RegNetY teacher |
| NumPy, Pandas | Data processing |
| Matplotlib, Seaborn | Results visualisation |

---

## 👤 Author

**Danial Shariati** · [LinkedIn](https://linkedin.com/in/shariatidanial) · shariatidani@gmail.com
University of Genoa — MSc Engineering for Natural Risk Management
