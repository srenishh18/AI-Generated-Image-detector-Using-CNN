# 🧠 AI-Generated Image Detector (PyTorch CNN)

An end-to-end Machine Learning binary classifier and web application designed to detect whether an image is **Real** or **AI-Generated**, powered by a Custom 5-Layer Convolutional Neural Network (CNN) in PyTorch.

---

## 🚀 Key Features

* **High Performance**: **98.99% Test Accuracy** and **0.9994 ROC-AUC Score** on an unseen held-out test dataset (894 images).
* **Robust Preprocessing & Augmentation**: Prevents shortcut learning by resizing all inputs to 128x128 and applying random flips, rotations, and color jitter.
* **Class Balance Weighting**: Uses weighted Binary Cross-Entropy loss (`pos_weight = 1.545`) to compensate for the 60/40 Real/AI dataset split.
* **Overfitting Protection**: Employs **Batch Normalization**, **Dropout (`p=0.3`)**, and **Weight Decay (`1e-4`)** regularization.
* **Dedicated Terminal Metrics Suite**: Run `python evaluate_metrics.py` to print formatted Test Accuracy, Precision, Recall, F1, ROC-AUC, Loss, and Confusion Matrix.
* **Ultra-Minimalist Web UI**: Flask backend (`app.py`) providing instant real-time predictions (`http://127.0.0.1:5000`) with confidence percentage meters.

---

## 📁 Directory & File Overview

```text
d:\AI Image detector/
├── dataset/                  # Dataset directory
│   ├── AI/                   # 3,506 AI-generated images
│   └── Real/                 # 5,422 Real photography images
├── train_model.py            # CNN Model architecture, data loaders, & training engine
├── evaluate_metrics.py       # Terminal evaluation script displaying metrics & confusion matrix
├── app.py                    # Flask Web Server providing real-time inference API
├── best_model.pth            # Saved PyTorch trained model weights checkpoint
├── metrics.json              # Saved evaluation metrics JSON report
├── templates/
│   └── index.html            # Minimalist single-page web application UI
├── static/
│   └── style.css             # Glassmorphism dark-mode CSS styling
├── README.md                 # Project Overview & Quick Start guide
└── PROJECT.md                # Comprehensive Technical Deep-Dive & Q&A
```

---

## 📊 Dataset Overview

* **Total Images**: **8,928 images**
* **AI-Generated Images**: **3,506 images** (~39.3%)
* **Real Images**: **5,422 images** (~60.7%)
* **Data Splits**: 
  * 🟢 **Training Set (80%)**: **7,142 images**
  * 🟡 **Validation Set (10%)**: **893 images**
  * 🔴 **Unseen Test Set (10%)**: **894 images**

---

## 🛠️ Quick Start Commands

### 1. Train the CNN Model (One-Time Execution)
```bash
python train_model.py custom 3 128
```

### 2. View Evaluation Metrics in Terminal
```bash
python evaluate_metrics.py
```

### 3. Launch the Web UI Server
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser!
