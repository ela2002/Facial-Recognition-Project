# Facial Recognition using Feature Fusion (SIFT + HOG), PCA, SVM and KNN

## 📌 Project Overview

This project aims to develop a **robust facial recognition system** by fusing two types of visual features — **SIFT** and **HOG** — followed by **dimensionality reduction using PCA**, and **classification using SVM and KNN**.

The system is evaluated using the **FEI face database**, and results are presented using accuracy scores, visualizations of predictions, PCA projection, and confusion matrices.

---

## 🧠 Objectives

- Extract rich and complementary facial features using **SIFT** and **HOG**.
- Reduce dimensionality with **Principal Component Analysis (PCA)**.
- Classify faces using **Support Vector Machines (SVM)** and **K-Nearest Neighbors (KNN)**.
- Combine the strengths of both classifiers to improve recognition performance.

---

## 📂 Dataset

- **Dataset**: [FEI Face Database](https://fei.edu.br/~cet/facedatabase.html)
- **Images**: 2800 total (200 individuals × 14 images each)
- **Image Size**: 640×480 pixels (converted to grayscale and resized)
- **Content**: Various facial expressions and head poses under different lighting conditions

---

## 🔧 Technologies Used

- Python 3.x
- OpenCV (`cv2`)
- Scikit-learn (`sklearn`)
- Matplotlib / Seaborn
- NumPy
- PIL (Pillow)

---

## 🧪 Pipeline

1. **Data Loading & Preprocessing**
   - Load images from FEI dataset
   - Convert to grayscale and resize

2. **Feature Extraction**
   - Detect and extract **SIFT** keypoints and descriptors
   - Compute **HOG** features
   - Concatenate both features

3. **Dimensionality Reduction**
   - Apply **PCA** to reduce high-dimensional fused features

4. **Classification**
   - Train and test **SVM** and **KNN** models
   - Evaluate performance separately and in combination

5. **Evaluation & Visualization**
   - Confusion matrices (SVM, KNN)
   - Accuracy scores
   - PCA projection (first two components)
   - Display predicted labels on sample images
