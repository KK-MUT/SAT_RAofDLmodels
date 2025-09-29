# Robustness Assessment of Deep Learning Models for Satellite Imagery 🛰️🛡️

## Project Overview

This project implements a systematic, multi-stage protocol to assess the **robustness** of four representative deep learning classification architectures (ResNet-50, RegNetY-8GF, Swin-T, and ShuffleNetV2-x1.0) against intentional, small input perturbations, specifically **L-infinity adversarial attacks**.

The study uses the AERO-RSI dataset for aircraft classification to evaluate how model effectiveness degrades under attacks like **FGSM**, **R+FGSM**, and **PGD**, combining quantitative metrics (Robust Accuracy, AUC, $\epsilon_{50}$, $\epsilon_{90}$) with Explainable AI (XAI) techniques (Saliency, Grad-CAM).

## Data

The project utilizes the [**AERO-RSI v1.0**](https://www.kaggle.com/datasets/faryalaurooj/aero-remote-sensing-images-aero-rsi-v-1-0) dataset, which contains satellite images with annotated objects.

* **Content:** Aircraft silhouettes cropped using bounding boxes.
* **Resolution:** All input images are rescaled to a fixed resolution of **$224 \times 224$ pixels**.
* **Split:** The dataset is divided into three parts: Training (70%), Validation (15%), and Test (15%).

---

## Repository Structure and Scripts

The project workflow is managed by four primary scripts tailored for the robustness study.

| Script | Description | Purpose (Stage) |
| :--- | :--- | :--- |
| `train.py` | **Model Training.** Implements the training protocol (Adam, Cross-Entropy Loss, Transfer Learning with frozen backbone, Early Stopping) for the four key models (ResNet-50, RegNetY-8GF, Swin-T, ShuffleNetV2-x1.0). Saves model weights (`.pth`) and metadata (`_meta.json`). | Training |
| `eval_models.py` | **Clean Data Evaluation.** Assesses the fundamental performance of all trained models on the unperturbed test set ($\epsilon=0$) by calculating Accuracy, Precision, Recall, F1-macro, and Confusion Matrices. | Baseline Evaluation |
| `robust_eval.py` | **Adversarial Robustness Evaluation.** Implements FGSM and PGD ($L_\infty$) attacks. Measures Robust Accuracy (RA) curves, $\epsilon_{50}$/$\epsilon_{90}$ thresholds, and Normalized AUC across models, including optional transferability checks. | Robustness Assessment |
| `xai_visualize.py` | **Explainability (XAI) Visualization.** Generates **Saliency Maps** and **Grad-CAM(++)** visualizations for single images to understand *why* the models make decisions and *where* adversarial perturbations target the representation. | Analysis & XAI |
---
