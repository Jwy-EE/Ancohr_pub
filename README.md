# ANCHOR: Adaptive Network based on Cascaded Harmonic Offset Routing

This repository contains the official implementation code for the paper "ANCHOR: Adaptive Network based on Cascaded Harmonic Offset Routing".

Project hosted at: [https://github.com/Jwy-EE/Ancohr_pub](https://github.com/Jwy-EE/Ancohr_pub)

## 👥 Authors

This project is jointly developed by:

- **Wangye Jiang** (Suzhou University of Technology)
- **Haoming Yang** (Jinling Institute of Technology)
- **Jingya Zhang** (Suzhou University of Technology) - Corresponding Author: zhangjy0611@163.com

> **Note:** Wangye Jiang and Haoming Yang contributed equally to this work.

## 📖 Introduction

ANCHOR is an explicit-implicit coupled backbone network designed for time-series analysis. Its core contributions are:

- **Macroscopic Physical Navigation:** Utilizes RFFT to extract explicit dominant periods, serving as spatial priors for underlying deformable operators.
- **Microscopic Continuous Phase Compensation:** Introduces an infinitely differentiable 1D Gaussian Radial Basis Function (Gaussian RBF) interpolator. This completely resolves the gradient truncation issues inherent in traditional bilinear interpolation, achieving high-precision sub-pixel phase alignment.
- **Frequency-Spatial Adaptive Routing:** Dynamically balances the extraction weights between strong low-frequency features and weak high-frequency perturbations through orthogonal channel partitioning.

## 🚀 Quick Start

### ⚠️ Plugin-Style Open-Source Declaration

To avoid redundant engineering code and ensure absolute fairness in baseline evaluation, ANCHOR is designed as a native academic plugin. Please embed it into the corresponding official benchmark libraries to run.

### 1. Forecasting & Classification and other Tasks (via Time-Series-Library)

Short-term forecasting, long-term forecasting, and time-series classification tasks strictly rely on the official TSlib framework.

**Deployment Steps:**

1. Clone the official TSlib repository.
2. Place `ANCHOR.py` from this repository into the TSlib's `models/` directory; place `dcnv4_1D_Gaussian.py` and `fft_seek.py` into the `layers/` directory.
3. Register `'ANCHOR': ANCHOR` within the `model_dict` of the TSlib execution engine (e.g., `run.py`).

**Execution Example (M4 Monthly Short-Term Forecasting):**

```bash
python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model ANCHOR \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --num_workers 0
```

### 2. UCR Anomaly Detection Task (via KAN-AD)

When evaluating anomaly detection on the UCR archive, to avoid the evaluation bias introduced by the traditional Point Adjustment (PA) strategy, our reproduction strictly adopts the standardized unsupervised reconstruction pipeline from KAN-AD.

**Reproduction Instructions:**

Please clone the [https://github.com/issaccv/KAN-AD](https://github.com/issaccv/KAN-AD) repository and embed the ANCHOR backbone network into its evaluation framework to fully reproduce the core detection metrics (Event F1, Delay F1, and AUPRC) reported in our paper.

## 📄 License

This project is open-sourced under the MIT License. The code is intended for academic research purposes only.