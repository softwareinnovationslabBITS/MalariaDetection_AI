# H-CAST: Hierarchical Classification Framework with Swin Transformer for Automated Malaria Diagnosis

## Overview

This repository contains the implementation of **H-CAST (Hierarchical Classification and Attention-based Swin Transformer)**, a deep learning framework for automated malaria diagnosis from microscopy image embeddings.

The proposed framework performs hierarchical classification at two clinically relevant levels:

* **Level 1:** Infection Detection (Negative vs Positive)
* **Level 2:** Species Classification (*Plasmodium vivax* vs *Plasmodium falciparum*)

The model combines a Swin Transformer backbone with hierarchical prediction heads and a Tree-Path Consistency mechanism to enforce biologically meaningful predictions across classification levels.

---

## Key Features

* Hierarchical malaria classification
* Swin Transformer backbone with shifted window attention
* Tree-Path Consistency regularization
* Automated infection detection
* Species-level malaria classification
* Comprehensive performance evaluation
* ROC-AUC analysis
* Confusion matrix visualization
* Automated report generation

---

## Model Architecture

### Swin Transformer Backbone

The feature extractor consists of:

* Patch Embedding Layer
* Window-based Multi-Head Self-Attention
* Shifted Window Attention
* Patch Merging Layers
* Global Feature Aggregation

### Hierarchical Classification Heads

The extracted features are processed by two hierarchical prediction branches:

1. Infection Detection Head
2. Species Classification Head

A Tree-Path Consistency layer is incorporated to ensure prediction consistency across the classification hierarchy.

---

## Dataset

The framework is designed for microscopy image analysis of malaria blood smears.

The dataset includes:

* Uninfected blood smear samples
* *Plasmodium vivax* infected samples
* *Plasmodium falciparum* infected samples

Image embeddings are generated prior to training and stored as compressed NumPy files (`.npz`).

---

## Repository Structure

```text
.
├── HCAST_Swin.py
├── HCAST_ViT.py
├── HCAST_3level_RESNET.py
├── SwinTransformer.py
├── ViT.py
├── inceptionv3new.py
├── README.md
├── requirements.txt
└── figures/
```

---

## Evaluation Metrics

The following metrics are reported:

* Accuracy
* Sensitivity
* Specificity
* Precision
* F1-Score
* Balanced Accuracy
* Matthews Correlation Coefficient (MCC)
* Geometric Mean (G-Mean)
* Area Under the ROC Curve (AUC)

---

## Authors

* Ghufran Alam Siddiqui
* Ajay B
* Asish Bera
* Nitika
* Praveen K. Bharti
* Ashis Das
* Tanmaya Mahapatra

