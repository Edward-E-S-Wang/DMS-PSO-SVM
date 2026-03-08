# Dynamic Multi-Swarm Particle Swarm Optimization Support Vector Machine

A Python implementation of **Dynamic Multi-Swarm Particle Swarm Optimization Support Vector Machine (DMS-PSO-SVM)** for binary classification.

This repository provides a simple and reproducible example of using **DMS-PSO** to optimize the hyperparameters of an **RBF-kernel SVM**, with model evaluation based on **AUC, accuracy, sensitivity, and specificity**.

The current script is designed for tabular data stored in a CSV file, where:

- the **first column** is the binary label
- the **remaining columns** are the input features

---

## Overview

This project implements a DMS-PSO-based hyperparameter search strategy for SVM classification.

The workflow includes:

1. Loading raw data from a CSV file
2. Converting values to real-valued floats
3. Splitting data into training and validation sets
4. Optimizing `C` and `gamma` for an RBF-SVM using DMS-PSO
5. Training the final SVM model with the best parameters
6. Evaluating model performance on the validation set

The optimization objective is based on **5-fold cross-validated ROC-AUC** on the training set.

---

## Features

- Binary classification with **SVM (RBF kernel)**
- Hyperparameter optimization using **Dynamic Multi-Swarm PSO**
- Stratified train/validation split
- Cross-validation-based fitness evaluation
- Validation metrics:
  - AUC
  - Accuracy
  - Sensitivity
  - Specificity
- Supports numeric data stored in CSV format
- Includes conversion of complex-like strings to real values

---
**Citation:**
If you use this code, this implementation strategy, or a modified version of it in academic work, please cite the original article:
- Li Y, Zhang D, Wang Y, et al. *Construction of an oligometastatic prediction model for nasopharyngeal carcinoma patients based on pathomics features and dynamic multi-swarm particle swarm optimization support vector machine*. Frontiers in Oncology, 2025. DOI: 10.3389/fonc.2025.1589919
## File Structure
---
```text
.
├── main.py         # main training and evaluation script
└── README.md       # project documentation
