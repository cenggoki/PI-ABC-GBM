# Steel Fatigue Strength Prediction – Desktop Application
**A Data Science & Metallurgy Integrated Approach for Rotating Bending Fatigue Prediction**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
# PI-ABC-GBM
Desktop application for predicting rotating-bending fatigue strength (10⁷ cycles) in steels. Hybrid approach: GBM optimized via Artificial Bee Colony (ABC). Integrates metallurgical feature engineering (CE_IIW, HI, PSI). PyQt5 GUI offers EDA, SHAP interpretability, Nested CV, Bootstrap uncertainty, and batch CSV predictions.



##  Overview

This project presents a comprehensive desktop application developed. It aims to predict the **rotating bending fatigue strength at 10⁷ cycles** of various steel grades using a hybrid approach that combines advanced machine learning (Gradient Boosting Machine - GBM) with a nature-inspired metaheuristic algorithm (Artificial Bee Colony - ABC).

Unlike traditional black-box models, this application integrates domain-specific metallurgical knowledge by automatically engineering derived features (such as CE_IIW, Pcm, Hardenability Index, and Diffusion Carbon Index) from raw process and composition data. Built with Python and PyQt, it offers a fully interactive GUI for data exploration, model training, statistical validation, and interpretability.

## Key Features

- **Interactive Data Loading:** Support for CSV files with robust encoding handling (UTF-8, Latin-1, etc.).
- **Metallurgical Feature Engineering:** Automatic calculation of Carbon Equivalent (CE_IIW), Pcm, Hardenability Index (HI), Penetration Index (PSI), and Diffusion Index (DCI) based on raw process parameters.
- **Comprehensive EDA & Feature Selection:** Correlation matrices, boxplots, VIF (Variance Inflation Factor) analysis, and SHAP-based feature importance filtering.
- **Advanced Model Optimization:** Hyperparameter optimization of the **Gradient Boosting Machine (GBM)** using the **Artificial Bee Colony (ABC)** algorithm to achieve superior predictive accuracy.
- **Rigorous Statistical Validation:**
    - **Nested Cross-Validation** (5-fold outer, 3-fold inner) for unbiased generalization error estimation.
    - **Friedman Test & Nemenyi Post-hoc** analysis with Critical Difference (CD) diagrams for statistical model comparison.
    - **Bootstrap Uncertainty Analysis** to quantify prediction intervals and model reliability.
- **Model Interpretability:** Built-in SHAP (SHapley Additive exPlanations) summary and dependence plots, along with Partial Dependence Plots (PDP) to explain feature effects.
- **User-Friendly GUI:** Seven-tab PyQt interface guiding the user step-by-step from raw data to final prediction.

## Dataset Structure

The application is designed for the provided dataset (`Fatigue_Dataset_for_Steel.csv`), which contains 437 samples with 24 input features (covering normalizing, hardening, carburizing, tempering parameters, chemical composition, and inclusion ratings) and 1 target variable (`Fatigue` in MPa).

## Installation & Setup

Follow these steps to set up the environment and run the application.

- **Make sure Python is installed. Python 3.8 or later :**
  cmd> python --version

- **Create a virtual environment :**
  cmd> cd project directory

  cmd> python -m venv fatigue_env

  cmd> fatigue_env\Scripts\activate.bat

- **Install the necessary packages :**
   pip install -r requirements.txt

- **Start the program :**
  python SteelFatigue38_Eng.py

### 1. Clone the Repository
```bash
git clone https://github.com/cenggoki/PI-ABC-GBM.git
cd steel-fatigue-prediction
