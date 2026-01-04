# Credit Risk Default Prediction – End-to-End Data Science Project

This repository contains a **full end-to-end machine learning pipeline** for **credit risk default prediction**, implemented using production-style architecture and best practices.

The project covers the entire ML lifecycle:
- data ingestion
- validation
- cleaning & feature engineering
- model training
- model evaluation with **MLflow + DagsHub**
- model persistence
- inference via **Flask web application**

---

## 📌 Problem Statement

The goal of this project is to **predict whether a person will default on a loan** based on demographic, financial, and credit-history information.

The target variable:
- `1` → **default**
- `0` → **no default**

This is a **binary classification problem** 

---

## 🧠 Machine Learning Pipeline

### ML Workflow

1. **Data Ingestion**
   - Load raw dataset
   - Store artifacts in a structured directory

2. **Data Validation**
   - Validate schema (column names & data types)
   - Detect missing or unexpected columns

3. **Data Cleaning**
   - Handle missing values
   - Encode categorical variables using one-hot encoding
   - Prepare final modeling dataset

4. **Data Transformation**
   - Train / test split
   - Persist transformed datasets

5. **Model Training**
   - Model: **XGBoost Classifier**
   - Hyperparameters loaded from `params.yaml`
   - Trained model saved using `joblib`

6. **Model Evaluation**
   - Metrics computed on test set
   - Metrics logged to **MLflow**
   - Model registered in **DagsHub Model Registry**

7. **Prediction Pipeline**
   - Load trained model
   - Accept user input
   - Generate prediction

## ⚙️ Configuration-Driven Design

The project is fully configuration-driven:

- `config.yaml` → paths & artifact locations
- `schema.yaml` → column definitions & target column
- `params.yaml` → model hyperparameters

This allows:
- easy experimentation
- reproducibility
- separation of code and configuration

---

## 🤖 Model Details

- **Algorithm**: XGBoost Classifier
- **Objective**: Binary classification
- **Target**: Credit default (0 / 1)
- **Features**:
  - numerical features (age, income, loan amount, interest rate, etc.)
  - categorical features encoded as dummy variables
    - loan intent
    - loan grade
    - home ownership

---

## 📊 Model Evaluation & Experiment Tracking

- **MLflow** is used for:
  - parameter logging
  - metric logging
  - model versioning

- **DagsHub** is used as:
  - MLflow Tracking Server
  - Model Registry backend

Tracked metrics include:
- AUC
- Accuracy
- Precision
- Recall
- F1-score
- KS statistic
- Gini coefficient

## 🌐 Web Application (Flask)

A simple **Flask web application** allows:
- manual input of features
- prediction using the trained model
- human-readable output:
  - *“Customer is likely to default”*
  - *“Customer is not likely to default”*

Run the app:
```bash
python app.py

Then open:
```bash
http://localhost:8080
