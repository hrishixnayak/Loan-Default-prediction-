🏦 Loan Default Prediction & Credit Risk Scoring
📌 Project Overview

In real-world lending, banks don’t just ask “Will this borrower default?”
They ask:

How risky is this borrower?

Why is the risk high or low?

What decision should we take based on that risk?

This project builds an end-to-end Machine Learning system that predicts the Probability of Default (PD) for loan applicants, converts it into a Credit Risk Score (0–100), and categorizes borrowers into Low / Medium / High Risk — closely mirroring how modern fintech and banking systems operate.

🎯 Problem Statement

Given borrower demographic, financial, and loan-related information, predict whether a borrower is likely to default and quantify that risk in a business-friendly way.

🧠 Solution Approach

Instead of a simple binary classifier, the system is designed as a risk modeling pipeline:

Predict Probability of Default (PD)

Convert PD into a Credit Risk Score

Assign Risk Categories

Enable business decision simulation

📊 Dataset

Source: Kaggle (Loan Default Dataset)

Type: Structured tabular data

Target Variable: Default (0 = No Default, 1 = Default)

Key Features

Borrower details: Age, Income, Education, Employment Type

Credit information: Credit Score, DTI Ratio, Number of Credit Lines

Loan details: Loan Amount, Interest Rate, Loan Term, Loan Purpose

Financial stability indicators

⚙️ Feature Engineering

Domain-inspired features were created to reflect real credit risk logic:

Loan_to_Income Ratio

Monthly EMI (approximation)

EMI_to_Income Ratio

Employment Stability Index

These features significantly improve model interpretability and performance.

🏗️ System Architecture
Raw Loan Data
      ↓
Data Cleaning & Preprocessing
      ↓
Feature Engineering
      ↓
Train-Test Split
      ↓
ML Model (Logistic Regression / Random Forest)
      ↓
Probability of Default (PD)
      ↓
Credit Risk Score (0–100)
      ↓
Risk Category (Low / Medium / High)
      ↓
Business Decision Simulation
🤖 Models Used

Logistic Regression (baseline, interpretable)

Random Forest Classifier (non-linear risk modeling)

Class imbalance is handled using class weights, reflecting real-world default distributions.

📈 Evaluation Metrics

Accuracy is not reliable for imbalanced financial data.
The project focuses on:

ROC-AUC Score

Precision & Recall

Default rate by risk category

Business impact metrics

🧮 Credit Risk Scoring Logic
Credit Risk Score = Probability of Default × 100
Risk Buckets
Score Range	Risk Category
0 – 30	Low Risk
31 – 60	Medium Risk
61 – 100	High Risk
🧪 Business Simulation

A simple lending decision simulation demonstrates:

Lower default rates among approved (Low Risk) borrowers

Higher default concentration in rejected (High Risk) group

This bridges the gap between ML predictions and business decisions.

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

Matplotlib / Seaborn

Jupyter Notebook

📁 Project Structure
loan-default-prediction/
│
├── data/
│   └── loan_data.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_risk_scoring.ipynb
│
├── README.md
└── requirements.txt
✅ Key Learnings

Credit risk modeling is more about decision systems than pure prediction

Feature engineering grounded in domain logic is critical

ML outputs must be translated into business-friendly metrics

Preventing data leakage is essential for trustworthy models

🚀 Future Improvements

SHAP-based explainability

Threshold optimization for loan approval policies

Streamlit dashboard for real-time predictions

Model deployment as an API

📌 Final Note

This project was built as part of Day 25 of a 30-Day Machine Learning Challenge, with a focus on real-world applicability over toy examples.
