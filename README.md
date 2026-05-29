# 💳 Credit Card Default Prediction

A machine learning project to predict whether a credit card client will default on their next payment, using demographic, financial, and payment history data.

---

## 📌 Overview

Credit card default is a major concern for financial institutions. This project builds and evaluates multiple predictive models on a dataset of 30,000 credit card clients from Taiwan to classify whether a client will default on their payment next month.

The analysis covers the full data science pipeline — from exploratory data analysis and feature engineering to model training, evaluation, and comparison.

---

## 📁 Repository Structure

```
default_credit_card_predictive_model/
│
├── default of credit card clients.csv                       # Dataset
└── default_of_credit_card_clients_predictive_models.ipynb  # Main notebook
```

---

## 📊 Dataset

**File:** `default of credit card clients.csv`

The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) and contains information on **30,000 credit card clients** in Taiwan (April–September 2005).

### Features

| Feature | Description |
|---|---|
| `LIMIT_BAL` | Credit limit (in NT dollars) |
| `SEX` | Gender (1 = Male, 2 = Female) |
| `EDUCATION` | Education level (1 = Graduate, 2 = University, 3 = High school, 4 = Others) |
| `MARRIAGE` | Marital status (1 = Married, 2 = Single, 3 = Others) |
| `AGE` | Age in years |
| `PAY_0` – `PAY_6` | Repayment status for past 6 months (-1 = On time, 1–9 = Months delayed) |
| `BILL_AMT1` – `BILL_AMT6` | Bill statement amount for past 6 months (NT dollars) |
| `PAY_AMT1` – `PAY_AMT6` | Amount paid for past 6 months (NT dollars) |
| `default.payment.next.month` | **Target variable** (1 = Default, 0 = No default) |

---

## 🧪 Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis of demographic features (age, gender, education, marital status)
- Class imbalance inspection (default vs. non-default)
- Correlation analysis between features and the target variable
- Visualization of payment history trends

### 2. Data Preprocessing
- Handling missing or erroneous values
- Feature encoding and scaling
- Train/test split

### 3. Model Building & Evaluation

Multiple classification models are trained and compared:

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Tree-based interpretable model |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting / XGBoost | Boosted ensemble model |
| Support Vector Machine (SVM) | Margin-based classifier |
| K-Nearest Neighbors (KNN) | Distance-based classifier |

### 4. Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Environment:** Jupyter Notebook
- **Libraries:**
  - `pandas`, `numpy` — Data manipulation
  - `matplotlib`, `seaborn` — Data visualization
  - `scikit-learn` — Machine learning models and evaluation
  - `xgboost` *(if applicable)* — Gradient boosting

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### Running the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/anitakokane/default_credit_card_predictive_model.git
   cd default_credit_card_predictive_model
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open `default_of_credit_card_clients_predictive_models.ipynb` and run all cells.

---

## 📈 Key Insights

- **Class imbalance:** Approximately 22% of clients defaulted, meaning the dataset is imbalanced — this is accounted for during model evaluation.
- **Payment history** (`PAY_0` – `PAY_6`) is among the strongest predictors of default.
- **Credit limit** (`LIMIT_BAL`) shows a negative correlation with default probability — higher limits tend to mean lower default risk.
- Ensemble models (Random Forest, Gradient Boosting) generally outperform simpler classifiers on this dataset.

---

