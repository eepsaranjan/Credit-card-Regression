# Credit Card Balance Prediction — Regression Project

Predicting customer credit card balance from demographic and financial attributes using regularized linear regression, with a focus on handling multicollinearity between credit limit and credit rating.

## 📌 Overview

This project builds and compares three regression models — **Linear Regression**, **Lasso**, and **Ridge** — to predict a customer's revolving credit card balance (`Balance`) from attributes like income, credit limit, credit rating, number of cards, age, education, gender, student status, marital status, and ethnicity.

Understanding what drives a customer's carried balance is directly useful to a card issuer: it's a proxy for interest revenue, a signal for credit risk exposure, and a lever for marketing segmentation (e.g. targeting high-balance customers differently from "transactor" customers who pay off their card monthly).

## 📊 Dataset

- **Source:** Credit dataset (400 customers, 12 columns) — a classic dataset used for regression benchmarking (ISLR).
- **Target variable:** `Balance`
- **Features:** `Income`, `Limit`, `Rating`, `Cards`, `Age`, `Education`, `Gender`, `Student`, `Married`, `Ethnicity`
- **Missing values:** `Income` (24 rows), `Age` (20 rows), `Student` (16 rows), `Married` (16 rows) — imputed with median/mean (numeric) and mode (categorical).

## 🔍 Key Findings

- `Limit` and `Rating` are by far the strongest predictors of `Balance` (correlation ≈ 0.86 each) — and are almost perfectly collinear with **each other** (r ≈ 0.997).
- `Income` has a moderate positive relationship with `Balance`.
- `Student` status shows a visible secondary effect — students tend to carry higher balances.
- `Age`, `Education`, `Cards`, `Gender`, `Married`, and `Ethnicity` show little to no predictive value.
- `Balance` is right-skewed with a spike at zero, representing "transactor" customers who don't carry a balance.

## 🛠️ Tech Stack

- Python, pandas, NumPy
- scikit-learn (`LinearRegression`, `Lasso`, `Ridge`, `GridSearchCV`, `StandardScaler`)
- matplotlib, seaborn (EDA & visualization)

## 🧪 Methodology

1. **Data Cleaning:** Dropped non-informative `ID` column; imputed missing values.
2. **EDA:** Univariate → Bivariate → Multivariate analysis across 15+ charts, including a correlation heatmap and pairplot.
3. **Feature Engineering:** One-hot encoded categorical variables (`drop_first=True`); checked multicollinearity via correlation and VIF; scaled numeric features.
4. **Modeling:** Trained Linear Regression, Lasso, and Ridge, each tuned via `GridSearchCV` with 5-fold cross-validation.
5. **Evaluation:** Compared models using R², MAE, and RMSE (MAPE was found unreliable for this dataset — see Results below).

## 📈 Results

| Model | CV R² | Test R² | Test MAE | Test RMSE |
|---|---|---|---|---|
| Linear Regression | 0.9459 | 0.7340 | 131.81 | 210.80 |
| Lasso (α=1) | 0.9460 | 0.7341 | 131.85 | 210.77 |
| **Ridge (α=0.1)** | **0.9460** | **0.7347** | **131.67** | **210.54** |

**Final model: Ridge Regression.** All three models perform similarly, but Ridge is preferred because its L2 regularization directly addresses the strong collinearity between `Limit` and `Rating`, stabilizing coefficients in a way plain Linear Regression can't, and without arbitrarily dropping one of the two correlated features the way Lasso can.

> ⚠️ **Note on MAPE:** Mean Absolute Percentage Error was excluded from model comparison. Because a subset of customers have a `Balance` of zero (or very close to it), MAPE's division-by-actual-value calculation produces meaningless, extremely large values. MAE and RMSE (in dollar terms) were used instead as the primary error metrics.

> ⚠️ **Note on generalization:** There's a gap between training R² (~0.95) and test R² (~0.73) across all three models, suggesting some overfitting and/or a test set too small (80 rows) to fully represent the data. Future work with more data or repeated cross-validation would help confirm and address this.

## 🚀 Getting Started

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
jupyter notebook Project_Regression.ipynb
```

### Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## 📁 Project Structure

```
├── Credit_Data.csv              # Dataset
├── Project_Regression.ipynb     # Full analysis notebook (EDA, modeling, evaluation)
└── README.md
```

## 💡 Business Impact

The final model helps a bank:
- Forecast expected interest revenue from customer balances
- Flag customers likely to build large balances for proactive credit-limit or risk management
- Identify segments (e.g. students) that may benefit from differentiated product offers
- Treat `Limit` and `Rating` as a single underwriting signal rather than two independent ones, given their near-perfect collinearity

## ✍️ Author

Eepsa Ranjan

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
