# Lending Club Loan Interest Rate Prediction

## Mission
To predict the interest rate assigned to a loan applicant based on their
financial profile and loan characteristics — helping lenders make
data-driven, fair pricing decisions and helping borrowers understand what
factors drive their assigned rate.

## Dataset
**Name:** Lending Club Loan Data  
**Source:** Kaggle — https://www.kaggle.com/datasets/wordsforthewise/lending-club  
**File:** loan.csv (50,000 rows loaded for RAM stability)  
**License:** CC0 Public Domain  

### Columns
| Column                      | Type    | Description |
|-----------------------------|---------|-------------|
| loan_amnt                   | float   | Requested loan amount |
| term                        | str→int | Loan term in months (36 or 60) |
| installment                 | float   | Monthly payment amount |
| sub_grade                   | str→int | Lending Club internal risk sub-grade |
| emp_length                  | str→int | Years of employment |
| home_ownership              | str→int | Renter / Owner / Mortgage |
| annual_inc                  | float   | Annual income of applicant |
| verification_status         | str→int | Income verification status |
| purpose                     | str→int | Reason for the loan |
| dti                         | float   | Debt-to-income ratio |
| delinq_2yrs                 | int     | Delinquencies in past 2 years |
| inq_last_6mths              | int     | Credit inquiries in last 6 months |
| open_acc                    | int     | Number of open credit lines |
| pub_rec                     | int     | Number of derogatory public records |
| revol_bal                   | float   | Total revolving balance |
| revol_util                  | float   | Revolving line utilization rate |
| total_acc                   | int     | Total number of credit lines |
| **int_rate**                | float   | **Target** — interest rate assigned (%) |

### Columns Dropped
- **Leaky** (known only after loan issuance): `funded_amnt`, `loan_status`, `total_pymnt`, etc.
- **Identifiers**: `id`, `member_id`, `url`, `zip_code`, `addr_state`, etc.
- **>30% missing**: dropped automatically during feature engineering
- **Near-zero variance**: dropped automatically (>99% same value)

## Files
| File | Description |
|------|-------------|
| `loan_regression.py`      | Full pipeline: EDA → feature engineering → training → evaluation → saving |
| `predict_loan.py`         | Task 2 prediction script using the saved best model |
| `best_loan_model.pkl`     | Saved Random Forest model (lowest MSE / best R²) |
| `scaler.pkl`              | Fitted StandardScaler — preserves training data scaling for new predictions |
| `viz1_interest_rate_distribution.png` | Histogram of loan interest rates |
| `viz2_top_correlations.png`           | Top 10 features correlated with interest rate |
| `viz3_correlation_heatmap.png`        | Heatmap of top feature relationships |
| `viz4_subgrade_vs_intrate.png`        | Boxplot of interest rate by sub-grade |
| `viz5_loss_curve.png`                 | Train vs Test MSE loss curve (SGD) |
| `viz6_scatter_before_after_linear.png`| Before/after scatter with linear fit line |
| `viz7_rf_actual_vs_predicted.png`     | Random Forest actual vs predicted |

## Models Trained
- **SGDRegressor** — Linear regression via gradient descent (loss curve tracked per epoch)
- **DecisionTreeRegressor** — max_depth=10
- **RandomForestRegressor** — 100 estimators, max_depth=10 ← **Best Model (lowest MSE)**

## How to Run (Google Colab)
```python
# 1. Upload loan.csv to /content/sample_data/ then run the full pipeline
exec(open('loan_regression.py').read())

# 2. Task 2 — run the prediction script
exec(open('predict_loan.py').read())
```

## How to Run (Local)
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# Run full training pipeline
python loan_regression.py

# Make a prediction (Task 2)
python predict_loan.py
```
