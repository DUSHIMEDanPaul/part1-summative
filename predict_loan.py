
import joblib
import numpy as np
import pandas as pd

def make_loan_prediction(input_data: pd.DataFrame):
    model  = joblib.load('best_loan_model.pkl')
    scaler = joblib.load('scaler.pkl')
    scaled_input = scaler.transform(input_data)
    predictions  = model.predict(scaled_input)
    return predictions

# ── Single-row prediction from test set ──
single_row        = X_test_raw.iloc[[0]]
single_row_scaled = scaler.transform(single_row)
single_pred       = rf_reg.predict(single_row_scaled)[0]
single_actual     = float(y_test.iloc[0])

print(f"Input features:\n{single_row.to_string()}")
print(f"\nActual Interest Rate   : {single_actual:.2f}%")
print(f"Predicted Interest Rate: {single_pred:.2f}%")
print(f"Difference             : {abs(single_actual - single_pred):.2f}%")

# ── Standalone prediction function demo ──
sample = pd.DataFrame([{
    'loan_amnt': 15000, 'term': 36, 'installment': 450.5,
    'sub_grade': 10, 'emp_length': 5, 'home_ownership': 2,
    'annual_inc': 65000, 'verification_status': 1, 'purpose': 3,
    'dti': 18.5, 'delinq_2yrs': 0, 'inq_last_6mths': 1,
    'open_acc': 10, 'pub_rec': 0, 'revol_bal': 8000,
    'revol_util': 45.0, 'total_acc': 22,
    'collections_12_mths_ex_med': 0, 'application_type': 0,
    'tot_coll_amt': 0, 'tot_cur_bal': 50000,
    'open_acc_6m': 1, 'open_act_il': 3, 'open_il_12m': 1,
    'open_il_24m': 2, 'mths_since_rcnt_il': 6, 'total_bal_il': 12000,
    'il_util': 55.0, 'open_rv_12m': 2, 'open_rv_24m': 3,
    'max_bal_bc': 3000, 'all_util': 48.0, 'total_rev_hi_lim': 20000,
    'inq_fi': 1, 'total_cu_tl': 2, 'inq_last_12m': 3,
    'acc_open_past_24mths': 4, 'avg_cur_bal': 5000,
    'bc_open_to_buy': 4000, 'bc_util': 40.0,
    'mo_sin_old_il_acct': 60, 'mo_sin_old_rev_tl_op': 120,
    'mo_sin_rcnt_rev_tl_op': 3, 'mo_sin_rcnt_tl': 3,
    'mort_acc': 1, 'mths_since_recent_bc': 5,
    'mths_since_recent_inq': 4, 'num_accts_ever_120_pd': 0,
    'num_actv_bc_tl': 3, 'num_actv_rev_tl': 4, 'num_bc_sats': 4,
    'num_bc_tl': 6, 'num_il_tl': 8, 'num_op_rev_tl': 7,
    'num_rev_accts': 10, 'num_rev_tl_bal_gt_0': 4, 'num_sats': 10,
    'num_tl_90g_dpd_24m': 0, 'num_tl_op_past_12m': 3,
    'pct_tl_nvr_dlq': 92.0, 'percent_bc_gt_75': 25.0,
    'pub_rec_bankruptcies': 0, 'tot_hi_cred_lim': 75000,
    'total_bal_ex_mort': 20000, 'total_bc_limit': 10000,
    'total_il_high_credit_limit': 25000
}])

predicted_rate = make_loan_prediction(sample)
print(f"\nStandalone Prediction — Predicted Interest Rate: {predicted_rate[0]:.2f}%")
