# Loan Approval Prediction Website

A machine-learning web application that predicts **Approved** or **Rejected** loan status using the supplied loan dataset.

## Dataset
The supplied dataset contains **4,269 records** and these input features:
- No. of dependents
- Education
- Self employed
- Annual income
- Loan amount
- Loan term
- CIBIL score

Target: `loan_status` (`Approved` / `Rejected`).

## Model
- Algorithm: Random Forest Classifier
- Train/test split: 80/20
- Validation accuracy on the held-out test set: **98.13%**
- Categorical features are one-hot encoded.
- Numeric features are standardized.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000/`.

## Important
This project is intended as an educational ML demonstration. The prediction is not an actual bank/lender decision and should not be used as the sole basis for approving or rejecting a real loan.
