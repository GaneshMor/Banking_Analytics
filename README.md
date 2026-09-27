# Banking Customer Analytics — Streamlit App

Deployable version of the AADM End-Term Project: **Customer Segmentation, Profitability and Default-Risk Analysis for Indian Banking Customers** (Ojas Paradkar, WeSchool PGDM E-Business, 2024–26).

It turns the Colab notebook into a web app with six pages:

| Page | What it does |
|---|---|
| Overview | KPIs, key findings, strategic customer matrix, recommended action per segment |
| Score a customer | Enter one customer → default probability (random forest), risk flag, DTI policy rule, behavioural segment, decision-tree explanation |
| Batch scoring | Upload a CSV of customers → download the scored file |
| Segment explorer | Interactive factor-space scatter and segment comparisons |
| Model performance | Test-set metrics, confusion matrices, ROC, feature importance, policy impact, tree rules |
| Report figures | All 20 report figures and every results table |

## Models (reproduced exactly from the notebook)

| Model | Test result |
|---|---|
| Random forest (default risk) | ROC-AUC 0.936 · accuracy 0.863 · precision 0.599 · recall 0.800 |
| Decision tree (explanation + policy rule) | ROC-AUC 0.923 · recall 0.914 · root rule DTI > 0.733 |
| Factor analysis → K-Means (segments) | k = 4 · silhouette 0.620 · 100% match with notebook segments |

Split: 75/25 grouped by Customer_ID (random_state = 42). Gender and the leakage variables (Default_Probability, Credit_Risk_Category, Revenue, Cost, Profitability_Score) are **not** used. Profitability is **not** scored, because every regression model had test R² ≈ 0.

## Folder structure

```
app.py                  Streamlit app (entry point)
core.py                 Feature engineering + Scorer class (shared by app and training)
train.py                Re-trains all models from data/ → models/
requirements.txt        Runtime packages (used by the host)
requirements-train.txt  Extra packages to re-train / run tests
.streamlit/config.toml  Theme and server settings
Dockerfile              Optional: deploy on Render / Railway / Cloud Run
data/bank_data_clean.csv    Cleaned dataset with segments (3,000 rows)
models/                 Trained models + metadata.json
assets/figs, assets/tables  Report figures and result tables
sample/sample_customers.csv Template for batch scoring
notebook/               Original Colab notebook
tests/test_core.py      Smoke tests
```

---

## Option 1 — Deploy free on Streamlit Community Cloud (recommended)

You need a free GitHub account. No coding needed.

1. **Unzip** the folder on your laptop.
2. Go to **github.com** → click **New repository** → name it `banking-analytics-app` → select **Public** → **Create repository**.
3. On the new repo page click **uploading an existing file**. Drag **everything inside** the unzipped folder (not the folder itself) into the page.
   - Hidden folders like `.streamlit` may not show in Finder/Explorer. That is fine — the app still works without it (you only lose the custom theme). To show them: Mac `Cmd + Shift + .`, Windows *View → Show → Hidden items*.
4. Click **Commit changes**.
5. Go to **share.streamlit.io** → **Sign in with GitHub** → **Create app** → **Deploy a public app from GitHub**.
6. Fill in:
   - Repository: `your-username/banking-analytics-app`
   - Branch: `main`
   - Main file path: `app.py`
   - **Advanced settings → Python version: 3.11** (3.12 or 3.13 also work)
7. Click **Deploy**. First build takes 2–4 minutes. You get a public link like `https://banking-analytics-app.streamlit.app`.

Free-tier apps sleep after a few days without visits; anyone opening the link can wake them up in ~30 seconds.

## Option 2 — Run on your laptop

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501.

## Option 3 — Docker (Render, Railway, Google Cloud Run, Azure)

```bash
docker build -t banking-analytics .
docker run -p 8501:8501 banking-analytics
```
The container respects the `$PORT` variable that most hosts set.

---

## Re-training the models (only if the data changes)

```bash
pip install -r requirements-train.txt
python train.py          # ~1–3 minutes, overwrites models/
python -m pytest -q      # 4 smoke tests
```
`train.py` also accepts the raw `bank_data.csv` if you rename it to `data/bank_data_clean.csv` (missing Loan_Product is recoded to "No Loan"; the segment-agreement check is skipped).

**Important:** `scikit-learn` is pinned to 1.8.0 because the saved models were trained with it. If you change that version, re-run `train.py`.

## Batch-scoring input columns

```
Credit_Score, Credit_Utilization_Ratio, Credit_Limit_INR, Outstanding_Loan_INR, Interest_Rate_Percent,
EMI_Amount_INR, Debt_to_Income_Ratio, Annual_Income_INR, Account_Balance_Before_INR, Customer_Tenure_Months,
Transactions_Last_30_Days, Average_Monthly_Spend_INR, Days_Since_Last_Transaction, Age, Loan_Product,
Primary_Account_Type, Occupation, Customer_Segment, Investment_Value_INR, Portfolio_Return_Percent,
Portfolio_Risk_Score
```
Categorical values must match the training data (e.g. Loan_Product: Business Loan, Education Loan, Home Loan, Personal Loan, Vehicle Loan, No Loan). Unseen categories are treated as the reference level.

## Limitations (state these if asked)

- Dataset is a 3,000-row snapshot; Customer_ID is not a reliable customer key, so rows are treated as customer snapshots.
- Probabilities come from class-weighted models: they rank risk well but overstate the raw 19.8% default rate. Use them for ranking / thresholds, not as calibrated PDs.
- Churn is not modelled (Target_Churn has one class). Fraud is out of scope.
- Academic project — not a production credit-decision system.
