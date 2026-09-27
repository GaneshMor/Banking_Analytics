"""Shared feature engineering and scoring logic (used by train.py and app.py).

Mirrors the preprocessing in Banking_Analytics_Project.ipynb so that the deployed
models see exactly the same inputs they were trained on.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "bank_data_clean.csv"

# ---- Default-risk model inputs (notebook section: Problem C) -----------------
DEF_NUM = ["Credit_Score", "Credit_Utilization_Ratio", "Credit_Limit_INR", "Outstanding_Loan_INR",
           "Interest_Rate_Percent", "EMI_Amount_INR", "Debt_to_Income_Ratio", "Annual_Income_INR",
           "Account_Balance_Before_INR", "Customer_Tenure_Months", "Transactions_Last_30_Days",
           "Average_Monthly_Spend_INR", "Days_Since_Last_Transaction", "Age"]
# categorical column -> reference level dropped in the notebook
DEF_CAT = {"Loan_Product": "No Loan", "Primary_Account_Type": "Savings",
           "Occupation": "Salaried", "Customer_Segment": "Retail"}

# ---- Segmentation inputs (factor analysis -> K-Means) ------------------------
FA_VARS = ["Outstanding_Loan_INR", "EMI_Amount_INR", "Interest_Rate_Percent", "Debt_to_Income_Ratio",
           "log_Investment_Value_INR", "Portfolio_Return_Percent", "Portfolio_Risk_Score"]

# Everything a user must supply to score one customer
INPUT_COLUMNS = DEF_NUM + list(DEF_CAT) + ["Investment_Value_INR", "Portfolio_Return_Percent",
                                           "Portfolio_Risk_Score"]

SEGMENT_ACTIONS = {
    "Transactional-only": "Low risk. Cross-sell pre-screened credit card / personal loan; focus on complaint reduction.",
    "Debt-free Investors": "Low risk, high wealth. Wealth-management upsell; eligible for pre-approved secured credit.",
    "Pure Borrowers": "High risk (~39% default). DTI-tiered underwriting, monthly early-warning score, utilisation trigger.",
    "Leveraged Investors": "Highest risk (~41% default) but hold investments. Offer secured restructuring / loan-against-securities.",
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw or pre-cleaned records into the modelling frame."""
    d = df.copy()
    d["Loan_Product"] = d["Loan_Product"].fillna("No Loan")
    d.loc[d["Loan_Product"].astype(str).str.strip().isin(["", "nan", "None"]), "Loan_Product"] = "No Loan"
    if "Target_Default" in d and "Default" not in d:
        d["Default"] = (d["Target_Default"] == "Yes").astype(int)
    d["log_Investment_Value_INR"] = np.log1p(d["Investment_Value_INR"].astype(float))
    return d


def _prefix(col: str) -> str:
    return col.split("_")[0][:4]          # same naming rule as the notebook's dummies()


def tree_features(d: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """One-hot + numeric frame for the tree models (raw units, readable thresholds)."""
    parts = [d[DEF_NUM].astype(float)]
    for col, ref in DEF_CAT.items():
        dm = pd.get_dummies(d[col].astype(str), prefix=_prefix(col), dtype=int)
        parts.append(dm.drop(columns=[c for c in dm.columns if c.endswith("_" + ref)]))
    X = pd.concat(parts, axis=1).astype(float)
    if columns is not None:                 # align to the training layout
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def validate(d: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems with an input frame (empty = OK)."""
    problems = [f"Missing column: {c}" for c in INPUT_COLUMNS if c not in d.columns]
    if problems:
        return problems
    for c in DEF_NUM + ["Investment_Value_INR", "Portfolio_Return_Percent", "Portfolio_Risk_Score"]:
        bad = pd.to_numeric(d[c], errors="coerce").isna() & d[c].notna()
        if bad.any():
            problems.append(f"{c}: {int(bad.sum())} non-numeric value(s)")
        if d[c].isna().any():
            problems.append(f"{c}: {int(d[c].isna().sum())} blank value(s)")
    return problems


class Scorer:
    """Loads the trained artefacts once and scores new customers."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.rf = joblib.load(model_dir / "default_rf.joblib")
        self.tree = joblib.load(model_dir / "default_tree.joblib")
        self.meta = json.loads((model_dir / "metadata.json").read_text())
        seg = json.loads((model_dir / "segmenter.json").read_text())
        self.fa_mean = np.array(seg["fa_mean"]); self.fa_W = np.array(seg["fa_weights"])
        self.fa_b = np.array(seg["fa_intercept"])
        self.sc_mean = np.array(seg["score_scaler_mean"]); self.sc_scale = np.array(seg["score_scaler_scale"])
        self.centers = np.array(seg["kmeans_centers"]); self.names = seg["cluster_names"]
        self.columns = self.meta["tree_feature_columns"]
        self.threshold = self.meta["rf_threshold"]
        self.rule_feat, self.rule_thr = self.meta["root_rule"]["feature"], self.meta["root_rule"]["threshold"]

    # -- segmentation --------------------------------------------------------
    def factor_scores(self, d: pd.DataFrame) -> np.ndarray:
        return d[FA_VARS].astype(float).values @ self.fa_W + self.fa_b

    def segment(self, d: pd.DataFrame) -> pd.DataFrame:
        f = self.factor_scores(d)
        z = (f - self.sc_mean) / self.sc_scale
        dist = ((z[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        lab = dist.argmin(1)
        return pd.DataFrame({"Credit_Burden_Score": f[:, 0], "Investment_Depth_Score": f[:, 1],
                             "Segment": [self.names[str(i)] for i in lab]}, index=d.index)

    # -- default risk --------------------------------------------------------
    def tree_path(self, x_row: pd.DataFrame) -> list[str]:
        t = self.tree.tree_; node, steps = 0, []
        v = x_row.iloc[0]
        while t.children_left[node] != -1:
            f = self.columns[t.feature[node]]; thr = t.threshold[node]
            if v[f] <= thr:
                steps.append(f"{f} ≤ {thr:,.3f}  (value {v[f]:,.3f})"); node = t.children_left[node]
            else:
                steps.append(f"{f} > {thr:,.3f}  (value {v[f]:,.3f})"); node = t.children_right[node]
        return steps

    def score(self, raw: pd.DataFrame) -> pd.DataFrame:
        d = prepare(raw)
        X = tree_features(d, self.columns)
        out = pd.DataFrame(index=d.index)
        out["Default_Probability_RF"] = self.rf.predict_proba(X)[:, 1]
        out["Default_Flag_RF"] = np.where(out.Default_Probability_RF >= self.threshold, "High risk", "Low risk")
        out["Default_Probability_Tree"] = self.tree.predict_proba(X)[:, 1]
        out[f"Rule_{self.rule_feat}_gt_{self.rule_thr:.3f}"] = np.where(d[self.rule_feat] > self.rule_thr, "Flag", "Pass")
        out = out.join(self.segment(d))
        out["Recommended_Action"] = out.Segment.map(SEGMENT_ACTIONS)
        return out
