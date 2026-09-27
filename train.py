"""Re-trains every deployed artefact from data/bank_data_clean.csv.

Run:  python train.py
Reproduces the notebook's pipeline (same seed, same customer-grouped 75/25 split,
same grids) and writes models/ + models/metadata.json. Takes ~1-3 minutes.
"""
import inspect
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_array as _check_array

import factor_analyzer.factor_analyzer as _fa_mod
from factor_analyzer import FactorAnalyzer

from core import DATA_PATH, FA_VARS, MODEL_DIR, prepare, tree_features

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# factor_analyzer 0.5.x passes `force_all_finite`, renamed in newer scikit-learn
_NEW_SK = "ensure_all_finite" in inspect.signature(_check_array).parameters
def _check_array_compat(*a, **k):
    if _NEW_SK and "force_all_finite" in k:
        k["ensure_all_finite"] = k.pop("force_all_finite")
    return _check_array(*a, **k)
_fa_mod.check_array = _check_array_compat


def clf_metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred),
            "Recall": recall_score(y, pred), "F1": f1_score(y, pred), "ROC_AUC": roc_auc_score(y, prob)}


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    d = prepare(pd.read_csv(DATA_PATH))
    print("Records:", len(d))

    # ---------------- Segmentation: FA (2 factors, varimax) -> K-Means -------
    Xfa_raw = d[FA_VARS].astype(float).values
    Xfa = StandardScaler().fit_transform(Xfa_raw)
    fa = FactorAnalyzer(n_factors=2, rotation="varimax", method="principal").fit(Xfa)
    scores = fa.transform(Xfa)
    emi_col = int(np.argmax(np.abs(fa.loadings_[FA_VARS.index("EMI_Amount_INR")])))
    scores = scores[:, [emi_col, 1 - emi_col]]              # [Credit_Burden, Investment_Depth]

    # FA scoring is linear in the raw inputs -> store it as W, b (no factor_analyzer needed at runtime)
    A = np.column_stack([Xfa_raw, np.ones(len(Xfa_raw))])
    coef, *_ = np.linalg.lstsq(A, scores, rcond=None)
    W, b = coef[:-1], coef[-1]
    max_err = np.abs(Xfa_raw @ W + b - scores).max()
    assert max_err < 1e-6, f"Linear FA reconstruction error too large: {max_err}"

    sc = StandardScaler().fit(scores)
    Zc = sc.transform(scores)
    sil = {k: silhouette_score(Zc, KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(Zc).labels_)
           for k in range(2, 7)}
    K = max(sil, key=sil.get)
    km = KMeans(n_clusters=K, n_init=20, random_state=RANDOM_STATE).fit(Zc)
    d["Cluster"] = km.labels_
    names = {}
    for c, g in d.groupby("Cluster"):
        loan = g.Has_Loan.mean() > 0.5 if "Has_Loan" in g else (g.Loan_Product != "No Loan").mean() > 0.5
        inv = (g.Investment_Value_INR > 0).mean() > 0.5
        names[str(c)] = {(True, True): "Leveraged Investors", (True, False): "Pure Borrowers",
                         (False, True): "Debt-free Investors", (False, False): "Transactional-only"}[(loan, inv)]
    d["Segment_New"] = d.Cluster.astype(str).map(names)
    agree = (d.Segment_New == d.Segment_Name).mean() if "Segment_Name" in d else None
    print(f"k = {K}, silhouette = {sil[K]:.3f}; agreement with notebook segments = {agree}")

    (MODEL_DIR / "segmenter.json").write_text(json.dumps({
        "fa_vars": FA_VARS, "fa_mean": Xfa_raw.mean(0).tolist(), "fa_weights": W.tolist(), "fa_intercept": b.tolist(),
        "factor_names": ["Credit_Burden", "Investment_Depth"],
        "score_scaler_mean": sc.mean_.tolist(), "score_scaler_scale": sc.scale_.tolist(),
        "kmeans_centers": km.cluster_centers_.tolist(), "cluster_names": names}, indent=2))

    # ---------------- Default risk: Decision Tree + Random Forest ------------
    X = tree_features(d)
    y = d.Default
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    tr, te = next(gss.split(d, groups=d.Customer_ID))
    gkf = GroupKFold(n_splits=5); groups = d.Customer_ID.iloc[tr].values
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

    dt_gs = GridSearchCV(DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
                         {"max_depth": [2, 3, 4, 5, 6, 8], "min_samples_leaf": [10, 25, 50, 100]},
                         cv=gkf, scoring="roc_auc").fit(Xtr, ytr, groups=groups)
    dt = dt_gs.best_estimator_
    rf_gs = GridSearchCV(RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=RANDOM_STATE, n_jobs=-1),
                         {"max_depth": [4, 6, 8, 12], "min_samples_leaf": [5, 10, 20], "max_features": ["sqrt", 0.5]},
                         cv=gkf, scoring="roc_auc", n_jobs=1).fit(Xtr, ytr, groups=groups)
    rf = rf_gs.best_estimator_

    m_rf, m_dt = clf_metrics(yte, rf.predict_proba(Xte)[:, 1]), clf_metrics(yte, dt.predict_proba(Xte)[:, 1])
    print("RF  test:", {k: round(v, 4) for k, v in m_rf.items()}, rf_gs.best_params_)
    print("DT  test:", {k: round(v, 4) for k, v in m_dt.items()}, dt_gs.best_params_)

    joblib.dump(rf, MODEL_DIR / "default_rf.joblib", compress=3)
    joblib.dump(dt, MODEL_DIR / "default_tree.joblib", compress=3)

    import sklearn
    meta = {
        "trained_on": str(DATA_PATH.name), "n_records": int(len(d)), "n_train": int(len(tr)), "n_test": int(len(te)),
        "split": "GroupShuffleSplit 75/25 grouped by Customer_ID, random_state=42",
        "tree_feature_columns": list(X.columns), "rf_threshold": 0.5,
        "rf_best_params": rf_gs.best_params_, "dt_best_params": dt_gs.best_params_,
        "rf_test_metrics": {k: round(v, 4) for k, v in m_rf.items()},
        "dt_test_metrics": {k: round(v, 4) for k, v in m_dt.items()},
        "rf_cv_auc": round(rf_gs.best_score_, 4), "dt_cv_auc": round(dt_gs.best_score_, 4),
        "root_rule": {"feature": X.columns[dt.tree_.feature[0]], "threshold": float(dt.tree_.threshold[0])},
        "segmentation": {"k": int(K), "silhouette": round(float(sil[K]), 4),
                         "agreement_with_notebook": None if agree is None else round(float(agree), 4)},
        "sklearn_version": sklearn.__version__,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("Saved artefacts to", MODEL_DIR)


if __name__ == "__main__":
    main()
