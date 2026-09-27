"""Banking Customer Analytics — Streamlit app.

Segmentation, profitability and default-risk results from the AADM End-Term Project
(Ojas Paradkar, WeSchool PGDM E-Business 2024-26), plus live scoring of new customers.

Run locally:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from core import DATA_PATH, DEF_CAT, INPUT_COLUMNS, ROOT, SEGMENT_ACTIONS, Scorer, prepare, tree_features, validate

st.set_page_config(page_title="Banking Customer Analytics", page_icon="🏦", layout="wide")

FIGS = ROOT / "assets" / "figs"
TABLES = ROOT / "assets" / "tables"
SEG_ORDER = ["Transactional-only", "Debt-free Investors", "Pure Borrowers", "Leveraged Investors"]
SEG_COLORS = dict(zip(SEG_ORDER, ["#66c2a5", "#8da0cb", "#fc8d62", "#e78ac3"]))


@st.cache_resource
def load_scorer() -> Scorer:
    return Scorer()


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


scorer = load_scorer()
df = load_data()
meta = scorer.meta

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🏦 Banking Analytics")
page = st.sidebar.radio("Go to", ["Overview", "Score a customer", "Batch scoring", "Segment explorer",
                                  "Model performance", "Report figures"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Data: {meta['n_records']:,} records · 62 variables\n\n"
                   f"Models trained on {meta['n_train']:,} records, tested on {meta['n_test']:,} "
                   f"(split grouped by Customer_ID).\n\nAADM End-Term Project · Ojas Paradkar")


# ================================================================= OVERVIEW
if page == "Overview":
    st.title("Customer Segmentation, Profitability & Default Risk")
    st.caption("Indian banking dataset · 3,000 records · what the data can and cannot tell a bank")

    rf = meta["rf_test_metrics"]
    c = st.columns(5)
    c[0].metric("Records", f"{len(df):,}")
    c[1].metric("Default rate", f"{df.Default.mean()*100:.1f}%")
    c[2].metric("Loss-making records", f"{(df.Net_Profit_INR < 0).mean()*100:.1f}%")
    c[3].metric("Default model ROC-AUC", f"{rf['ROC_AUC']:.3f}")
    c[4].metric("Segments (silhouette)", f"{meta['segmentation']['k']} ({meta['segmentation']['silhouette']:.2f})")

    st.subheader("Key findings")
    st.markdown(f"""
- **Four behavioural segments** from factor analysis (Credit Burden, Investment Depth) + K-Means.
  Default ranges from **1.1% to 40.7%** across segments; net profit does **not** differ significantly (p = 0.12).
- **The two borrower segments hold 48% of records and 97% of defaults.**
- **Profitability is not predictable** from customer attributes — OLS / Ridge / Lasso test R² ≈ 0.
  Only consistent driver: complaints (≈ ₹443 lower net profit per complaint).
- **Default is highly predictable** from debt-to-income, credit utilisation and credit score.
  Random forest ROC-AUC **{rf['ROC_AUC']:.3f}**, recall **{rf['Recall']:.3f}**.
- Root rule **{meta['root_rule']['feature']} > {meta['root_rule']['threshold']:.3f}** flagged 93.6% of test-set defaulters
  (97.8% of their outstanding), at the cost of flagging 155 of 611 non-defaulters.
""")

    seg = load_table("T14_integrated_segment_matrix")
    st.subheader("Strategic customer matrix")
    fig = px.scatter(seg, x="Default_Rate_pct", y="Mean_Net_Profit_INR", size="Records", color="Segment_Name",
                     color_discrete_map=SEG_COLORS, size_max=60, text="Segment_Name",
                     labels={"Default_Rate_pct": "Default rate (%)", "Mean_Net_Profit_INR": "Mean net profit (₹)"})
    fig.update_traces(textposition="top center", showlegend=False)
    fig.add_hline(y=df.Net_Profit_INR.mean(), line_dash="dot", annotation_text="Portfolio mean profit")
    fig.update_layout(height=480, xaxis_range=[-5, 50], yaxis_range=[2500, 3700])
    st.plotly_chart(fig, width="stretch")

    show = seg[["Segment_Name", "Records", "Share_of_Records_pct", "Default_Rate_pct", "Share_of_Defaults_pct",
                "Mean_Net_Profit_INR", "Share_of_Profit_pct", "Total_Outstanding_Loan_Cr"]].rename(columns={
        "Segment_Name": "Segment", "Share_of_Records_pct": "% of records", "Default_Rate_pct": "Default rate %",
        "Share_of_Defaults_pct": "% of defaults", "Mean_Net_Profit_INR": "Mean profit ₹",
        "Share_of_Profit_pct": "% of profit", "Total_Outstanding_Loan_Cr": "Outstanding ₹ Cr"})
    st.dataframe(show, hide_index=True, width="stretch")
    st.subheader("Recommended action by segment")
    st.table(pd.DataFrame({"Segment": SEGMENT_ACTIONS.keys(), "Action": SEGMENT_ACTIONS.values()}).set_index("Segment"))


# ========================================================= SCORE A CUSTOMER
elif page == "Score a customer":
    st.title("Score a customer")
    st.caption("Enter one customer's details. Gender and leakage variables (Default_Probability, "
               "Credit_Risk_Category) are deliberately not used.")

    med = df[[c for c in INPUT_COLUMNS if c not in DEF_CAT]].median()
    opts = {c: sorted(df[c].fillna("No Loan").astype(str).unique()) for c in DEF_CAT}

    with st.form("score"):
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Profile**")
            age = st.number_input("Age", 18, 100, int(med.Age))
            occ = st.selectbox("Occupation", opts["Occupation"], index=opts["Occupation"].index("Salaried"))
            segl = st.selectbox("Bank's customer segment", opts["Customer_Segment"], index=opts["Customer_Segment"].index("Retail"))
            acct = st.selectbox("Primary account type", opts["Primary_Account_Type"], index=opts["Primary_Account_Type"].index("Savings"))
            income = st.number_input("Annual income (₹)", 0.0, 1e9, float(med.Annual_Income_INR), step=50000.0)
            tenure = st.number_input("Tenure (months)", 0, 600, int(med.Customer_Tenure_Months))
        with b:
            st.markdown("**Credit**")
            score = st.slider("Credit score", 300, 900, int(med.Credit_Score))
            util = st.slider("Credit utilisation ratio", 0.0, 1.0, float(round(med.Credit_Utilization_Ratio, 3)), 0.005)
            dti = st.slider("Debt-to-income ratio", 0.0, 0.95, 0.42, 0.01)
            limit = st.number_input("Credit limit (₹)", 0.0, 1e9, float(med.Credit_Limit_INR), step=50000.0)
            loan = st.selectbox("Loan product", opts["Loan_Product"], index=opts["Loan_Product"].index("No Loan"))
            has_loan = loan != "No Loan"
            outstanding = st.number_input("Outstanding loan (₹)", 0.0, 1e10, 4_800_000.0 if has_loan else 50_000.0, step=100000.0)
            rate = st.number_input("Interest rate (%)", 0.0, 40.0, 12.9 if has_loan else 0.0, step=0.1)
            emi = st.number_input("EMI (₹ / month)", 0.0, 1e8, 74_000.0 if has_loan else 0.0, step=1000.0)
        with c:
            st.markdown("**Behaviour & investments**")
            bal = st.number_input("Account balance (₹)", 0.0, 1e10, float(med.Account_Balance_Before_INR), step=50000.0)
            tx30 = st.number_input("Transactions, last 30 days", 0, 500, int(med.Transactions_Last_30_Days))
            spend = st.number_input("Average monthly spend (₹)", 0.0, 1e9, float(med.Average_Monthly_Spend_INR), step=5000.0)
            recency = st.number_input("Days since last transaction", 0, 3650, int(med.Days_Since_Last_Transaction))
            inv_val = st.number_input("Investment value (₹, 0 if none)", 0.0, 1e10, 0.0, step=100000.0)
            inv_ret = st.number_input("Portfolio return (%)", -100.0, 200.0, 0.0, step=0.5)
            inv_risk = st.number_input("Portfolio risk score (0-100)", 0.0, 100.0, 0.0, step=1.0)
        go = st.form_submit_button("Score customer", type="primary")

    if go:
        row = pd.DataFrame([{
            "Credit_Score": score, "Credit_Utilization_Ratio": util, "Credit_Limit_INR": limit,
            "Outstanding_Loan_INR": outstanding, "Interest_Rate_Percent": rate, "EMI_Amount_INR": emi,
            "Debt_to_Income_Ratio": dti, "Annual_Income_INR": income, "Account_Balance_Before_INR": bal,
            "Customer_Tenure_Months": tenure, "Transactions_Last_30_Days": tx30, "Average_Monthly_Spend_INR": spend,
            "Days_Since_Last_Transaction": recency, "Age": age, "Loan_Product": loan, "Primary_Account_Type": acct,
            "Occupation": occ, "Customer_Segment": segl, "Investment_Value_INR": inv_val,
            "Portfolio_Return_Percent": inv_ret, "Portfolio_Risk_Score": inv_risk}])
        r = scorer.score(row).iloc[0]
        rule_col = [k for k in r.index if k.startswith("Rule_")][0]

        k = st.columns(4)
        k[0].metric("Default probability (RF)", f"{r.Default_Probability_RF*100:.1f}%")
        k[1].metric("Risk flag (threshold 0.5)", r.Default_Flag_RF)
        k[2].metric("Policy rule", f"{r[rule_col]}", help=f"{meta['root_rule']['feature']} > {meta['root_rule']['threshold']:.3f}")
        k[3].metric("Behavioural segment", r.Segment)
        (st.error if r.Default_Flag_RF == "High risk" else st.success)(f"**Action:** {r.Recommended_Action}")

        with st.expander("Why? Decision-tree path for this customer"):
            for s in scorer.tree_path(tree_features(prepare(row), scorer.columns)):
                st.write("→ " + s)
            st.caption(f"Tree leaf default probability (class-weighted): {r.Default_Probability_Tree*100:.1f}%")
        st.caption(f"Factor scores — Credit Burden {r.Credit_Burden_Score:.2f}, Investment Depth {r.Investment_Depth_Score:.2f}. "
                   "Probabilities come from class-weighted models, so they rank risk well but overstate the raw default rate.")


# ============================================================ BATCH SCORING
elif page == "Batch scoring":
    st.title("Batch scoring")
    st.write("Upload a CSV with the columns below (one row per customer). Extra columns are kept and returned.")
    st.code(", ".join(INPUT_COLUMNS), language=None, wrap_lines=True)
    sample = pd.read_csv(ROOT / "sample" / "sample_customers.csv")
    st.download_button("Download template / sample CSV", sample.to_csv(index=False), "sample_customers.csv", "text/csv")

    up = st.file_uploader("Upload customers CSV", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        problems = validate(raw)
        if problems:
            st.error("Fix these problems and upload again:\n\n- " + "\n- ".join(problems))
        else:
            for col in [c for c in INPUT_COLUMNS if c not in DEF_CAT]:
                raw[col] = pd.to_numeric(raw[col])
            res = pd.concat([raw.reset_index(drop=True), scorer.score(raw).reset_index(drop=True)], axis=1)
            c = st.columns(3)
            c[0].metric("Customers scored", f"{len(res):,}")
            c[1].metric("Flagged high risk (RF)", f"{(res.Default_Flag_RF == 'High risk').sum():,}")
            c[2].metric("Mean default probability", f"{res.Default_Probability_RF.mean()*100:.1f}%")
            st.plotly_chart(px.histogram(res, x="Segment", color="Default_Flag_RF", barmode="group",
                                         category_orders={"Segment": SEG_ORDER}), width="stretch")
            st.dataframe(res, width="stretch")
            st.download_button("Download scored CSV", res.to_csv(index=False), "scored_customers.csv", "text/csv",
                               type="primary")


# ========================================================= SEGMENT EXPLORER
elif page == "Segment explorer":
    st.title("Segment explorer")
    segs = st.multiselect("Segments", SEG_ORDER, SEG_ORDER)
    d = df[df.Segment_Name.isin(segs)]
    if d.empty:
        st.warning("Select at least one segment."); st.stop()
    st.plotly_chart(px.scatter(d, x="F_Credit_Burden", y="F_Investment_Depth", color="Segment_Name", opacity=.6,
                               color_discrete_map=SEG_COLORS, height=450,
                               labels={"F_Credit_Burden": "Credit Burden factor", "F_Investment_Depth": "Investment Depth factor"}),
                    width="stretch")
    metric = st.selectbox("Compare segments on", ["Default", "Net_Profit_INR", "Debt_to_Income_Ratio", "Credit_Score",
                                                  "Credit_Utilization_Ratio", "Annual_Income_INR", "Digital_Engagement_Score",
                                                  "Customer_Satisfaction_Score", "Complaint_Count", "Customer_Lifetime_Value_INR"])
    agg = d.groupby("Segment_Name")[metric].mean().reindex([s for s in SEG_ORDER if s in segs]).reset_index()
    if metric == "Default":
        agg[metric] *= 100
    st.plotly_chart(px.bar(agg, x="Segment_Name", y=metric, color="Segment_Name", color_discrete_map=SEG_COLORS,
                           text_auto=".1f", labels={metric: ("Default rate (%)" if metric == "Default" else f"Mean {metric}")}),
                    width="stretch")
    st.caption("Profitability differences across segments are not statistically significant (Kruskal-Wallis p = 0.12).")
    with st.expander("Full segment profiles"):
        st.dataframe(load_table("T08b_cluster_profiles"), hide_index=True, width="stretch")


# ======================================================== MODEL PERFORMANCE
elif page == "Model performance":
    st.title("Model performance (held-out test set)")
    st.subheader("Default classification")
    st.dataframe(load_table("T13_classification_comparison"), hide_index=True, width="stretch")
    st.image(str(FIGS / "F19_classification_eval.png"), width="stretch")
    a, b = st.columns(2)
    a.image(str(FIGS / "F18_rf_importance.png"), width="stretch")
    b.subheader("Credit-policy impact (test set)")
    b.dataframe(load_table("T15_policy_impact"), hide_index=True, width="stretch")
    st.subheader("Decision-tree rules")
    st.dataframe(load_table("T12a_tree_rules"), hide_index=True, width="stretch")
    st.subheader("Profitability regression — no predictive power")
    st.dataframe(load_table("T10a_regression_comparison"), hide_index=True, width="stretch")
    st.caption("All models perform at or near the naive-mean baseline, so profitability is not scored in this app.")
    with st.expander("Deployed model details"):
        st.json(meta)


# ============================================================ REPORT FIGURES
elif page == "Report figures":
    st.title("Report figures")
    figs = sorted(FIGS.glob("*.png"))
    choice = st.selectbox("Figure", figs, format_func=lambda p: p.stem.replace("_", " "))
    st.image(str(choice), width="stretch")
    with st.expander("All result tables"):
        t = st.selectbox("Table", sorted(p.stem for p in TABLES.glob("*.csv")))
        tbl = load_table(t)
        st.dataframe(tbl, width="stretch")
        st.download_button("Download table", tbl.to_csv(index=False), f"{t}.csv", "text/csv")
