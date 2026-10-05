"""Credit Card Fraud Detection System - Streamlit UI.
Run:  streamlit run app.py
Maps to PRD requirements: UI-1, UI-2, RES-1, RES-2, ML-3, ERR-1, VIS-1.
"""
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.predict import FraudPredictor
from src.validation import validate_dataframe

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")


@st.cache_resource
def get_predictor():
    return FraudPredictor()


@st.cache_data
def load_samples():
    p = C.SAMPLE_DIR / "sample_transactions.csv"
    return pd.read_csv(p) if p.exists() else None


def gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%"},
        title={"text": "Fraud Probability"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
               "steps": [{"range": [0, C.RISK_LOW_MAX * 100], "color": "#8fd19e"},
                         {"range": [C.RISK_LOW_MAX * 100, C.RISK_HIGH_MIN * 100], "color": "#ffd966"},
                         {"range": [C.RISK_HIGH_MIN * 100, 100], "color": "#f4a09a"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# ------------------------------------------------------------------ setup
try:
    predictor = get_predictor()
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

samples = load_samples()
defaults = {f: 0.0 for f in C.FEATURES}
for k, v in defaults.items():
    st.session_state.setdefault(f"in_{k}", v)

st.title("💳 Credit Card Fraud Detection")
st.caption("Machine-learning prototype · classifies a transaction as Fraudulent or Legitimate. "
           "Academic prototype - it does not authorise or block real transactions.")

tab_pred, tab_batch, tab_perf, tab_about = st.tabs(
    ["🔍 Single Transaction", "📂 Batch (CSV)", "📊 Model Performance", "ℹ️ About"])

# ------------------------------------------------------------ single prediction
with tab_pred:
    st.subheader("Enter transaction details")
    c1, c2, c3 = st.columns([1, 1, 2])
    if samples is not None:
        if c1.button("Load sample: legitimate"):
            row = samples[samples[C.TARGET] == 0].sample(1).iloc[0]
            for f in C.FEATURES: st.session_state[f"in_{f}"] = float(row[f])
        if c2.button("Load sample: fraud"):
            row = samples[samples[C.TARGET] == 1].sample(1).iloc[0]
            for f in C.FEATURES: st.session_state[f"in_{f}"] = float(row[f])
    c3.info("V1-V28 are anonymised PCA components from the dataset, so sample buttons "
            "are the easiest way to try the model.")

    st.number_input("Transaction Amount", min_value=None, step=1.0, format="%.2f",
                    key="in_Amount", help="Must be between 0 and 1,000,000")
    with st.expander("PCA features V1 - V28", expanded=False):
        cols = st.columns(4)
        for i, f in enumerate(C.PCA_FEATURES):
            cols[i % 4].number_input(f, step=0.1, format="%.4f", key=f"in_{f}")

    if st.button("Analyse Transaction", type="primary"):
        values = {f: st.session_state[f"in_{f}"] for f in C.FEATURES}
        res = predictor.predict_one(values)
        if not res["ok"]:
            st.error("Please correct the following before prediction:")
            for e in res["errors"]: st.warning(e)
        else:
            left, right = st.columns([1, 1])
            with left:
                if res["prediction"] == "FRAUDULENT":
                    st.error(f"## Prediction: {res['prediction']}")
                else:
                    st.success(f"## Prediction: {res['prediction']}")
                risk_icon = {"Low Risk": "🟢", "Medium Risk": "🟠", "High Risk": "🔴"}[res["risk"]]
                st.metric("Risk Level", f"{risk_icon} {res['risk']}")
                st.write(f"Fraud probability: **{res['fraud_probability']:.2%}**  \n"
                         f"Legitimate probability: **{res['legit_probability']:.2%}**")
            with right:
                st.plotly_chart(gauge(res["fraud_probability"]), width="stretch")

# ------------------------------------------------------------ batch prediction
with tab_batch:
    st.subheader("Score many transactions")
    st.write("Upload a CSV containing columns `V1`...`V28` and `Amount` "
             "(an optional `Class`/`Time` column is ignored).")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as ex:
            st.error(f"Could not read the file: {ex}"); df = None
        if df is not None:
            errs = validate_dataframe(df)
            if errs:
                for e in errs: st.error(e)
            else:
                out = predictor.predict_batch(df)
                n_f = int((out["prediction"] == "FRAUDULENT").sum())
                m1, m2, m3 = st.columns(3)
                m1.metric("Transactions", f"{len(out):,}")
                m2.metric("Flagged fraudulent", f"{n_f:,}")
                m3.metric("Flag rate", f"{n_f / len(out):.2%}")
                st.dataframe(out.sort_values("fraud_probability", ascending=False)
                             .head(500), width="stretch")
                st.download_button("Download results (CSV)", out.to_csv(index=False),
                                   "fraud_predictions.csv", "text/csv")
    elif samples is not None:
        st.download_button("Download sample file", samples.to_csv(index=False),
                           "sample_transactions.csv", "text/csv")

# ------------------------------------------------------------ performance
with tab_perf:
    st.subheader("Model evaluation (held-out test set)")
    if not C.METRICS_PATH.exists():
        st.warning("Run `python -m src.train` to generate evaluation results.")
    else:
        m = json.loads(C.METRICS_PATH.read_text())
        sel = m["selected_model"]; res = m["test_results"]
        st.write(f"**Selected model:** {sel}  ·  "
                 f"Dataset: {m['dataset']['rows']:,} transactions, "
                 f"{m['dataset']['fraud']:,} fraud ({m['dataset']['fraud_percent']}%)")
        r = res[sel]
        cols = st.columns(6)
        for col, k in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]):
            col.metric(k.upper().replace("_", "-"), f"{r[k]:.4f}")
        st.dataframe(pd.DataFrame({n: {k: v for k, v in d.items() if k != "confusion_matrix"}
                                   for n, d in res.items()}).T.round(4),
                     width="stretch")
        g = st.columns(2)
        for i, (fn, cap) in enumerate([("confusion_matrix.png", "Confusion matrix"),
                                       ("roc_curves.png", "ROC curves"),
                                       ("pr_curves.png", "Precision-Recall curves"),
                                       ("model_comparison.png", "Model comparison"),
                                       ("feature_importance.png", "Feature importance"),
                                       ("class_distribution.png", "Class distribution")]):
            p = C.FIG_DIR / fn
            if p.exists(): g[i % 2].image(str(p), caption=cap)

# ------------------------------------------------------------ about
with tab_about:
    st.markdown("""
**Workflow:** Input → Validation → Preprocessing (Amount scaling) → Trained model →
Probability → Fraud / Legitimate + Risk level.

**Risk bands:** Low < 30% · Medium 30-70% · High ≥ 70% fraud probability.
**Decision threshold:** 50%.

**Why not just accuracy?** Only ~0.17% of transactions are fraud, so a model that always says
"legitimate" scores ~99.8% accuracy. Precision, recall, F1, ROC-AUC and PR-AUC are reported instead.

Group 08 · Techno Main Salt Lake · B.Tech CSE (Final Year, 7th Sem)
    """)
