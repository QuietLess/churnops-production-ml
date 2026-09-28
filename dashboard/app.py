"""ChurnOps monitoring dashboard (read-only; no training logic lives here).

Run:  streamlit run dashboard/app.py
Env:  DATABASE_URL (prediction logs), MONITORING_DIR (drift artifacts), API_URL (optional)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'churnops.db'}")
MONITORING_DIR = Path(os.getenv("MONITORING_DIR", ROOT / "artifacts" / "monitoring"))
API_URL = os.getenv("API_URL", "").rstrip("/")
RISK_ORDER = ["low", "medium", "high"]
RISK_COLORS = {"low": "#2e7d32", "medium": "#f9a825", "high": "#c62828"}

st.set_page_config(page_title="ChurnOps Monitoring", page_icon="📉", layout="wide")


@st.cache_data(ttl=30)
def load_logs(limit: int) -> pd.DataFrame:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    query = text(
        "SELECT prediction_id, timestamp, customer_id, prediction, probability, risk_level, "
        "threshold, model_name, model_version, source FROM prediction_logs "
        "ORDER BY timestamp DESC LIMIT :limit"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"limit": limit})
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
        df["simulated"] = df["customer_id"].fillna("").str.startswith("SIM-")
    return df


@st.cache_data(ttl=30)
def load_drift_runs() -> list[dict]:
    runs = []
    for path in MONITORING_DIR.glob("*/*/summary.json"):
        try:
            run = json.loads(path.read_text())
            run["_dir"] = str(path.parent)
            runs.append(run)
        except json.JSONDecodeError:
            continue
    return sorted(runs, key=lambda r: r["run_at"], reverse=True)


@st.cache_data(ttl=60)
def load_model_info() -> dict | None:
    if not API_URL:
        return None
    try:
        import httpx

        return httpx.get(f"{API_URL}/model-info", timeout=5).json()
    except Exception:  # noqa: BLE001
        return None


st.title("📉 ChurnOps — Monitoring")
st.caption(
    "Prediction logs and data-drift checks for the churn-risk API. "
    "Traffic with `SIM-` customer IDs and all drift scenarios are **simulated** "
    "(the IBM Telco dataset is static)."
)

with st.sidebar:
    st.header("Filters")
    limit = st.slider("Most recent predictions", 100, 20_000, 5_000, step=100)
    include_sim = st.checkbox("Include simulated traffic", value=True)
    if st.button("Refresh"):
        st.cache_data.clear()

try:
    logs = load_logs(limit)
    db_error = None
except Exception as exc:  # noqa: BLE001
    logs, db_error = pd.DataFrame(), str(exc)

if not logs.empty and not include_sim:
    logs = logs[~logs["simulated"]]

model_info = load_model_info()
current_version = (
    model_info["model_version"] if model_info else (logs["model_version"].iloc[0] if not logs.empty else "—")
)

# --- KPI cards --------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total predictions", f"{len(logs):,}")
k2.metric("Avg churn probability", f"{logs['probability'].mean():.3f}" if not logs.empty else "—")
k3.metric(
    "High-risk customers",
    f"{(logs['risk_level'] == 'high').sum():,}" if not logs.empty else "—",
    help="Risk bands are UI labels: low < 0.30 ≤ medium ≤ 0.60 < high",
)
k4.metric("Serving model version", f"v{current_version}")

if db_error:
    st.warning(f"Prediction log database unavailable: {db_error}")
elif logs.empty:
    st.info("No predictions logged yet. Send traffic with `make simulate-traffic`.")
else:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Churn probability distribution")
        fig = px.histogram(logs, x="probability", nbins=30, color_discrete_sequence=["#1565c0"])
        fig.add_vline(
            x=float(logs["threshold"].iloc[0]), line_dash="dash", annotation_text="decision threshold"
        )
        fig.update_layout(height=320, margin=dict(t=10, b=10), xaxis_title="P(churn)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Risk band distribution")
        counts = (
            logs["risk_level"]
            .value_counts()
            .reindex(RISK_ORDER, fill_value=0)
            .rename_axis("risk_level")
            .reset_index(name="count")
        )
        fig = px.bar(counts, x="risk_level", y="count", color="risk_level", color_discrete_map=RISK_COLORS)
        fig.update_layout(height=320, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Predictions over time")
    span = logs["timestamp"].max() - logs["timestamp"].min()
    freq = "1min" if span < pd.Timedelta(hours=3) else ("1h" if span < pd.Timedelta(days=3) else "1D")
    ts = (
        logs.set_index("timestamp")
        .groupby([pd.Grouper(freq=freq), "risk_level"])
        .size()
        .rename("count")
        .reset_index()
    )
    fig = px.bar(
        ts,
        x="timestamp",
        y="count",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        category_orders={"risk_level": RISK_ORDER},
    )
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

# --- Drift ------------------------------------------------------------------
st.divider()
st.header("Data drift")
runs = load_drift_runs()
if not runs:
    st.info("No drift reports yet. Run `make drift`.")
else:
    labels = [f"{r['run_at'][:19].replace('T', ' ')} · {r['name']}" for r in runs]
    selected = runs[st.selectbox("Monitoring run", range(len(runs)), format_func=lambda i: labels[i])]
    d1, d2, d3, d4 = st.columns(4)
    status = (
        "🔴 DRIFT" if selected["dataset_drift"] else ("🟡 PARTIAL" if selected["n_drifted"] else "🟢 STABLE")
    )
    d1.metric("Status", status)
    d2.metric("Drifted features", f"{selected['n_drifted']} / {selected['n_features']}")
    d3.metric("Current rows", f"{selected['n_current']:,}")
    d4.metric("Last monitoring run", selected["run_at"][:16].replace("T", " ") + " UTC")
    if selected.get("simulated"):
        st.caption("⚠️ This run used simulated production data.")

    feats = pd.DataFrame(selected["features"])
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Feature drift")
        st.dataframe(
            feats[["feature", "drifted", "score", "threshold", "method"]],
            width="stretch",
            hide_index=True,
            height=420,
        )
    with right:
        st.subheader("Top drifted features")
        top = feats[feats["drifted"]].head(8)
        if top.empty:
            st.success("No drifted features in this run.")
        else:
            st.dataframe(top[["feature", "score", "method"]], hide_index=True, width="stretch")
        if selected.get("predictions"):
            p = selected["predictions"]
            st.subheader("Model output on this batch")
            st.write(f"Avg churn probability: **{p['avg_churn_probability']:.3f}**")
            st.write(f"High-risk share: **{p['high_risk_share']:.1%}**")
            st.write(f"Model version: **v{p.get('model_version')}**")

    st.subheader("Drift over monitoring runs")
    history = pd.DataFrame(
        [{"run_at": r["run_at"], "name": r["name"], "share_drifted": r["share_drifted"]} for r in runs]
    )
    fig = px.bar(history.sort_values("run_at"), x="run_at", y="share_drifted", color="name")
    fig.update_layout(height=280, margin=dict(t=10, b=10), yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    html = Path(selected["_dir"]) / selected.get("report_html", "report.html")
    if html.exists():
        with st.expander("Full Evidently report"):
            st.iframe(html, height=900)  # our own generated report, not user input

if model_info:
    with st.expander("Model metadata"):
        st.json(model_info)
