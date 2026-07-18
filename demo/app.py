"""
Retail Demand Forecasting — Streamlit Demo App
Visualises point forecasts + Q5/Q95 uncertainty ribbon per Store/Item.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Allow importing from project src
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Retail Demand Forecasting Dashboard")
st.markdown(
    "**Real-World LightGBM Model** · Point forecast + Q5/Q95 prediction interval ribbon"
)

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
DATA_DIR = ROOT / "data" / "raw"
PRED_PATH = ROOT / "outputs" / "predictions.parquet"


@st.cache_data
def load_train():
    path = DATA_DIR / "train.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df


@st.cache_data
def load_predictions():
    if not PRED_PATH.exists():
        return None
    return pd.read_parquet(PRED_PATH)


@st.cache_data
def load_test():
    path = DATA_DIR / "test.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


train_df = load_train()
pred_df = load_predictions()
test_df = load_test()

if train_df is None:
    st.error("⚠️ Training data not found at `data/raw/train.csv`. Please add the raw Kaggle data.")
    st.stop()

# ──────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────
st.sidebar.header("📌 Filters")

stores = sorted(train_df["store"].unique())
items = sorted(train_df["item"].unique())

selected_store = st.sidebar.selectbox("Store", stores, index=0)
selected_item = st.sidebar.selectbox("Item", items, index=0)

show_ribbon = st.sidebar.checkbox("Show Q5/Q95 Uncertainty Ribbon", value=True)
show_actuals = st.sidebar.checkbox("Show Historical Actuals", value=True)

# ──────────────────────────────────────────────
# Filter data for selected store/item
# ──────────────────────────────────────────────
hist = train_df[(train_df["store"] == selected_store) & (train_df["item"] == selected_item)].copy()

forecast = None
if pred_df is not None and test_df is not None:
    test_filtered = test_df[(test_df["store"] == selected_store) & (test_df["item"] == selected_item)].copy()
    if "id" in test_filtered.columns and "id" in pred_df.columns:
        forecast = test_filtered.merge(pred_df[["id", "sales", "q05", "q95"]], on="id", how="left")

# ──────────────────────────────────────────────
# Chart
# ──────────────────────────────────────────────
try:
    import plotly.graph_objects as go

    fig = go.Figure()

    if show_actuals:
        fig.add_trace(
            go.Scatter(
                x=hist["date"],
                y=hist["sales"],
                mode="lines",
                name="Historical Sales",
                line=dict(color="#6c8ebf", width=1.5),
            )
        )

    if forecast is not None and not forecast.empty:
        if show_ribbon and "q05" in forecast.columns and "q95" in forecast.columns:
            # Confidence ribbon
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([forecast["date"], forecast["date"].iloc[::-1]]),
                    y=pd.concat([forecast["q95"], forecast["q05"].iloc[::-1]]),
                    fill="toself",
                    fillcolor="rgba(255, 140, 0, 0.20)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Q5–Q95 Interval",
                )
            )

        # Point forecast
        fig.add_trace(
            go.Scatter(
                x=forecast["date"],
                y=forecast["sales"],
                mode="lines",
                name="Point Forecast",
                line=dict(color="#e67e22", width=2.5, dash="solid"),
            )
        )

    fig.update_layout(
        title=f"Store {selected_store} · Item {selected_item} — Demand Forecast",
        xaxis_title="Date",
        yaxis_title="Units Sold",
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

except ImportError:
    st.line_chart(hist.set_index("date")["sales"])
    st.warning("Install plotly for the full interactive chart: `pip install plotly`")

# ──────────────────────────────────────────────
# Metrics panel
# ──────────────────────────────────────────────
if forecast is not None and not forecast.empty and "sales" in forecast.columns:
    st.subheader("📊 Forecast Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Point Forecast", f"{forecast['sales'].mean():.1f}")
    if "q05" in forecast.columns:
        col2.metric("Mean Q5 (Lower)", f"{forecast['q05'].mean():.1f}")
    if "q95" in forecast.columns:
        col3.metric("Mean Q95 (Upper)", f"{forecast['q95'].mean():.1f}")

    st.dataframe(
        forecast[["date", "store", "item", "sales", "q05", "q95"]].rename(
            columns={"sales": "point_forecast"}
        ),
        use_container_width=True,
    )

# ──────────────────────────────────────────────
# Model Info
# ──────────────────────────────────────────────
METADATA_PATH = ROOT / "models" / "metadata.json"
METRICS_PATH = ROOT / "models" / "metrics.json"

if METADATA_PATH.exists() or METRICS_PATH.exists():
    st.subheader("🔬 Model Info")
    info_col, metric_col = st.columns(2)

    if METADATA_PATH.exists():
        import json
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        info_col.json(meta)

    if METRICS_PATH.exists():
        import json
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        metric_col.json(metrics)

st.markdown("---")
st.caption("Built with LightGBM · SMAPE-optimised · Q5/Q95 quantile regression bounds")
