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
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🛒 Retail Demand Forecasting Dashboard</h1>', unsafe_allow_html=True)
st.markdown(
    "**Real-World LightGBM Model** · Point forecast + Q5/Q95 prediction interval ribbon · SMAPE: 11.6%"
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

# Add model info to sidebar
st.sidebar.header("🔬 Model Information")
METADATA_PATH = ROOT / "models" / "metadata.json"
METRICS_PATH = ROOT / "models" / "metrics.json"

if METRICS_PATH.exists():
    import json
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    st.sidebar.metric("Model Version", meta.get("version", "1.0.0"))
    st.sidebar.metric("Feature Count", meta.get("feature_count", "N/A"))

if METRICS_PATH.exists():
    import json
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    st.sidebar.metric("CV SMAPE", f"{metrics.get('smape', 0):.2f}%")
    st.sidebar.metric("Interval Coverage", f"{metrics.get('interval_coverage_pct', 0):.1f}%")

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
# Tabs for different views
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Analysis", "🔬 Model Info"])

with tab1:
    st.header(f"Store {selected_store} · Item {selected_item} — Demand Forecast")
    
    # Chart
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

    # Metrics panel
    if forecast is not None and not forecast.empty and "sales" in forecast.columns:
        st.subheader("📊 Forecast Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Forecast", f"{forecast['sales'].mean():.1f}")
        col2.metric("Min Forecast", f"{forecast['sales'].min():.1f}")
        col3.metric("Max Forecast", f"{forecast['sales'].max():.1f}")
        col4.metric("Forecast Range", f"{forecast['sales'].max() - forecast['sales'].min():.1f}")
        
        if "q05" in forecast.columns and "q95" in forecast.columns:
            st.subheader("📉 Prediction Interval Quality")
            col1, col2, col3 = st.columns(3)
            interval_width = (forecast['q95'] - forecast['q05']).mean()
            col1.metric("Avg Interval Width", f"{interval_width:.1f}")
            col2.metric("Mean Q5", f"{forecast['q05'].mean():.1f}")
            col3.metric("Mean Q95", f"{forecast['q95'].mean():.1f}")

        st.subheader("📋 Forecast Data")
        st.dataframe(
            forecast[["date", "store", "item", "sales", "q05", "q95"]].rename(
                columns={"sales": "point_forecast"}
            ),
            use_container_width=True,
        )

with tab2:
    st.header("📊 Analysis")
    
    # Feature Importance
    FI_PATH = ROOT / "outputs" / "feature_importance.csv"
    if FI_PATH.exists():
        st.subheader("🎯 Feature Importance (Top 15)")
        fi_df = pd.read_csv(FI_PATH)
        
        try:
            import plotly.express as px
            fig_fi = px.bar(fi_df.head(15), x='importance', y='feature', 
                           orientation='h', color='importance',
                           color_continuous_scale='Blues',
                           title='Top 15 Most Important Features')
            fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_fi, use_container_width=True)
        except ImportError:
            st.bar_chart(fi_df.head(15).set_index('feature')['importance'])
    
    # Historical Statistics
    st.subheader("📈 Historical Statistics")
    hist_stats = hist['sales'].describe()
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Sales", f"{hist_stats['mean']:.1f}")
    col2.metric("Std Sales", f"{hist_stats['std']:.1f}")
    col3.metric("Total Sales", f"{hist_stats['sum']:.0f}")
    
    # Seasonality Analysis
    st.subheader("📅 Seasonality Patterns")
    hist_copy = hist.copy()
    hist_copy['dayofweek'] = hist_copy['date'].dt.dayofweek
    hist_copy['month'] = hist_copy['date'].dt.month
    
    dow_avg = hist_copy.groupby('dayofweek')['sales'].mean()
    month_avg = hist_copy.groupby('month')['sales'].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Sales by Day of Week**")
        st.bar_chart(dow_avg)
    with col2:
        st.write("**Sales by Month**")
        st.bar_chart(month_avg)
    
    # ML Diagnostics
    st.subheader("🔬 ML Diagnostics")
    
    # Residual Analysis
    if forecast is not None and not forecast.empty and len(hist) > 0:
        try:
            from src.diagnostics import residual_analysis
            
            # Use recent historical data for residual analysis
            recent_hist = hist.tail(90)  # Last 90 days
            if len(forecast) >= 90:
                recent_forecast = forecast.head(90)
                # Align dates for comparison
                recent_hist = recent_hist.tail(90)
                
                if len(recent_hist) == len(recent_forecast):
                    residuals = residual_analysis(
                        recent_hist['sales'].values,
                        recent_forecast['sales'].values
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mean Residual", f"{residuals['mean_residual']:.2f}")
                    col2.metric("Std Residual", f"{residuals['std_residual']:.2f}")
                    col3.metric("Median Residual", f"{residuals['median_residual']:.2f}")
                    
                    # Residual distribution
                    st.write("**Residual Distribution**")
                    residual_values = recent_hist['sales'].values - recent_forecast['sales'].values
                    fig_residuals = px.histogram(x=residual_values, nbins=30, 
                                               title='Residual Distribution')
                    st.plotly_chart(fig_residuals, use_container_width=True)
        except ImportError:
            st.info("Install scipy for advanced diagnostics")
        except Exception as e:
            st.warning(f"Could not compute diagnostics: {str(e)}")

with tab3:
    st.header("🔬 Model Information")
    
    if METADATA_PATH.exists():
        import json
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        st.subheader("📋 Model Metadata")
        st.json(meta)
    
    if METRICS_PATH.exists():
        import json
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        st.subheader("📊 Model Performance Metrics")
        
        # Display metrics as cards
        metrics_display = [
            ('SMAPE', f"{metrics.get('smape', 0):.2f}%"),
            ('MAE', f"{metrics.get('mae', 0):.2f}"),
            ('RMSE', f"{metrics.get('rmse', 0):.2f}"),
            ('WAPE', f"{metrics.get('wape', 0):.2f}%"),
            ('Interval Coverage', f"{metrics.get('interval_coverage_pct', 0):.1f}%"),
            ('Pinball Q05', f"{metrics.get('pinball_q05', 0):.4f}"),
            ('Pinball Q95', f"{metrics.get('pinball_q95', 0):.4f}"),
        ]
        
        cols = st.columns(4)
        for i, (name, value) in enumerate(metrics_display):
            cols[i % 4].metric(name, value)
    
    st.subheader("🔧 Model Architecture")
    st.markdown("""
    **Model Type:** Global LightGBM Regressor with Quantile Loss
    
    **Training Strategy:**
    - 3-fold Walk-Forward Cross-Validation
    - Leakage-aware feature engineering (lags start at 91 days)
    - Separate models for quantile forecasting (Q05, Q95)
    
    **Feature Engineering:**
    - Lag features (91-728 days)
    - Rolling statistics (7-91 day windows)
    - Cyclical date encodings
    - Cross-series aggregates
    - Target encodings
    
    **Performance:** SMAPE 11.6% on 3-fold Walk-Forward CV
    """)

st.markdown("---")
st.caption("Built with LightGBM · SMAPE-optimised · Q5/Q95 quantile regression bounds · Walk-Forward CV")
