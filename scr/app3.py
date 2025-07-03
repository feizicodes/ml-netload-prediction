import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from dotenv import load_dotenv

# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(layout="wide", page_title="Net Load Forecast")

# Custom CSS for the dark theme, centered title, and larger widgets
st.markdown("""
<style>
    /* Main background and text color */
    .stApp {
        background-color: #0f1116;
        color: #fafafa;
    }
    
    /* --- Zentrierter Titel --- */
    .main-title {
        text-align: center;
        font-size: 3.5em;
        font-weight: bold;
        padding-top: 20px;
    }
    .sub-title {
        text-align: center;
        font-size: 1.5em;
        color: #fafafa;
        padding-bottom: 30px;
    }

    /* --- Alle Überschriften in Weiß --- */
    h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
    }

    /* --- Größere Widgets für die Präsentation --- */
    .stDateInput > div > div, .stSelectbox > div > div {
        font-size: 1.5em;
    }
    .stDateInput label, .stSelectbox label {
        font-size: 1.5em !important;
        font-weight: bold !important;
        color: #fafafa !important;
    }
    
    /* --- GEÄNDERT: Styling für die Metrik-Karten mit größerer Schrift --- */
    .metric-card {
        background-color: #1e2029;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .metric-label {
        font-size: 1.6em !important; /* Deutlich größer */
        font-weight: bold;
        color: #fafafa;
    }
    .metric-value {
        font-size: 2.8em !important; /* Deutlich größer */
        font-weight: bold;
        color: #00aaff;
    }
</style>
""", unsafe_allow_html=True)

plt.style.use('dark_background')


# --- DATA LOADING & MODEL TRAINING ---
@st.cache_data
def load_and_train_model():
    """
    Loads data, trains the model, and returns predictions. Cached for performance.
    """
    try:
        df = pd.read_csv("../data/eda_b4s_clean_d-2.csv")
    except FileNotFoundError:
        st.error("Error: Data file `../data/eda_b4s_clean_d-2.csv` not found. Please check the path.")
        return None

    df["time"] = pd.to_datetime(df["time"])
    features = ['feature_11', 'hour', 'weekday', 'is_weekend', 'Holiday', 'Net Load (d-1)']
    target = 'Nettolast_P_kW'

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        learning_rate=0.1, max_depth=5, n_estimators=200,
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    df_plot = pd.DataFrame({
        "time": df.loc[y_test.index, "time"].values,
        "actual": y_test.values,
        "predicted": y_pred
    }).sort_values("time").reset_index(drop=True)
    
    return df_plot

df_plot = load_and_train_model()
if df_plot is None:
    st.stop()


# --- APP LAYOUT ---

# Titel wird mit Markdown zentriert
st.markdown('<h1 class="main-title">⚡️ Net Load Forecast Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Analyze and forecast net load on a daily and monthly basis.</p>', unsafe_allow_html=True)


# --- DAILY FORECAST ---
st.header("📅 Daily Net Load Forecast")
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Settings")
    available_dates = sorted(df_plot["time"].dt.date.unique())
    selected_day = st.date_input("Select a date:", value=available_dates[0],
                                 min_value=min(available_dates), max_value=max(available_dates))
    
    df_day = df_plot[df_plot["time"].dt.date == selected_day].copy()

    if not df_day.empty:
        mae_day = mean_absolute_error(df_day["actual"], df_day["predicted"])
        r2_day = r2_score(df_day["actual"], df_day["predicted"])

        st.markdown('<div class="metric-card">'
                    '<div class="metric-label">Mean Absolute Error (MAE)</div>'
                    f'<div class="metric-value">{mae_day:.2f} kW</div>'
                    '</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card">'
                    '<div class="metric-label">R² Score</div>'
                    f'<div class="metric-value">{r2_day:.4f}</div>'
                    '</div>', unsafe_allow_html=True)
    else:
        st.warning("No data available for this day.")

with col2:
    if not df_day.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_day["time"], df_day["actual"], label="Actual", marker='o', linestyle='-', color='#00A0FF', markersize=5)
        ax.plot(df_day["time"], df_day["predicted"], label="Predicted", marker='x', linestyle='--', color='#FFC107', markersize=5)
        ax.set_title(f"Net Load on {selected_day}", fontsize=16)
        ax.set_xlabel("Hour of Day", fontsize=12)
        ax.set_ylabel("Net Load (kW)", fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#444444')
        ax.legend(fontsize=10)
        ax.tick_params(axis='x', labelrotation=45)
        plt.tight_layout()
        st.pyplot(fig)

# --- MONTHLY FORECAST ---
st.header("🗓️ Monthly Net Load Forecast")
col3, col4 = st.columns([1, 3])

with col3:
    st.subheader("Settings")
    df_plot["month"] = df_plot["time"].dt.to_period("M")
    available_months = sorted(df_plot["month"].unique())
    selected_month_str = st.selectbox("Select a month:", [m.strftime('%B %Y') for m in available_months])
    selected_month = pd.Period(selected_month_str, freq='M')

    df_month = df_plot[df_plot["month"] == selected_month].copy()

    if not df_month.empty:
        mae_month = mean_absolute_error(df_month["actual"], df_month["predicted"])
        r2_month = r2_score(df_month["actual"], df_month["predicted"])

        st.markdown('<div class="metric-card">'
                    '<div class="metric-label">Mean Absolute Error (MAE)</div>'
                    f'<div class="metric-value">{mae_month:.2f} kW</div>'
                    '</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card">'
                    '<div class="metric-label">R² Score</div>'
                    f'<div class="metric-value">{r2_month:.4f}</div>'
                    '</div>', unsafe_allow_html=True)
    else:
        st.warning("No data available for this month.")

with col4:
    if not df_month.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_month["time"], df_month["actual"], label="Actual", alpha=0.8, color='#00A0FF')
        ax.plot(df_month["time"], df_month["predicted"], label="Predicted", linestyle='--', alpha=0.7, color='#FFC107')
        ax.set_title(f"Net Load for {selected_month_str}", fontsize=16)
        ax.set_xlabel("Date & Time", fontsize=12)
        ax.set_ylabel("Net Load (kW)", fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#444444')
        ax.legend(fontsize=10)
        ax.tick_params(axis='x', labelrotation=45)
        plt.tight_layout()
        st.pyplot(fig)
