import streamlit as st
import pandas as pd
import psycopg
from dotenv import load_dotenv
import os

# results = [
#     {"Modell": "XGBoost (Tuned)", "MAE": 6.755709, "R²": 0.813803},
#     {"Modell": "Gradient Boosting (Tuned)", "MAE": 6.804205, "R²": 0.812173},
#     {"Modell": "LightGBM (Tuned)", "MAE": 6.825612, "R²": 0.810949},
#     {"Modell": "SVR (Tuned)", "MAE": 6.809984, "R²": 0.806794},
#     {"Modell": "Random Forest (Tuned)", "MAE": 6.904226, "R²": 0.804118},
#     {"Modell": "KNN (Tuned)", "MAE": 7.022471, "R²": 0.798374},
#     {"Modell": "Ridge Regression (Tuned)", "MAE": 7.411271, "R²": 0.781098},
#     {"Modell": "Lasso Regression (Tuned)", "MAE": 7.411931, "R²": 0.781085},

# ]


# st.title("Model Comparison Results")
# results_df = pd.DataFrame(results).sort_values("R²", ascending=False)
# st.dataframe(
#     results_df.style
#     .background_gradient(cmap="Blues", subset=["R²"])
#     .format({"MSE": "{:.2f}", "R²": "{:.4f}"})
# )

# === Umgebungsvariablen laden (für spätere Nutzung, z. B. für Supabase) ===
load_dotenv()
db_url = os.getenv("SUPABASE_DB_URL")

# === Forecast – Daily ===
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("../data/eda_b4s_clean_d-2.csv")
df["time"] = pd.to_datetime(df["time"])
features = ['feature_11', 'hour', 'weekday', 'is_weekend', 'Holiday', 'Net Load (d-1)']
target = 'Nettolast_P_kW'

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell mit (optional) besten Parametern
model = XGBRegressor(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=200,
    subsample=0.8,
    random_state=42
)

# Trainieren und Vorhersage
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plot-Daten vorbereiten
df_plot = pd.DataFrame({
    "time": df.loc[y_test.index, "time"].values,
    "actual": y_test.values,
    "predicted": y_pred
}).sort_values("time").reset_index(drop=True)





# 📊 Daily Forecast
st.title("Daily Net Load Forecast")
available_dates = sorted(df_plot["time"].dt.date.unique())
selected_day = st.date_input("Select a date:", value=available_dates[0],
                             min_value=min(available_dates), max_value=max(available_dates))
df_day = df_plot[df_plot["time"].dt.date == selected_day].copy()
df_day = df_day.sort_values("time")

if df_day.empty:
    st.warning("No data available for this day.")
else:
    df_day["hour_decimal"] = df_day["time"].dt.hour + df_day["time"].dt.minute / 60
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_day["hour_decimal"], df_day["actual"], label="Actual", marker='o')
    ax.plot(df_day["hour_decimal"], df_day["predicted"], label="Predicted", marker='x', linestyle='--')
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Net Load (kW)")
    ax.set_title(f"Net Load Forecast on {selected_day}")
    ax.set_xticks(range(0, 24))
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    mse = mean_squared_error(df_day["actual"], df_day["predicted"])
    r2 = r2_score(df_day["actual"], df_day["predicted"])
    st.subheader("Error Metrics")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**R²:** {r2:.4f}")

# 📊 Monthly Forecast
st.title("Monthly Net Load Forecast")
df_plot["month"] = df_plot["time"].dt.to_period("M")
available_months = sorted(df_plot["month"].unique())
selected_month = st.selectbox("Select a month:", available_months)
df_month = df_plot[df_plot["month"] == selected_month].copy()
df_month = df_month.sort_values("time")

if df_month.empty:
    st.warning("No data available for this month.")
else:
    df_month["hour"] = df_month["time"].dt.hour
    df_month["date_str"] = df_month["time"].dt.strftime("%Y-%m-%d %H:%M")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_month["time"], df_month["actual"], label="Actual", alpha=0.7)
    ax.plot(df_month["time"], df_month["predicted"], label="Predicted", linestyle='--', alpha=0.7)
    ax.set_title(f"Net Load Forecast – {selected_month}")
    ax.set_xlabel("Date & Time")
    ax.set_ylabel("Net Load (kW)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    mse = mean_squared_error(df_month["actual"], df_month["predicted"])
    r2 = r2_score(df_month["actual"], df_month["predicted"])
    st.subheader("Error Metrics")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**R²:** {r2:.4f}")
