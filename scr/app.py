import streamlit as st
import pandas as pd
import psycopg
from dotenv import load_dotenv
import os


import streamlit as st
import pandas as pd

# Your model results
results = [
    {"Model": "XGBoost (Tuned)", "MSE": 81.635382, "R²": 0.801149},
    {"Model": "Gradient Boosting (Tuned)", "MSE": 81.957831, "R²": 0.800364},
    {"Model": "LightGBM (Tuned)", "MSE": 81.967927, "R²": 0.800339},
    {"Model": "Random Forest (Tuned)", "MSE": 83.696020, "R²": 0.796130},
    {"Model": "KNN (Tuned)", "MSE": 102.520732, "R²": 0.750276},
    {"Model": "SVR (Tuned)", "MSE": 203.580247, "R²": 0.504112},
    {"Model": "Ridge Regression (Tuned)", "MSE": 220.444144, "R²": 0.463034},
    {"Model": "Lasso Regression (Tuned)", "MSE": 220.445116, "R²": 0.463032}
]

# Convert to DataFrame and sort
#schreibe einen titel für die Tabelle
st.title("Model Comparison Results")
results_df = pd.DataFrame(results).sort_values("R²", ascending=False)

# Display in Streamlit
st.dataframe(
    results_df.style
    .background_gradient(cmap="Blues", subset=["R²"])
    .format({"MSE": "{:.2f}", "R²": "{:.4f}"})
)






# === 1. Umgebungsvariablen laden ===
load_dotenv()
db_url = os.getenv("SUPABASE_DB_URL")

# === 2. Verbindung zur Datenbank ===
@st.cache_data(ttl=600)
def load_data():
    with psycopg.connect(db_url) as conn:
        df = pd.read_sql("SELECT * FROM b4s_data ORDER BY time ASC", conn)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df

df = load_data()

# === 3. Streamlit UI ===
st.title("📊 Netzlast – Gebäude B4S")
st.markdown("Darstellung von Wirkleistung und Blindleistung aus Supabase")

# Debug: Spalten anzeigen
st.sidebar.write("🧾 Verfügbare Spalten:", df.columns.tolist())

# Zeitbereichsauswahl
start, end = st.slider(
    "Zeitraum wählen:",
    min_value=df["time"].min().to_pydatetime(),
    max_value=df["time"].max().to_pydatetime(),
    value=(df["time"].min().to_pydatetime(), df["time"].min().to_pydatetime() + pd.Timedelta(days=1)),
    format="DD.MM.YYYY HH:mm"
)

mask = (df["time"] >= start) & (df["time"] <= end)
filtered = df.loc[mask]

# === 4. Dynamische Auswahl numerischer Spalten ===
numeric_cols = df.select_dtypes(include="number").columns.tolist()
default_cols = [col for col in numeric_cols if "P" in col or "Q" in col]

selected_cols = st.multiselect(
    "Spalten auswählen für das Diagramm:",
    options=numeric_cols,
    default=default_cols
)

# === 5. Liniendiagramm anzeigen ===
if selected_cols:
    st.line_chart(filtered.set_index("time")[selected_cols], height=400)
else:
    st.warning("Bitte wähle mindestens eine numerische Spalte aus.")

# === 6. Tabelle optional anzeigen ===
with st.expander("📄 Rohdaten anzeigen"):
    st.dataframe(filtered)

# === 7. Hinweis ===
st.markdown("Hinweis: Die Daten stammen aus einem Smart-Meter eines Gebäudes im Smart-City-Projekt.")






import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 📂 Load data
df = pd.read_csv("../data/eda_b4s_clean.csv")
df["time"] = pd.to_datetime(df["time"])

# 🎯 Define features and target
features = ['gre000h0', 'hour', 'weekday', 'is_weekend', 'Holiday']
target = 'Nettolast_P_kW'

X = df[features]
y = df[target]

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔁 Train model
model = XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📈 Prepare prediction data
df_plot = pd.DataFrame({
    "time": df.loc[y_test.index, "time"].values,
    "actual": y_test.values,
    "predicted": y_pred
})
df_plot["time"] = pd.to_datetime(df_plot["time"])

# 🧭 Streamlit UI
st.title("Daily Net Load Forecast ")

# ⏱️ Date selection
available_dates = sorted(df_plot["time"].dt.date.unique())
selected_day = st.date_input("Select a date:", value=available_dates[0],
                             min_value=min(available_dates), max_value=max(available_dates))

# 🔍 Filter by selected day
df_day = df_plot[df_plot["time"].dt.date == selected_day].copy()
df_day = df_day.sort_values("time")

if df_day.empty:
    st.warning("No data available for this day.")
else:
    df_day["hour_decimal"] = df_day["time"].dt.hour + df_day["time"].dt.minute / 60

    # 📊 Plot
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

    # 📉 Show error metrics
    mse = mean_squared_error(df_day["actual"], df_day["predicted"])
    r2 = r2_score(df_day["actual"], df_day["predicted"])
    st.subheader("Error Metrics")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**R²:** {r2:.4f}")




import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 📂 Load data
df = pd.read_csv("../data/eda_b4s_clean.csv")
df["time"] = pd.to_datetime(df["time"])

# 🎯 Define features and target
features = ['gre000h0', 'hour', 'weekday', 'is_weekend', 'Holiday']
target = 'Nettolast_P_kW'

X = df[features]
y = df[target]

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔁 Train model
model = XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📈 Prepare prediction data
df_plot = pd.DataFrame({
    "time": df.loc[y_test.index, "time"].values,
    "actual": y_test.values,
    "predicted": y_pred
})
df_plot["time"] = pd.to_datetime(df_plot["time"])
df_plot["month"] = df_plot["time"].dt.to_period("M")  # YYYY-MM format

# 🧭 Streamlit UI
st.title("Monthly Net Load Forecast ")

# Get list of months available in the data
available_months = sorted(df_plot["month"].unique())

# Let the user select a month
selected_month = st.selectbox("Select a month:", available_months)

# Filter by selected month
df_month = df_plot[df_plot["month"] == selected_month].copy()
df_month = df_month.sort_values("time")

if df_month.empty:
    st.warning("No data available for this month.")
else:
    # Optional: Show day and hour as labels
    df_month["hour"] = df_month["time"].dt.hour
    df_month["date_str"] = df_month["time"].dt.strftime("%Y-%m-%d %H:%M")

    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_month["time"], df_month["actual"], label="Actual", alpha=0.7)
    ax.plot(df_month["time"], df_month["predicted"], label="Predicted", linestyle='--', alpha=0.7)
    ax.set_title(f"Net Load Forecast – {selected_month}")
    ax.set_xlabel("Date & Time")
    ax.set_ylabel("Net Load (kW)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Show metrics
    mse = mean_squared_error(df_month["actual"], df_month["predicted"])
    r2 = r2_score(df_month["actual"], df_month["predicted"])
    st.subheader("Error Metrics")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**R²:** {r2:.4f}")








