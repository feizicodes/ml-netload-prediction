import streamlit as st
import pandas as pd
import psycopg
from dotenv import load_dotenv
import os

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
