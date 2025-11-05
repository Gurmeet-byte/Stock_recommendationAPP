import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import yfinance as yf
import os
from datetime import datetime

# ------------------------------------------------------
# 🧠 Title + Introduction
# ------------------------------------------------------
st.set_page_config(page_title="AI Stock Recommendation System", layout="wide")
st.title("💹 AI Stock Recommendation System")
st.write("Get personalized stock suggestions based on your investment preferences and real-time market data.")


# ------------------------------------------------------
# 🧩 Model + Scaler Loading
# ------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    base_path = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(base_path, "models")

    scaler_path = os.path.join(models_path, "scaler.pkl")
    model_path = os.path.join(models_path, "stock_recommendor_model(2).pkl")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler file not found in models/")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found in models/")

    with open(scaler_path, "rb") as f:
        scaler = pkl.load(f)
    with open(model_path, "rb") as f:
        model = pkl.load(f)
    return scaler, model


scaler, model = load_model_and_scaler()


# ------------------------------------------------------
# ⚙️ Live Data Fetching
# ------------------------------------------------------
@st.cache_data
def live_fetch_data(symbols):
    all_data = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="6mo")

            if hist.empty:
                print(f"No historical data for {symbol}")
                continue

            volatality = hist["Close"].pct_change().std()

            # Extract company info with fallbacks
            ceo = ""
            if "companyOfficers" in info and isinstance(info["companyOfficers"], list) and len(info["companyOfficers"]) > 0:
                ceo = info["companyOfficers"][0].get("name", "")

            stock_data = {
                "Symbol": symbol,
                "Company": info.get("longName", symbol),
                "Sector": info.get("sector", "Unknown"),
                "Industry": info.get("industry", "Unknown"),
                "CEO": ceo,
                "Website": info.get("website", ""),
                "Description": info.get("longBusinessSummary", "No description available."),
                "Employees": info.get("fullTimeEmployees", 0),

                # Financial metrics
                "PE_Ratio": info.get("trailingPE", 0),
                "EPS": info.get("trailingEps", 0),
                "ROE": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else 0,
                "DebtToEquity": info.get("debtToEquity", 0),
                "Price": info.get("currentPrice", 0),
                "YearHigh": info.get("fiftyTwoWeekHigh", 0),
                "YearLow": info.get("fiftyTwoWeekLow", 0),
                "AvgVolume": info.get("averageVolume", 0),
                "Volatality": volatality,
                "Current_price": info.get("currentPrice", 0),
                "Future_price": info.get("targetMeanPrice", 0),
                "Volatality_index": volatality,
                "Quality_Score": 0.7,
                "Growth_Score": 0.6,
            }

            all_data.append(stock_data)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")

    return pd.DataFrame(all_data)


# ------------------------------------------------------
# 🧮 Prediction + Recommendation Logic
# ------------------------------------------------------
def predict_and_recommend(model, scaler, live_df, budget, sector_pref, horizon, target_stock):
    feature_cols = [
        "PE_Ratio", "EPS", "ROE", "DebtToEquity", "Price",
        "YearHigh", "YearLow", "AvgVolume", "Volatality",
        "Current_price", "Future_price", "Volatality_index",
        "Quality_Score", "Growth_Score"
    ]

    for col in feature_cols:
        if col not in live_df.columns:
            live_df[col] = 0

    X_scaled = scaler.transform(live_df[feature_cols])
    preds = model.predict(X_scaled)

    live_df["Predicted_score"] = preds
    df = live_df[live_df["Price"] <= budget].copy()

    if sector_pref.lower() != "any":
        df = df[df["Sector"].str.lower().str.contains(sector_pref.lower(), na=False)]

    if target_stock.upper() in df["Symbol"].values:
        target_sector = df.loc[df["Symbol"] == target_stock.upper(), "Sector"].values[0]
        related_stock = df[df["Sector"] == target_sector]
        df = pd.concat([related_stock, df]).drop_duplicates(subset="Symbol", keep="first")

    if horizon == "short":
        df = df.sort_values(by="Volatality", ascending=True)
    elif horizon == "long":
        df = df.sort_values(by="Growth_Score", ascending=False)

    return df.head(10)


# ------------------------------------------------------
# 🧍 User Input UI
# ------------------------------------------------------
st.subheader("📊 Input Parameters")

col1, col2, col3 = st.columns(3)
with col1:
    budget = st.number_input("Investment Budget ($)", value=10000, step=5000)
with col2:
    horizon = st.selectbox("Investment Horizon", ["short", "long"])
with col3:
    sector_pref = st.selectbox(
        "Sector Preference",
        ["any", "technology", "healthcare", "financial", "energy", "consumer goods", "industrial","utilities"]
    )

target_stock = st.text_input("Target Stock (optional, e.g. NVDA)", "")

st.markdown("---")

# Predefined symbols
default_symbols = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN",
    "JPM", "BAC", "GS", "CVX", "XOM", "COP",
    "PFE", "JNJ", "LLY", "MRNA",
    "CAT", "HON", "UPS", "WMT", "KO", "PEP", "COST",
]

user_symbols = st.text_area("Enter Stock Symbols (comma-separated)", ",".join(default_symbols))
user_symbols = [s.strip().upper() for s in user_symbols.split(",") if s.strip()]


# ------------------------------------------------------
# 🚀 Recommendation Generator
# ------------------------------------------------------
if st.button("🔍 Generate Recommendations"):
    with st.spinner("Fetching live market data..."):
        df = live_fetch_data(user_symbols)

    if df.empty:
        st.warning("No live data fetched. Try again or use other symbols.")
    else:
        recommendations = predict_and_recommend(model, scaler, df, budget, sector_pref, horizon, target_stock)
        if recommendations.empty:
            st.warning("No stocks matched your filters.")
        else:
            st.success("✅ Top Recommended Stocks")

            for _, row in recommendations.iterrows():
                with st.expander(f"{row['Symbol']} — {row['Company']} (${row['Price']})"):
                    st.write(f"**Sector:** {row['Sector']} | **Industry:** {row['Industry']}")
                    st.write(f"**CEO:** {row['CEO'] or 'N/A'} | **Employees:** {row['Employees']:,}")
                    st.write(f"**Website:** [{row['Website']}]({row['Website']})")
                    
                    st.write(f"**Description:** {row['Description'][:400]}{'...' if len(row['Description']) > 400 else ''}")



