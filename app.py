import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import yfinance as yf
from datetime import datetime
import os

import traceback
try:
    import yfinance as yf
except Exception as e:
    st.error(traceback.format_exc())

st.title(" AI Stock Recommendation System")
st.write("Get personalized stock suggestions based on your preferences and real-time data.")


@st.cache_resource
def load_model_and_scaler():
    base_path=os.path.dirname(os.path.abspath(__file__))
    models_path=os.path.join(base_path,'models')

    scaler_path = os.path.join(models_path,'scaler.pkl')
    model_path = os.path.join(models_path,'stock_recommendor_model(2).pkl')

    if not  os.path.exists(scaler_path):
        raise FileNotFoundError("The Scaler file Didn't exist")
    if not os.path.exists(model_path):
        raise FileNotFoundError("The model file didn't exists ")
    with open(scaler_path, 'rb') as f:
        scaler = pkl.load(f)
    with open(model_path, "rb") as f:
        model = pkl.load(f)
    return scaler, model


scaler, model = load_model_and_scaler()

@st.cache_data
def live_fetch_data(symbols):
    all_data = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='6mo')

            if hist.empty:
                print(f"No historical data for {symbol}")
                continue

            volatality = hist['Close'].pct_change().std()  # keep the same typo for compatibility

            stock_data = {
                "Symbol": symbol,
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
                "Volatality_index": volatality,  # use SAME NAME as training
                "Quality_Score": 0.7,
                "Growth_Score": 0.6,
                "Sector": info.get("sector", "Unknown"),
            }
            all_data.append(stock_data)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")

    df = pd.DataFrame(all_data)
    return df



def predict_and_recommend(model, scaler, live_df, budget, sector_pref, horizon, target_stock):
    feature_cols = [
        "PE_Ratio", "EPS", "ROE", "DebtToEquity", "Price",
        "YearHigh", "YearLow", "AvgVolume", "Volatality",
        "Current_price", "Future_price", "Volatality_index",
        "Quality_Score", "Growth_Score"
    ]

   
    rename_map = {
        "Volatility": "Volatality",
        "Volatility_index": "Volatality_index"
    }
    live_df.rename(columns=rename_map, inplace=True)

    available_features = [f for f in feature_cols if f in live_df.columns]

    for col in feature_cols:
        if col not in live_df.columns:
            live_df[col] = 0

    X_scaled = scaler.transform(live_df[feature_cols])
    preds = model.predict(X_scaled)

    live_df['Predicted_score'] = preds
    df = live_df[live_df['Price'] <= budget].copy()


    if "Sector" not in df.columns:
        df['Sector'] = "Unknown"

    def clean_sector(s):
        if not isinstance(s, str):
            return "unknown"
        return s.lower().replace("information ", "").replace("&", "and").strip()

    df["Sector_clean"] = df["Sector"].apply(clean_sector)
    sector_pref_clean = sector_pref.lower().replace("information ", "").replace("&", "and").strip()

    if sector_pref_clean != "any":
        df = df[df["Sector_clean"].str.contains(sector_pref_clean, na=False)]

  
    if target_stock.upper() in df['Symbol'].values:
        target_sector = df.loc[df['Symbol'] == target_stock.upper(), "Sector_clean"].values[0]
        related_stock = df[df['Sector_clean'] == target_sector]
        df = pd.concat([related_stock, df]).drop_duplicates(subset="Symbol", keep='first')


    if horizon == 'short':
        df = df.sort_values(by='Volatality', ascending=True)
    elif horizon == 'long':
        df = df.sort_values(by="Growth_Score", ascending=False)

    return df.head(10)



st.subheader(" Input Parameters")

col1, col2, col3 = st.columns(3)
with col1:
    budget = st.number_input("Investment Budget ($)", value=10000, step=5000)
with col2:
    horizon = st.selectbox("Select Investment Horizon", ['short', 'long'])
with col3:
    sector_pref = st.text_input("Sector Preference (e.g. Technology, Healthcare, any)", "any")

target_stock = st.text_input("Enter Target Stock (optional, e.g. NVDA)", "")

st.markdown("---")

default_symbols = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "META", "AMZN",
    "JNJ", "PFE", "MRNA", "LLY",
    "JPM", "BAC", "GS", "MS",
    "XOM", "CVX", "COP",
    "PG", "KO", "PEP", "COST",
    "CAT", "HON", "GE", "UPS",
]

user_symbols = st.text_area("Enter Stock Symbols (comma-separated)", ",".join(default_symbols))
user_symbols = [s.strip().upper() for s in user_symbols.split(",") if s.strip()]


if st.button("🔍 Generate Recommendations"):
    with st.spinner("Fetching live data..."):
        df = live_fetch_data(user_symbols)

    if df.empty:
        st.warning(" No live data fetched. Please try again later.")
    else:
        recommendations = predict_and_recommend(model, scaler, df, budget, sector_pref, horizon, target_stock)
        if recommendations.empty:
            st.warning(f" No stocks matched your inputs. Available sectors: {df['Sector'].dropna().unique()}")
        else:
            st.success(" Top Recommended Stocks")
            st.dataframe(
                recommendations[["Symbol", "Sector", "Price", "Predicted_score"]],
                use_container_width=True
            )
