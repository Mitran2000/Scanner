# Ultimate AI Forex Intelligence Platform - Next Generation
# Powered by TradingView DataFeed + Advanced Quant Modules

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import requests
from datetime import datetime
from tvDatafeed import TvDatafeed, Interval

# ===== ULTRA PROFESSIONAL UI CONFIG =====
st.set_page_config(page_title="Quantum Forex AI", layout="wide")
st.title("QUANTUM FOREX AI – Institutional Grade Analysis System")

# ===== TELEGRAM CONFIG =====
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

# ===== INITIALIZE DATA SOURCE =====
tv = TvDatafeed()

# ===== SYMBOL UNIVERSE =====
AVAILABLE_SYMBOLS = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD",
    "XAUUSD","XAGUSD","EURJPY","GBPJPY","AUDJPY","EURGBP"
]

symbol = st.selectbox("Select Instrument", AVAILABLE_SYMBOLS)

TIMEFRAME_MAP = {
    "1m": Interval.in_1_minute,
    "5m": Interval.in_5_minute,
    "15m": Interval.in_15_minute,
    "30m": Interval.in_30_minute,
    "1h": Interval.in_1_hour,
    "4h": Interval.in_4_hour,
    "1d": Interval.in_daily
}

timeframes = st.multiselect("Timeframes", list(TIMEFRAME_MAP.keys()), default=["15m","1h"])

# ===== ADVANCED DATA FETCHER =====
def load_tv_data(symbol, timeframe, candles=400):
    try:
        data = tv.get_hist(symbol=symbol, exchange='OANDA', interval=TIMEFRAME_MAP[timeframe], n_bars=candles)
        data = data.reset_index()
        data.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','datetime':'time'}, inplace=True)
        return data
    except Exception:
        return None

# ===== INSTITUTIONAL INDICATORS =====
def add_indicators(data):
    data['EMA_9'] = data['Close'].ewm(span=9).mean()
    data['EMA_21'] = data['Close'].ewm(span=21).mean()
    data['EMA_50'] = data['Close'].ewm(span=50).mean()

    data['Trend'] = np.where(data['EMA_9'] > data['EMA_21'], 'UPTREND', 'DOWNTREND')

    data['Momentum'] = data['Close'].pct_change(10)
    data['Volatility'] = data['Close'].rolling(20).std()
    data['Range'] = data['High'] - data['Low']

    data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().rolling(14).mean()))

    return data.dropna()

# ===== SMART PATTERN RECOGNITION =====
def detect_candlestick(data):
    patterns = []
    data = data.reset_index(drop=True)

    for i in range(len(data)):
        o,c,h,l = data['Open'].iloc[i], data['Close'].iloc[i], data['High'].iloc[i], data['Low'].iloc[i]
        body = abs(c-o)
        wick = h-max(c,o)

        if c>o and wick<body*0.3:
            patterns.append("Bullish Impulse")
        elif o>c and wick<body*0.3:
            patterns.append("Bearish Impulse")
        elif body<(h-l)*0.2:
            patterns.append("Indecision")
        else:
            patterns.append("Neutral")

    data['Pattern'] = patterns
    return data

# ===== AI ENGINE – MULTI FEATURE LEARNING =====
def train_ai_model(data):
    if data is None or len(data)<60:
        return None

    features = data[['Close','Momentum','Volatility','Range']].dropna()
    scaler = MinMaxScaler()

    try:
        scaled = scaler.fit_transform(features)
    except Exception:
        return None

    X,y = [],[]
    for i in range(10,len(scaled)):
        X.append(scaled[i-10:i].flatten())
        y.append(scaled[i,0])

    if len(X)==0:
        return None

    X,y = np.array(X), np.array(y)

    try:
        model = MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=400)
        model.fit(X,y)
        pred = model.predict(scaled[-10:].flatten().reshape(1,-1))
        return scaler.inverse_transform([[pred[0],0,0,0]])[0][0]
    except Exception:
        return None

# ===== MARKET REGIME DETECTOR =====
def detect_market_state(data):
    vol = data['Volatility'].iloc[-1]
    mom = data['Momentum'].iloc[-1]

    if vol>data['Volatility'].mean() and abs(mom)>0.001:
        return "TRENDING"
    elif vol<data['Volatility'].mean():
        return "RANGING"
    return "CHOPPY"

# ===== TELEGRAM ALERT ENGINE =====
def send_telegram(msg):
    if TELEGRAM_BOT_TOKEN!="YOUR_BOT_TOKEN":
        url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url,data={"chat_id":TELEGRAM_CHAT_ID,"text":msg})

# ===== EXECUTION CORE =====
if st.button("Run Quantum Analysis"):
    for tf in timeframes:
        st.subheader(f"Analysis – {symbol} – {tf}")

        data = load_tv_data(symbol, tf)
        if data is None:
            st.error("Data Unavailable")
            continue

        data = add_indicators(data)
        data = detect_candlestick(data)

        prediction = train_ai_model(data)
        current = data['Close'].iloc[-1]

        state = detect_market_state(data)
        trend = data['Trend'].iloc[-1]

        st.dataframe(data.tail(5), use_container_width=True)

        signal = "BUY" if prediction and prediction>current else "SELL"

        st.success(f"AI Signal: {signal}")
        st.info(f"Trend: {trend} | Market State: {state}")

        confidence = abs(prediction-current) if prediction else 0
        st.metric("Prediction Confidence", round(confidence,5))

        msg=f"{symbol} {tf} -> {signal} | Trend:{trend} | State:{state}"
        send_telegram(msg)

st.write("Quantum AI Engine Ready – Institutional Level Insights Delivered")
