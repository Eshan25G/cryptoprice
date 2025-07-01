import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator

st.set_page_config(layout="wide")
st.title("🪙 Crypto Trading Signal & Chart Pattern Analyzer")

# Input Section
symbol = st.selectbox("Select Crypto Pair", ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD'])
interval = st.selectbox("Select Interval", ["1h", "1d", "1wk"])
period = st.selectbox("Select Time Range", ["7d", "30d", "90d", "180d"])

# Fetch Data
data = yf.download(tickers=symbol, interval=interval, period=period)
data.dropna(inplace=True)
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

# Use native pandas EMA to avoid issues
data['EMA10'] = data['Close'].ewm(span=10, adjust=False).mean()
data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()


# Indicators
data['EMA10'] = EMAIndicator(data['Close'], window=10).ema_indicator()
data['EMA50'] = EMAIndicator(data['Close'], window=50).ema_indicator()
data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()
macd = MACD(data['Close'])
data['MACD'] = macd.macd()
data['MACD_Signal'] = macd.macd_signal()

# Signal Generator
def generate_signal(df):
    if df['RSI'].iloc[-1] < 30 and df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and df['EMA10'].iloc[-1] > df['EMA50'].iloc[-1]:
        return "📈 BUY"
    elif df['RSI'].iloc[-1] > 70 and df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1] and df['EMA10'].iloc[-1] < df['EMA50'].iloc[-1]:
        return "📉 SELL"
    else:
        return "⏸️ NEUTRAL"

signal = generate_signal(data)
st.subheader(f"Signal for {symbol}: {signal}")
# Simple Candlestick Pattern Detection
def detect_patterns(df):
    latest = df.iloc[-2]
    patterns = []

    # Doji Detection
    if abs(latest['Close'] - latest['Open']) < (latest['High'] - latest['Low']) * 0.1:
        patterns.append("🟨 Doji")

    # Bullish Engulfing
    prev = df.iloc[-3]
    if prev['Close'] < prev['Open'] and latest['Close'] > latest['Open'] and latest['Close'] > prev['Open'] and latest['Open'] < prev['Close']:
        patterns.append("🟩 Bullish Engulfing")

    # Bearish Engulfing
    if prev['Close'] > prev['Open'] and latest['Close'] < latest['Open'] and latest['Close'] < prev['Open'] and latest['Open'] > prev['Close']:
        patterns.append("🟥 Bearish Engulfing")

    return patterns

patterns = detect_patterns(data)
st.subheader("📐 Chart Patterns Detected:")
if patterns:
    for p in patterns:
        st.markdown(f"- {p}")
else:
    st.markdown("No strong patterns detected.")

# Candlestick Chart
fig = go.Figure(data=[go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'],
                                     name='Candles')])
fig.add_trace(go.Scatter(x=data.index, y=data['EMA10'], line=dict(color='blue'), name='EMA10'))
fig.add_trace(go.Scatter(x=data.index, y=data['EMA50'], line=dict(color='orange'), name='EMA50'))
fig.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# RSI and MACD Plot
with st.expander("📊 RSI & MACD"):
    st.line_chart(data[['RSI']])
    st.line_chart(data[['MACD', 'MACD_Signal']])
