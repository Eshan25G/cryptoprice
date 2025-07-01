import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# Function to fetch historical data (using Binance as a proxy, since CoinSwitch API is not public)
def get_binance_klines(symbol, interval='15m', lookback=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={lookback}"
    try:
        data = requests.get(url, timeout=10).json()
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame()  # Empty DataFrame if no data or error
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df[['time', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# Candlestick pattern detection functions
def is_bullish_engulfing(df):
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    return (
        prev['close'] < prev['open'] and
        curr['close'] > curr['open'] and
        curr['close'] > prev['open'] and
        curr['open'] < prev['close']
    )

def is_bearish_engulfing(df):
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    return (
        prev['close'] > prev['open'] and
        curr['close'] < curr['open'] and
        curr['open'] > prev['close'] and
        curr['close'] < prev['open']
    )

def is_hammer(df):
    if len(df) < 1:
        return False
    curr = df.iloc[-1]
    body = abs(curr['close'] - curr['open'])
    lower_shadow = min(curr['open'], curr['close']) - curr['low']
    upper_shadow = curr['high'] - max(curr['open'], curr['close'])
    candle_length = curr['high'] - curr['low']
    return (
        candle_length > 0 and
        body < candle_length * 0.3 and
        lower_shadow > body * 2 and
        upper_shadow < body
    )

def is_doji(df):
    if len(df) < 1:
        return False
    curr = df.iloc[-1]
    return abs(curr['close'] - curr['open']) < (curr['high'] - curr['low']) * 0.1

# Generate trading signals based on candlestick patterns
def generate_signal(df):
    if len(df) < 2:
        return None, None  # Not enough data for candlestick patterns
    if is_bullish_engulfing(df):
        return "Bullish Engulfing", "long"
    if is_bearish_engulfing(df):
        return "Bearish Engulfing", "short"
    if is_hammer(df):
        return "Hammer", "long"
    if is_doji(df):
        return "Doji", "neutral"
    return None, None

# SL/TP Calculation (simple: 1.5 x last candle ATR)
def calc_sl_tp(df, signal_type):
    if len(df) < 14:
        atr = df['high'].max() - df['low'].min()  # fallback if not enough data
    else:
        atr = df['high'].iloc[-14:].max() - df['low'].iloc[-14:].min()
    entry = df['close'].iloc[-1]
    if signal_type == "long":
        sl = entry - atr * 0.8
        tp = entry + atr * 1.5
    elif signal_type == "short":
        sl = entry + atr * 0.8
        tp = entry - atr * 1.5
    else:
        sl, tp = None, None
    return sl, tp

# Streamlit UI
st.set_page_config(page_title="Crypto Futures Signal Generator", layout="wide")
st.title("Crypto Futures Signal Generator (ETH, SOL, XRP) - Candlestick Patterns")

symbols = {
    "ETHUSDT": "Ethereum",
    "SOLUSDT": "Solana",
    "XRPUSDT": "XRP"
}

interval = st.selectbox("Candle Interval", ["15m", "1h", "4h"], index=0)
selected_symbols = st.multiselect("Select Coins", list(symbols.keys()), default=list(symbols.keys()))

for sym in selected_symbols:
    st.subheader(f"{symbols[sym]} ({sym})")
    df = get_binance_klines(sym, interval, lookback=200)
    if df is None or len(df) < 2:
        st.warning("Not enough data to generate signals.")
        continue
    pattern, direction = generate_signal(df)
    if pattern:
        sl, tp = calc_sl_tp(df, direction)
        st.success(f"Pattern Detected: **{pattern}** | Signal: **{direction.upper()}**")
        st.write(f"Entry: `{df['close'].iloc[-1]:.2f}` | SL: `{sl:.2f}` | TP: `{tp:.2f}`")
        st.line_chart(df.set_index('time')['close'][-50:])
    else:
        st.warning("No tradable candlestick pattern detected in the latest candle.")

st.info("Signals are for educational purposes. Always backtest and use proper risk management.")
