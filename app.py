import streamlit as st
import pandas as pd
import requests
import ta
import os
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ------------------- Config -------------------
st.set_page_config(page_title="Crypto Trading Dashboard", layout="wide")
st.title("📊 Real-Time Crypto Signal Dashboard")
st.caption("Powered by Binance API + Streamlit")

st_autorefresh(interval=30000, key="datarefresh")  # Auto-refresh every 30s

symbols = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "SOL/USDT": "SOLUSDT",
    "ADA/USDT": "ADAUSDT",
    "XRP/USDT": "XRPUSDT",
    "DOGE/USDT": "DOGEUSDT",
    "LTC/USDT": "LTCUSDT",
    "MATIC/USDT": "MATICUSDT",
    "DOT/USDT": "DOTUSDT",
    "AVAX/USDT": "AVAXUSDT"
}

# ------------------- Signal Generator -------------------
def generate_signal(df):
    latest = df.iloc[-1]
    if latest['RSI'] < 30 and latest['MACD'] > latest['MACD_Signal'] and latest['EMA10'] > latest['EMA50']:
        return "📈 BUY"
    elif latest['RSI'] > 70 and latest['MACD'] < latest['MACD_Signal'] and latest['EMA10'] < latest['EMA50']:
        return "📉 SELL"
    else:
        return "⏸️ NEUTRAL"

# ------------------- Trade Logger -------------------
def log_trade(symbol, signal, price):
    log_entry = {
        "timestamp": datetime.now(),
        "symbol": symbol,
        "signal": signal,
        "price": price
    }
    if os.path.exists("trades.csv"):
        df = pd.read_csv("trades.csv")
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])
    df.to_csv("trades.csv", index=False)

# ------------------- Get OHLCV Data -------------------
def get_ohlcv(symbol, interval='1m', limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "time", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"
    ])
    df["time"] = pd.to_datetime(df["time"], unit='ms')
    df.set_index("time", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    df['EMA10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    return df

# ------------------- Backtest -------------------
def backtest(df):
    cash = 1000
    position = 0
    trades = []
    for i in range(1, len(df)):
        if df['RSI'].iloc[i] < 30 and df['EMA10'].iloc[i] > df['EMA50'].iloc[i]:
            if position == 0:
                position = cash / df['close'].iloc[i]
                cash = 0
                trades.append(('BUY', df.index[i], df['close'].iloc[i]))
        elif df['RSI'].iloc[i] > 70 and df['EMA10'].iloc[i] < df['EMA50'].iloc[i]:
            if position > 0:
                cash = position * df['close'].iloc[i]
                position = 0
                trades.append(('SELL', df.index[i], df['close'].iloc[i]))

    final_value = cash + (position * df['close'].iloc[-1])
    return trades, final_value

# ------------------- Scan Multiple Coins -------------------
dashboard_data = []

for label, ticker in symbols.items():
    df = get_ohlcv(ticker)
    if df.empty or df.isnull().values.any():
        continue
    signal = generate_signal(df)
    price = df['close'].iloc[-1]
    change = (price - df['close'].iloc[0]) / df['close'].iloc[0] * 100
    dashboard_data.append({
        "Symbol": label,
        "Price (USDT)": round(price, 2),
        "Change (%)": round(change, 2),
        "Signal": signal
    })

    # Optional: log signal
    if signal in ["📈 BUY", "📉 SELL"]:
        log_trade(label, signal, price)

# ------------------- Show Dashboard -------------------
st.subheader("📈 Live Signals")
df_signals = pd.DataFrame(dashboard_data)
if not df_signals.empty and "Signal" in df_signals.columns:
    styled_df = df_signals.style.applymap(
        lambda x: "color: green" if x == "📈 BUY" else "color: red" if x == "📉 SELL" else "",
        subset=["Signal"]
    )
    st.dataframe(styled_df)
else:
    st.warning("⚠️ No signal data available.")


# ------------------- Trade Log -------------------
if os.path.exists("trades.csv"):
    st.subheader("📘 Trade Log")
    trade_log = pd.read_csv("trades.csv")
    st.dataframe(trade_log)

# ------------------- Backtesting -------------------
st.subheader("🔁 Backtest a Coin")
coin_choice = st.selectbox("Choose Coin for Backtesting", list(symbols.keys()))
df_backtest = get_ohlcv(symbols[coin_choice], interval="1h", limit=500)

if df_backtest.empty:
    st.error("⚠️ No data returned for backtest. Try another coin or check API rate limits.")
else:
    trades, final_value = backtest(df_backtest)
    st.write(f"💰 Final Portfolio Value (Starting $1000): **${final_value:.2f}**")

    st.write("📋 Trades:")
    if trades:
        st.table(pd.DataFrame(trades, columns=["Type", "Time", "Price"]))
    else:
        st.info("No trades executed in backtest period.")
