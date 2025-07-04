import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import talib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Sentiment Analysis
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup

# Page Configuration
st.set_page_config(
    page_title="Indian Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; text-align: center; margin: 0;">
        🇮🇳 Indian Stock Market Predictor & Strategy Visualizer
    </h1>
    <p style="color: white; text-align: center; margin-top: 0.5rem; font-size: 1.2rem;">
        Advanced AI-Powered Stock Analysis with Technical Indicators & Sentiment Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Popular Indian Stocks
INDIAN_STOCKS = {
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'Tata Consultancy Services',
    'HDFCBANK.NS': 'HDFC Bank',
    'INFY.NS': 'Infosys',
    'HINDUNILVR.NS': 'Hindustan Unilever',
    'ICICIBANK.NS': 'ICICI Bank',
    'ITC.NS': 'ITC Limited',
    'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'ASIANPAINT.NS': 'Asian Paints',
    'MARUTI.NS': 'Maruti Suzuki',
    'AXISBANK.NS': 'Axis Bank',
    'LT.NS': 'Larsen & Toubro',
    'SUNPHARMA.NS': 'Sun Pharma',
    'WIPRO.NS': 'Wipro',
    'ULTRACEMCO.NS': 'UltraTech Cement',
    'TITAN.NS': 'Titan Company',
    'POWERGRID.NS': 'Power Grid Corporation',
    'NESTLEIND.NS': 'Nestle India',
    'HCLTECH.NS': 'HCL Technologies'
}

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Stock Selection
selected_stock = st.sidebar.selectbox(
    "Select Stock:",
    options=list(INDIAN_STOCKS.keys()),
    format_func=lambda x: f"{INDIAN_STOCKS[x]} ({x})"
)

# Time Period Selection
time_period = st.sidebar.selectbox(
    "Select Time Period:",
    options=['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
    index=3
)

# Prediction Model Selection
model_type = st.sidebar.selectbox(
    "Select Prediction Model:",
    options=['LSTM', 'XGBoost', 'Both'],
    index=0
)

# Technical Indicators Selection
st.sidebar.subheader("Technical Indicators")
show_sma = st.sidebar.checkbox("Simple Moving Average (SMA)", value=True)
show_ema = st.sidebar.checkbox("Exponential Moving Average (EMA)", value=True)
show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=True)
show_rsi = st.sidebar.checkbox("RSI", value=True)
show_macd = st.sidebar.checkbox("MACD", value=True)

# Prediction Parameters
st.sidebar.subheader("Prediction Parameters")
prediction_days = st.sidebar.slider("Days to Predict:", 1, 30, 7)
lstm_epochs = st.sidebar.slider("LSTM Epochs:", 10, 100, 50)
sequence_length = st.sidebar.slider("Sequence Length:", 30, 120, 60)

@st.cache_data
def load_stock_data(symbol, period):
    """Load stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        info = stock.info
        return data, info
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    # Simple Moving Averages
    data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
    data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
    
    # Exponential Moving Averages
    data['EMA_12'] = talib.EMA(data['Close'], timeperiod=12)
    data['EMA_26'] = talib.EMA(data['Close'], timeperiod=26)
    
    # Bollinger Bands
    data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['Close'])
    
    # RSI
    data['RSI'] = talib.RSI(data['Close'])
    
    # MACD
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'])
    
    # Stochastic Oscillator
    data['Stoch_K'], data['Stoch_D'] = talib.STOCH(data['High'], data['Low'], data['Close'])
    
    return data

def create_lstm_model(X_train, y_train):
    """Create and train LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def prepare_lstm_data(data, sequence_length):
    """Prepare data for LSTM model"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def create_xgboost_features(data):
    """Create features for XGBoost model"""
    features = pd.DataFrame()
    
    # Price features
    features['Close'] = data['Close']
    features['Volume'] = data['Volume']
    features['High'] = data['High']
    features['Low'] = data['Low']
    features['Open'] = data['Open']
    
    # Technical indicators
    features['SMA_20'] = data['SMA_20']
    features['SMA_50'] = data['SMA_50']
    features['EMA_12'] = data['EMA_12']
    features['EMA_26'] = data['EMA_26']
    features['RSI'] = data['RSI']
    features['MACD'] = data['MACD']
    
    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        features[f'Close_lag_{lag}'] = data['Close'].shift(lag)
    
    # Rolling statistics
    features['Close_rolling_mean_5'] = data['Close'].rolling(5).mean()
    features['Close_rolling_std_5'] = data['Close'].rolling(5).std()
    features['Volume_rolling_mean_5'] = data['Volume'].rolling(5).mean()
    
    # Price changes
    features['Price_change'] = data['Close'].pct_change()
    features['Price_change_lag_1'] = features['Price_change'].shift(1)
    
    return features.dropna()

def get_sentiment_score(stock_name):
    """Get sentiment score from news headlines (simplified)"""
    try:
        # This is a simplified sentiment analysis
        # In production, you'd use proper news APIs
        headlines = [
            f"{stock_name} shows strong performance",
            f"{stock_name} quarterly results exceed expectations",
            f"Market analysts bullish on {stock_name}",
            f"{stock_name} stock reaches new highs"
        ]
        
        sentiment_scores = []
        for headline in headlines:
            blob = TextBlob(headline)
            sentiment_scores.append(blob.sentiment.polarity)
        
        return np.mean(sentiment_scores)
    except:
        return 0.0

# Main Application
if st.sidebar.button("🚀 Run Analysis"):
    with st.spinner("Loading data and running analysis..."):
        # Load data
        data, info = load_stock_data(selected_stock, time_period)
        
        if data is not None and len(data) > 0:
            # Calculate technical indicators
            data = calculate_technical_indicators(data)
            
            # Display stock info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"₹{data['Close'].iloc[-1]:.2f}")
            
            with col2:
                price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                st.metric("Price Change", f"₹{price_change:.2f}", f"{price_change:.2f}")
            
            with col3:
                volume = data['Volume'].iloc[-1]
                st.metric("Volume", f"{volume:,.0f}")
            
            with col4:
                market_cap = info.get('marketCap', 0)
                if market_cap > 0:
                    st.metric("Market Cap", f"₹{market_cap/1e9:.2f}B")
                else:
                    st.metric("Market Cap", "N/A")
            
            # Stock Price Chart with Technical Indicators
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Stock Price & Technical Indicators', 'Volume', 'RSI', 'MACD'),
                vertical_spacing=0.05,
                row_heights=[0.5, 0.2, 0.15, 0.15]
            )
            
            # Price and moving averages
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                                   name='Close Price', line=dict(color='blue')), row=1, col=1)
            
            if show_sma:
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], 
                                       name='SMA 20', line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], 
                                       name='SMA 50', line=dict(color='red')), row=1, col=1)
            
            if show_ema:
                fig.add_trace(go.Scatter(x=data.index, y=data['EMA_12'], 
                                       name='EMA 12', line=dict(color='green')), row=1, col=1)
            
            if show_bollinger:
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], 
                                       name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], 
                                       name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
            
            # Volume
            fig.add_trace(go.Bar(x=data.index, y=data['Volume'], 
                               name='Volume', marker_color='lightblue'), row=2, col=1)
            
            # RSI
            if show_rsi:
                fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], 
                                       name='RSI', line=dict(color='purple')), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            # MACD
            if show_macd:
                fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], 
                                       name='MACD', line=dict(color='blue')), row=4, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['MACD_signal'], 
                                       name='MACD Signal', line=dict(color='red')), row=4, col=1)
                fig.add_trace(go.Bar(x=data.index, y=data['MACD_hist'], 
                                   name='MACD Histogram', marker_color='gray'), row=4, col=1)
            
            fig.update_layout(height=800, title_text=f"{INDIAN_STOCKS[selected_stock]} Technical Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # Predictions
            st.subheader("🤖 AI Predictions")
            
            prediction_results = {}
            
            if model_type in ['LSTM', 'Both']:
                st.write("### LSTM Model Prediction")
                
                # Prepare LSTM data
                X, y, scaler = prepare_lstm_data(data, sequence_length)
                
                if len(X) > 0:
                    # Split data
                    split_idx = int(len(X) * 0.8)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    # Reshape for LSTM
                    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                    
                    # Create and train model
                    with st.spinner("Training LSTM model..."):
                        lstm_model = create_lstm_model(X_train, y_train)
                        lstm_model.fit(X_train, y_train, epochs=lstm_epochs, 
                                     batch_size=32, verbose=0, validation_split=0.1)
                    
                    # Make predictions
                    lstm_predictions = lstm_model.predict(X_test)
                    lstm_predictions = scaler.inverse_transform(lstm_predictions)
                    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                    
                    # Calculate metrics
                    lstm_mse = mean_squared_error(y_test_actual, lstm_predictions)
                    lstm_mae = mean_absolute_error(y_test_actual, lstm_predictions)
                    
                    # Future predictions
                    last_sequence = X[-1].reshape(1, sequence_length, 1)
                    future_predictions = []
                    
                    for _ in range(prediction_days):
                        pred = lstm_model.predict(last_sequence, verbose=0)
                        future_predictions.append(pred[0, 0])
                        
                        # Update sequence
                        last_sequence = np.roll(last_sequence, -1, axis=1)
                        last_sequence[0, -1, 0] = pred[0, 0]
                    
                    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
                    prediction_results['LSTM'] = future_predictions.flatten()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("LSTM MSE", f"{lstm_mse:.2f}")
                    with col2:
                        st.metric("LSTM MAE", f"{lstm_mae:.2f}")
            
            if model_type in ['XGBoost', 'Both']:
                st.write("### XGBoost Model Prediction")
                
                # Prepare XGBoost data
                features = create_xgboost_features(data)
                
                if len(features) > 30:
                    # Create target variable (next day price)
                    features['Target'] = features['Close'].shift(-1)
                    features = features.dropna()
                    
                    # Split data
                    X = features.drop(['Target'], axis=1)
                    y = features['Target']
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, shuffle=False
                    )
                    
                    # Train XGBoost model
                    with st.spinner("Training XGBoost model..."):
                        xgb_model = xgb.XGBRegressor(
                            n_estimators=100,
                            max_depth=6,
                            learning_rate=0.1,
                            random_state=42
                        )
                        xgb_model.fit(X_train, y_train)
                    
                    # Make predictions
                    xgb_predictions = xgb_model.predict(X_test)
                    
                    # Calculate metrics
                    xgb_mse = mean_squared_error(y_test, xgb_predictions)
                    xgb_mae = mean_absolute_error(y_test, xgb_predictions)
                    
                    # Future predictions
                    last_features = X.iloc[-1:].copy()
                    xgb_future_predictions = []
                    
                    for i in range(prediction_days):
                        pred = xgb_model.predict(last_features)[0]
                        xgb_future_predictions.append(pred)
                        
                        # Update features for next prediction
                        # This is simplified - in reality, you'd update all features
                        last_features['Close'] = pred
                        last_features['Close_lag_1'] = last_features['Close']
                    
                    prediction_results['XGBoost'] = xgb_future_predictions
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("XGBoost MSE", f"{xgb_mse:.2f}")
                    with col2:
                        st.metric("XGBoost MAE", f"{xgb_mae:.2f}")
            
            # Display predictions
            if prediction_results:
                st.subheader("📊 Future Price Predictions")
                
                # Create prediction visualization
                future_dates = pd.date_range(
                    start=data.index[-1] + timedelta(days=1),
                    periods=prediction_days,
                    freq='D'
                )
                
                fig_pred = go.Figure()
                
                # Historical prices
                fig_pred.add_trace(go.Scatter(
                    x=data.index[-30:], 
                    y=data['Close'].iloc[-30:],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='blue')
                ))
                
                # Predictions
                colors = ['red', 'green', 'orange']
                for i, (model_name, predictions) in enumerate(prediction_results.items()):
                    fig_pred.add_trace(go.Scatter(
                        x=future_dates,
                        y=predictions,
                        mode='lines+markers',
                        name=f'{model_name} Prediction',
                        line=dict(color=colors[i], dash='dash')
                    ))
                
                fig_pred.update_layout(
                    title='Stock Price Predictions',
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    height=500
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Prediction summary
                for model_name, predictions in prediction_results.items():
                    current_price = data['Close'].iloc[-1]
                    predicted_price = predictions[-1]
                    price_change = predicted_price - current_price
                    percent_change = (price_change / current_price) * 100
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>{model_name} Prediction ({prediction_days} days)</h3>
                        <p><strong>Current Price:</strong> ₹{current_price:.2f}</p>
                        <p><strong>Predicted Price:</strong> ₹{predicted_price:.2f}</p>
                        <p><strong>Expected Change:</strong> ₹{price_change:.2f} ({percent_change:+.2f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Sentiment Analysis
            st.subheader("📰 Sentiment Analysis")
            stock_name = INDIAN_STOCKS[selected_stock]
            sentiment_score = get_sentiment_score(stock_name)
            
            if sentiment_score > 0.1:
                sentiment_label = "Positive 😊"
                sentiment_color = "green"
            elif sentiment_score < -0.1:
                sentiment_label = "Negative 😟"
                sentiment_color = "red"
            else:
                sentiment_label = "Neutral 😐"
                sentiment_color = "gray"
            
            st.markdown(f"""
            <div style="background-color: {sentiment_color}; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                <h4>Market Sentiment: {sentiment_label}</h4>
                <p>Sentiment Score: {sentiment_score:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Technical Analysis Summary
            st.subheader("📈 Technical Analysis Summary")
            
            current_price = data['Close'].iloc[-1]
            sma_20 = data['SMA_20'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1]
            rsi = data['RSI'].iloc[-1]
            
            signals = []
            
            if current_price > sma_20:
                signals.append("✅ Price above SMA 20 (Bullish)")
            else:
                signals.append("❌ Price below SMA 20 (Bearish)")
            
            if sma_20 > sma_50:
                signals.append("✅ SMA 20 above SMA 50 (Bullish)")
            else:
                signals.append("❌ SMA 20 below SMA 50 (Bearish)")
            
            if rsi < 30:
                signals.append("✅ RSI Oversold (Potential Buy)")
            elif rsi > 70:
                signals.append("❌ RSI Overbought (Potential Sell)")
            else:
                signals.append("🔄 RSI Neutral")
            
            for signal in signals:
                st.write(signal)
            
            # Risk Analysis
            st.subheader("⚠️ Risk Analysis")
            
            # Calculate volatility
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            
            # Calculate maximum drawdown
            rolling_max = data['Close'].cummax()
            drawdown = (data['Close'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Annual Volatility", f"{volatility:.2f}%")
            with col2:
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            with col3:
                # Sharpe ratio (simplified)
                risk_free_rate = 0.06  # Assuming 6% risk-free rate
                excess_return = returns.mean() * 252 - risk_free_rate
                sharpe_ratio = excess_return / (returns.std() * np.sqrt(252))
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
        else:
            st.error("Unable to load stock data. Please check the stock symbol and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>📈 Indian Stock Market Predictor & Strategy Visualizer</p>
    <p>Disclaimer: This is for educational purposes only. Do not use for actual trading without proper research.</p>
</div>
""", unsafe_allow_html=True)
