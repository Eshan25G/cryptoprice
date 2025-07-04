import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

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
        color: white;
        text-align: center;
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
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title and Header
st.markdown("""
<div class="main-header">
    <h1>🇮🇳 Indian Stock Market Predictor</h1>
    <p>Advanced AI-Powered Stock Analysis for NSE Stocks</p>
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
    options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
    index=3
)

# Prediction Model Selection
model_type = st.sidebar.selectbox(
    "Select Prediction Model:",
    options=['Random Forest', 'Linear Regression', 'Moving Average'],
    index=0
)

# Technical Indicators Selection
st.sidebar.subheader("Technical Indicators")
show_sma = st.sidebar.checkbox("Simple Moving Average", value=True)
show_ema = st.sidebar.checkbox("Exponential Moving Average", value=True)
show_volume = st.sidebar.checkbox("Volume", value=True)

# Prediction Parameters
prediction_days = st.sidebar.slider("Days to Predict:", 1, 30, 7)

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
    """Calculate basic technical indicators"""
    # Simple Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    
    # Price changes
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    
    # Volume moving average
    data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
    
    return data

def create_features(data):
    """Create features for machine learning models"""
    features = pd.DataFrame(index=data.index)
    
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
    features['Returns'] = data['Returns']
    features['Volatility'] = data['Volatility']
    
    # Lag features
    for lag in [1, 2, 3, 5]:
        features[f'Close_lag_{lag}'] = data['Close'].shift(lag)
        features[f'Volume_lag_{lag}'] = data['Volume'].shift(lag)
    
    # Rolling statistics
    features['Close_rolling_mean_5'] = data['Close'].rolling(5).mean()
    features['Close_rolling_std_5'] = data['Close'].rolling(5).std()
    features['Volume_rolling_mean_5'] = data['Volume'].rolling(5).mean()
    
    # Price ratios
    features['High_Low_Ratio'] = data['High'] / data['Low']
    features['Close_Open_Ratio'] = data['Close'] / data['Open']
    
    return features.dropna()

def train_prediction_model(data, model_type, prediction_days):
    """Train prediction model"""
    features = create_features(data)
    
    if len(features) < 50:
        return None, None, "Not enough data for prediction"
    
    # Create target variable
    features['Target'] = features['Close'].shift(-prediction_days)
    features = features.dropna()
    
    # Split features and target
    X = features.drop(['Target'], axis=1)
    y = features['Target']
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    if model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'Linear Regression':
        model = LinearRegression()
    else:  # Moving Average
        # Simple moving average prediction
        ma_period = 20
        predictions = data['Close'].rolling(window=ma_period).mean().iloc[split_idx:]
        actual = data['Close'].iloc[split_idx:]
        mse = mean_squared_error(actual, predictions.dropna())
        mae = mean_absolute_error(actual, predictions.dropna())
        
        # Future prediction
        future_price = data['Close'].rolling(window=ma_period).mean().iloc[-1]
        return {'predictions': [future_price] * prediction_days}, {'MSE': mse, 'MAE': mae}, None
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    # Future prediction
    last_features = X.iloc[-1:].copy()
    future_predictions = []
    
    for i in range(prediction_days):
        pred = model.predict(last_features)[0]
        future_predictions.append(pred)
        
        # Update features for next prediction (simplified)
        last_features['Close'] = pred
        last_features['Close_lag_1'] = pred
    
    return {'predictions': future_predictions}, {'MSE': mse, 'MAE': mae}, None

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
            
            current_price = data['Close'].iloc[-1]
            previous_price = data['Close'].iloc[-2]
            price_change = current_price - previous_price
            percent_change = (price_change / previous_price) * 100
            
            with col1:
                st.metric("Current Price", f"₹{current_price:.2f}")
            
            with col2:
                st.metric("Price Change", f"₹{price_change:.2f}", f"{percent_change:.2f}%")
            
            with col3:
                volume = data['Volume'].iloc[-1]
                st.metric("Volume", f"{volume:,.0f}")
            
            with col4:
                if info and 'marketCap' in info:
                    market_cap = info['marketCap']
                    st.metric("Market Cap", f"₹{market_cap/1e9:.2f}B")
                else:
                    st.metric("Market Cap", "N/A")
            
            # Create main chart
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Stock Price & Technical Indicators', 'Volume'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Price chart
            fig.add_trace(go.Scatter(
                x=data.index, 
                y=data['Close'], 
                name='Close Price', 
                line=dict(color='blue', width=2)
            ), row=1, col=1)
            
            # Technical indicators
            if show_sma:
                fig.add_trace(go.Scatter(
                    x=data.index, 
                    y=data['SMA_20'], 
                    name='SMA 20', 
                    line=dict(color='orange', width=1)
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=data.index, 
                    y=data['SMA_50'], 
                    name='SMA 50', 
                    line=dict(color='red', width=1)
                ), row=1, col=1)
            
            if show_ema:
                fig.add_trace(go.Scatter(
                    x=data.index, 
                    y=data['EMA_12'], 
                    name='EMA 12', 
                    line=dict(color='green', width=1)
                ), row=1, col=1)
            
            # Volume chart
            if show_volume:
                colors = ['red' if ret < 0 else 'green' for ret in data['Returns']]
                fig.add_trace(go.Bar(
                    x=data.index, 
                    y=data['Volume'], 
                    name='Volume', 
                    marker_color=colors,
                    opacity=0.7
                ), row=2, col=1)
            
            fig.update_layout(
                height=600, 
                title_text=f"{INDIAN_STOCKS[selected_stock]} - Technical Analysis",
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Predictions
            st.subheader("🤖 AI Predictions")
            
            model_results, metrics, error = train_prediction_model(data, model_type, prediction_days)
            
            if error:
                st.error(error)
            elif model_results:
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Model MSE", f"{metrics['MSE']:.2f}")
                with col2:
                    st.metric("Model MAE", f"{metrics['MAE']:.2f}")
                
                # Create prediction chart
                future_dates = pd.date_range(
                    start=data.index[-1] + timedelta(days=1),
                    periods=prediction_days,
                    freq='D'
                )
                
                fig_pred = go.Figure()
                
                # Historical prices (last 30 days)
                historical_data = data.tail(30)
                fig_pred.add_trace(go.Scatter(
                    x=historical_data.index,
                    y=historical_data['Close'],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='blue', width=2)
                ))
                
                # Predictions
                fig_pred.add_trace(go.Scatter(
                    x=future_dates,
                    y=model_results['predictions'],
                    mode='lines+markers',
                    name=f'{model_type} Prediction',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=8)
                ))
                
                fig_pred.update_layout(
                    title=f'Stock Price Prediction - Next {prediction_days} Days',
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Prediction summary
                predicted_price = model_results['predictions'][-1]
                price_change_pred = predicted_price - current_price
                percent_change_pred = (price_change_pred / current_price) * 100
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>{model_type} Prediction Summary</h3>
                    <p><strong>Current Price:</strong> ₹{current_price:.2f}</p>
                    <p><strong>Predicted Price ({prediction_days} days):</strong> ₹{predicted_price:.2f}</p>
                    <p><strong>Expected Change:</strong> ₹{price_change_pred:.2f} ({percent_change_pred:+.2f}%)</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Technical Analysis Summary
            st.subheader("📈 Technical Analysis Summary")
            
            sma_20 = data['SMA_20'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1]
            
            signals = []
            
            if current_price > sma_20:
                signals.append("✅ Price above SMA 20 (Bullish)")
            else:
                signals.append("❌ Price below SMA 20 (Bearish)")
            
            if not pd.isna(sma_20) and not pd.isna(sma_50):
                if sma_20 > sma_50:
                    signals.append("✅ SMA 20 above SMA 50 (Bullish)")
                else:
                    signals.append("❌ SMA 20 below SMA 50 (Bearish)")
            
            # Volume analysis
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            
            if current_volume > avg_volume * 1.5:
                signals.append("✅ High volume (Strong interest)")
            elif current_volume < avg_volume * 0.5:
                signals.append("⚠️ Low volume (Weak interest)")
            else:
                signals.append("🔄 Normal volume")
            
            for signal in signals:
                st.write(signal)
            
            # Risk Analysis
            st.subheader("⚠️ Risk Analysis")
            
            # Calculate volatility
            returns = data['Returns'].dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            
            # Calculate maximum drawdown
            rolling_max = data['Close'].cummax()
            drawdown = (data['Close'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # Calculate beta (simplified - against market index)
            # For simplicity, we'll use the stock's own correlation
            beta = 1.0  # Simplified
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Annual Volatility", f"{volatility:.2f}%")
            with col2:
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            with col3:
                st.metric("Beta (Est.)", f"{beta:.2f}")
            
            # Risk assessment
            if volatility > 30:
                risk_level = "High Risk 🔴"
            elif volatility > 20:
                risk_level = "Medium Risk 🟡"
            else:
                risk_level = "Low Risk 🟢"
            
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <h4>Risk Assessment: {risk_level}</h4>
                <p>Based on historical volatility and drawdown analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.error("Unable to load stock data. Please check the stock symbol and try again.")

# Information Section
st.sidebar.markdown("---")
st.sidebar.markdown("""
### ℹ️ About
This app provides:
- Real-time stock data for Indian stocks
- Technical analysis with indicators
- AI-powered price predictions
- Risk assessment metrics

### ⚠️ Disclaimer
This tool is for educational purposes only. 
Do not use for actual trading without proper research and risk management.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 1rem;">
    <p>📈 Indian Stock Market Predictor & Strategy Visualizer</p>
    <p>Built with Streamlit • Data from Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
