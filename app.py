import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import warnings
import json
import time
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.error("yfinance not found. Please install it with: pip install yfinance")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("scikit-learn not found. ML predictions will be disabled. Install with: pip install scikit-learn")

# Configure Streamlit page
st.set_page_config(
    page_title="Crypto Price Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚀 Crypto Price Predictor</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")

# Cryptocurrency selection
crypto_options = {
    'Bitcoin': 'BTC-USD',
    'Ethereum': 'ETH-USD',
    'Binance Coin': 'BNB-USD',
    'Cardano': 'ADA-USD',
    'Solana': 'SOL-USD',
    'Polkadot': 'DOT-USD',
    'Dogecoin': 'DOGE-USD',
    'Avalanche': 'AVAX-USD',
    'Polygon': 'MATIC-USD',
    'Chainlink': 'LINK-USD'
}

selected_crypto = st.sidebar.selectbox(
    "Select Cryptocurrency",
    list(crypto_options.keys()),
    index=0
)

# Time period selection
period_options = {
    '1 Month': '1mo',
    '3 Months': '3mo',
    '6 Months': '6mo',
    '1 Year': '1y',
    '2 Years': '2y',
    '5 Years': '5y'
}

selected_period = st.sidebar.selectbox(
    "Select Time Period",
    list(period_options.keys()),
    index=3
)

# Prediction days
prediction_days = st.sidebar.slider(
    "Days to Predict",
    min_value=1,
    max_value=30,
    value=7,
    help="Number of days to predict into the future"
)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_crypto_data_yfinance(symbol, period):
    """Fetch cryptocurrency data from Yahoo Finance"""
    if not YFINANCE_AVAILABLE:
        return None
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_crypto_data_api(symbol, days=365):
    """Fetch cryptocurrency data from CoinGecko API as fallback"""
    try:
        # Convert symbol to CoinGecko format
        symbol_map = {
            'BTC-USD': 'bitcoin',
            'ETH-USD': 'ethereum', 
            'BNB-USD': 'binancecoin',
            'ADA-USD': 'cardano',
            'SOL-USD': 'solana',
            'DOT-USD': 'polkadot',
            'DOGE-USD': 'dogecoin',
            'AVAX-USD': 'avalanche-2',
            'MATIC-USD': 'matic-network',
            'LINK-USD': 'chainlink'
        }
        
        coin_id = symbol_map.get(symbol, 'bitcoin')
        
        # Fetch data from CoinGecko
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': min(days, 365),  # CoinGecko free tier limit
            'interval': 'daily' if days > 90 else 'hourly'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Convert to DataFrame
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            if prices:
                df = pd.DataFrame(prices, columns=['timestamp', 'close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Add volume data if available
                if volumes:
                    vol_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                    vol_df['timestamp'] = pd.to_datetime(vol_df['timestamp'], unit='ms')
                    vol_df.set_index('timestamp', inplace=True)
                    df = df.join(vol_df)
                else:
                    df['volume'] = 0
                
                # Create OHLC data (simplified - using close price)
                df['open'] = df['close'].shift(1).fillna(df['close'])
                df['high'] = df['close'] * 1.02  # Approximate
                df['low'] = df['close'] * 0.98   # Approximate
                
                # Rename columns to match yfinance format
                df.columns = ['Close', 'Volume', 'Open', 'High', 'Low']
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                
                return df
                
        return None
        
    except Exception as e:
        st.error(f"Error fetching data from CoinGecko: {e}")
        return None

def fetch_crypto_data(symbol, period):
    """Main function to fetch crypto data with fallbacks"""
    # Convert period to days for API
    period_to_days = {
        '1mo': 30,
        '3mo': 90, 
        '6mo': 180,
        '1y': 365,
        '2y': 730,
        '5y': 1825
    }
    
    days = period_to_days.get(period, 365)
    
    # Try Yahoo Finance first
    if YFINANCE_AVAILABLE:
        data = fetch_crypto_data_yfinance(symbol, period)
        if data is not None and not data.empty:
            return data
    
    # Fallback to CoinGecko API
    st.info("Using CoinGecko API as data source...")
    return fetch_crypto_data_api(symbol, days)

@st.cache_data(ttl=300)
def get_crypto_info(symbol):
    """Get cryptocurrency information"""
    if YFINANCE_AVAILABLE:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        except:
            pass
    
    # Fallback to basic info
    return {
        'marketCap': None,
        'symbol': symbol.replace('-USD', '').upper()
    }

def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    # Moving averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    
    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    
    # Volume indicators
    data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
    
    return data

def prepare_features(data):
    """Prepare features for machine learning model"""
    # Price-based features
    data['Price_Change'] = data['Close'].pct_change()
    data['High_Low_Ratio'] = data['High'] / data['Low']
    data['Price_Volume'] = data['Close'] * data['Volume']
    
    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
        data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
    
    # Rolling statistics
    for window in [5, 10, 20]:
        data[f'Close_Rolling_Mean_{window}'] = data['Close'].rolling(window=window).mean()
        data[f'Close_Rolling_Std_{window}'] = data['Close'].rolling(window=window).std()
        data[f'Volume_Rolling_Mean_{window}'] = data['Volume'].rolling(window=window).mean()
    
    return data

def create_ml_model(data, target_days=7):
    """Create and train machine learning model"""
    if not SKLEARN_AVAILABLE:
        return None, None, None, "scikit-learn not available"
        
    # Prepare features
    feature_columns = [
        'Open', 'High', 'Low', 'Volume',
        'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
        'MACD', 'MACD_Signal', 'RSI',
        'Price_Change', 'High_Low_Ratio', 'Price_Volume'
    ]
    
    # Add lag features
    for lag in [1, 2, 3, 5, 10]:
        feature_columns.extend([f'Close_Lag_{lag}', f'Volume_Lag_{lag}'])
    
    # Add rolling features
    for window in [5, 10, 20]:
        feature_columns.extend([
            f'Close_Rolling_Mean_{window}',
            f'Close_Rolling_Std_{window}',
            f'Volume_Rolling_Mean_{window}'
        ])
    
    # Create target (future price)
    data['Target'] = data['Close'].shift(-target_days)
    
    # Select features that exist in the data
    available_features = [col for col in feature_columns if col in data.columns]
    
    # Remove rows with NaN values
    model_data = data[available_features + ['Target']].dropna()
    
    if len(model_data) < 50:
        return None, None, None, "Insufficient data for modeling"
    
    X = model_data[available_features]
    y = model_data['Target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, scaler, available_features, {'MAE': mae, 'RMSE': rmse}

def predict_future_prices(model, scaler, data, features, days=7):
    """Predict future prices"""
    try:
        # Get the last row of features
        last_row = data[features].iloc[-1:].values
        last_row_scaled = scaler.transform(last_row)
        
        # Make prediction
        prediction = model.predict(last_row_scaled)[0]
        return prediction
    except Exception as e:
        return None

# Main app logic
if st.sidebar.button("🔄 Update Data", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Fetch data
symbol = crypto_options[selected_crypto]
period = period_options[selected_period]

with st.spinner(f"Fetching {selected_crypto} data..."):
    data = fetch_crypto_data(symbol, period)
    crypto_info = get_crypto_info(symbol)

if data is not None and not data.empty:
    # Calculate technical indicators
    data = calculate_technical_indicators(data)
    data = prepare_features(data)
    
    # Current price and metrics
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    # Display current metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Current Price</h3>
            <h2>${current_price:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        change_color = "green" if price_change >= 0 else "red"
        st.markdown(f"""
        <div class="metric-card">
            <h3>24h Change</h3>
            <h2 style="color: {change_color};">{price_change_pct:+.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        volume = data['Volume'].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <h3>Volume</h3>
            <h2>{volume:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        market_cap = crypto_info.get('marketCap', 'N/A')
        if market_cap != 'N/A':
            market_cap = f"${market_cap:,.0f}"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Market Cap</h3>
            <h2>{market_cap}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Machine Learning Prediction
    st.subheader("🤖 AI Price Prediction")
    
    if SKLEARN_AVAILABLE:
        with st.spinner("Training prediction model..."):
            model, scaler, features, metrics = create_ml_model(data, prediction_days)
        
        if model is not None:
            # Make prediction
            predicted_price = predict_future_prices(model, scaler, data, features, prediction_days)
            
            if predicted_price is not None:
                prediction_change = predicted_price - current_price
                prediction_change_pct = (prediction_change / current_price) * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>Predicted Price ({prediction_days} days)</h3>
                        <h1>${predicted_price:,.2f}</h1>
                        <p>Expected Change: {prediction_change_pct:+.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Model Performance:**")
                    st.metric("Mean Absolute Error", f"${metrics['MAE']:.2f}")
                    st.metric("Root Mean Square Error", f"${metrics['RMSE']:.2f}")
                    
                    # Model confidence
                    confidence = max(0, min(100, 100 - (metrics['MAE'] / current_price) * 100))
                    st.metric("Model Confidence", f"{confidence:.1f}%")
            else:
                st.warning("Unable to generate price prediction with current data.")
        else:
            st.warning("Insufficient data for machine learning model.")
    else:
        st.warning("🚫 Machine Learning predictions disabled. Install scikit-learn to enable.")
        
        # Simple trend-based prediction as fallback
        st.info("📈 Using simple trend analysis instead...")
        
        # Calculate simple moving average trend
        recent_prices = data['Close'].tail(7)
        trend = recent_prices.iloc[-1] - recent_prices.iloc[0]
        simple_prediction = current_price + (trend * prediction_days / 7)
        
        prediction_change = simple_prediction - current_price
        prediction_change_pct = (prediction_change / current_price) * 100
        
        st.markdown(f"""
        <div class="prediction-box">
            <h3>Trend-Based Estimate ({prediction_days} days)</h3>
            <h1>${simple_prediction:,.2f}</h1>
            <p>Expected Change: {prediction_change_pct:+.2f}%</p>
            <small>⚠️ Based on simple trend analysis</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Create interactive chart
    st.subheader("📈 Price Chart with Technical Indicators")
    
    # Main price chart
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Moving Averages', 'MACD', 'RSI'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # Moving averages
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper', 
                  line=dict(color='gray', dash='dash'), opacity=0.5),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower', 
                  line=dict(color='gray', dash='dash'), opacity=0.5),
        row=1, col=1
    )
    
    # MACD
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='red')),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
        row=3, col=1
    )
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
    
    fig.update_layout(
        title=f"{selected_crypto} Technical Analysis",
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical Analysis Summary
    st.subheader("📊 Technical Analysis Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Indicators:**")
        
        # RSI analysis
        current_rsi = data['RSI'].iloc[-1]
        if current_rsi > 70:
            rsi_signal = "🔴 Overbought"
        elif current_rsi < 30:
            rsi_signal = "🟢 Oversold"
        else:
            rsi_signal = "🟡 Neutral"
        
        st.write(f"RSI (14): {current_rsi:.1f} - {rsi_signal}")
        
        # MACD analysis
        current_macd = data['MACD'].iloc[-1]
        current_signal = data['MACD_Signal'].iloc[-1]
        macd_signal = "🟢 Bullish" if current_macd > current_signal else "🔴 Bearish"
        st.write(f"MACD: {macd_signal}")
        
        # Moving average analysis
        if current_price > data['SMA_20'].iloc[-1]:
            ma_signal = "🟢 Above SMA 20"
        else:
            ma_signal = "🔴 Below SMA 20"
        st.write(f"Price vs SMA 20: {ma_signal}")
    
    with col2:
        st.markdown("**Price Levels:**")
        
        # Support and resistance
        recent_high = data['High'].tail(20).max()
        recent_low = data['Low'].tail(20).min()
        
        st.write(f"20-day High: ${recent_high:.2f}")
        st.write(f"20-day Low: ${recent_low:.2f}")
        st.write(f"Price Range: {((recent_high - recent_low) / recent_low * 100):.1f}%")
        
        # Volatility
        volatility = data['Close'].pct_change().std() * np.sqrt(365) * 100
        st.write(f"Annualized Volatility: {volatility:.1f}%")
    
    # Risk Analysis
    st.subheader("⚠️ Risk Analysis")
    
    # Calculate risk metrics
    returns = data['Close'].pct_change().dropna()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Sharpe Ratio (simplified, assuming 0% risk-free rate)
        sharpe = returns.mean() / returns.std() * np.sqrt(365) if returns.std() != 0 else 0
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    with col2:
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
    
    with col3:
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5) * 100
        st.metric("VaR (95%)", f"{var_95:.1f}%")
    
    # Data table
    st.subheader("📋 Recent Data")
    
    # Display recent data
    recent_data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']].tail(10)
    recent_data = recent_data.round(2)
    st.dataframe(recent_data, use_container_width=True)
    
    # Download data option
    csv = data.to_csv()
    st.download_button(
        "📥 Download Historical Data",
        csv,
        f"{selected_crypto}_{selected_period}_data.csv",
        "text/csv",
        key='download-csv'
    )

else:
    st.error("Unable to fetch cryptocurrency data. Please try again later.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>⚠️ <strong>Disclaimer:</strong> This is for educational purposes only. Not financial advice. 
    Cryptocurrency investments are highly volatile and risky.</p>
    <p>Data provided by Yahoo Finance | Predictions are estimates and should not be used as sole investment guidance</p>
</div>
""", unsafe_allow_html=True)
