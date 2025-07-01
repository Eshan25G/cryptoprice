import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Crypto Trading Analyzer",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(45deg, #1f1f2e, #16213e);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B35;
        margin: 0.5rem 0;
    }
    .signal-buy {
        background-color: #00ff88;
        color: black;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        font-size: 18px;
        margin: 10px 0;
    }
    .signal-sell {
        background-color: #ff4444;
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        font-size: 18px;
        margin: 10px 0;
    }
    .signal-hold {
        background-color: #ffaa00;
        color: black;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        font-size: 18px;
        margin: 10px 0;
    }
    .trading-box {
        border: 2px solid #FF6B35;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

class CryptoAnalyzer:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        
    def get_crypto_data(self, symbol, days=30):
        """Fetch cryptocurrency data from CoinGecko API"""
        try:
            # Get current price and market data
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': symbol,
                'vs_currencies': 'inr',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true',
                'include_market_cap': 'true'
            }
            current_data = requests.get(url, params=params, timeout=10).json()
            
            # Get historical data
            hist_url = f"{self.base_url}/coins/{symbol}/market_chart"
            hist_params = {
                'vs_currency': 'inr',
                'days': days,
                'interval': 'hourly' if days <= 7 else 'daily'
            }
            hist_data = requests.get(hist_url, params=hist_params, timeout=10).json()
            
            # Convert to DataFrame
            df = pd.DataFrame(hist_data['prices'], columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add volume data
            volume_df = pd.DataFrame(hist_data['total_volumes'], columns=['timestamp', 'volume'])
            volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
            volume_df.set_index('timestamp', inplace=True)
            df['volume'] = volume_df['volume']
            
            # Add market cap data
            if 'market_caps' in hist_data:
                mcap_df = pd.DataFrame(hist_data['market_caps'], columns=['timestamp', 'market_cap'])
                mcap_df['timestamp'] = pd.to_datetime(mcap_df['timestamp'], unit='ms')
                mcap_df.set_index('timestamp', inplace=True)
                df['market_cap'] = mcap_df['market_cap']
            
            return df, current_data[symbol]
            
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None, None
    
    def calculate_sma(self, data, window):
        """Calculate Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    def calculate_ema(self, data, window):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    def calculate_rsi(self, data, window=14):
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, data, window=20):
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(data, window)
        std = data.rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, sma, lower_band
    
    def calculate_stochastic(self, high, low, close, k_window=14, d_window=3):
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    def calculate_technical_indicators(self, df):
        """Calculate various technical indicators"""
        # Moving Averages
        df['SMA_20'] = self.calculate_sma(df['price'], 20)
        df['SMA_50'] = self.calculate_sma(df['price'], 50)
        df['EMA_12'] = self.calculate_ema(df['price'], 12)
        df['EMA_26'] = self.calculate_ema(df['price'], 26)
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['price'])
        
        # MACD
        df['MACD'], df['MACD_signal'], df['MACD_histogram'] = self.calculate_macd(df['price'])
        
        # Bollinger Bands
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = self.calculate_bollinger_bands(df['price'])
        
        # For stochastic, we'll use price as proxy for high/low
        df['Stoch_K'], df['Stoch_D'] = self.calculate_stochastic(df['price'], df['price'], df['price'])
        
        # Support and Resistance levels
        df['Support'] = df['price'].rolling(window=20).min()
        df['Resistance'] = df['price'].rolling(window=20).max()
        
        # Volume indicators
        df['Volume_SMA'] = self.calculate_sma(df['volume'], 20)
        
        return df
    
    def detect_chart_patterns(self, df):
        """Detect common chart patterns"""
        patterns = {}
        
        # Trend detection
        short_trend = df['SMA_20'].iloc[-1] > df['SMA_20'].iloc[-5]
        long_trend = df['SMA_50'].iloc[-1] > df['SMA_50'].iloc[-10]
        
        if short_trend and long_trend:
            patterns['trend'] = 'Strong Bullish'
        elif short_trend and not long_trend:
            patterns['trend'] = 'Short-term Bullish'
        elif not short_trend and long_trend:
            patterns['trend'] = 'Weakening Bull'
        else:
            patterns['trend'] = 'Bearish'
        
        # Breakout detection
        current_price = df['price'].iloc[-1]
        resistance_level = df['Resistance'].iloc[-5:].mean()
        support_level = df['Support'].iloc[-5:].mean()
        
        if current_price > resistance_level * 1.02:
            patterns['breakout'] = 'Resistance Breakout - Bullish'
        elif current_price < support_level * 0.98:
            patterns['breakout'] = 'Support Breakdown - Bearish'
        else:
            patterns['breakout'] = 'Trading in Range'
        
        # Volume pattern
        avg_volume = df['volume'].rolling(window=10).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        
        if current_volume > avg_volume * 1.5:
            patterns['volume'] = 'High Volume - Strong Interest'
        elif current_volume < avg_volume * 0.5:
            patterns['volume'] = 'Low Volume - Weak Interest'
        else:
            patterns['volume'] = 'Normal Volume'
        
        return patterns
    
    def generate_trading_signals(self, df):
        """Generate comprehensive trading signals"""
        if len(df) < 50:  # Need enough data
            return {'action': 'HOLD', 'confidence': 0, 'reasons': ['Insufficient data']}
            
        current_data = df.iloc[-1]
        signals = {'action': 'HOLD', 'confidence': 0, 'reasons': []}
        
        buy_signals = 0
        sell_signals = 0
        signal_strength = 0
        
        # RSI Signals
        if not pd.isna(current_data['RSI']):
            if current_data['RSI'] < 30:
                buy_signals += 2
                signals['reasons'].append(f"RSI Oversold ({current_data['RSI']:.1f})")
            elif current_data['RSI'] > 70:
                sell_signals += 2
                signals['reasons'].append(f"RSI Overbought ({current_data['RSI']:.1f})")
        
        # MACD Signals
        if not pd.isna(current_data['MACD']) and not pd.isna(current_data['MACD_signal']):
            if len(df) > 1:
                prev_macd = df['MACD'].iloc[-2]
                prev_signal = df['MACD_signal'].iloc[-2]
                
                if current_data['MACD'] > current_data['MACD_signal'] and prev_macd <= prev_signal:
                    buy_signals += 3
                    signals['reasons'].append("MACD Bullish Crossover")
                elif current_data['MACD'] < current_data['MACD_signal'] and prev_macd >= prev_signal:
                    sell_signals += 3
                    signals['reasons'].append("MACD Bearish Crossover")
        
        # Moving Average Signals
        if not pd.isna(current_data['SMA_20']) and not pd.isna(current_data['SMA_50']):
            if current_data['price'] > current_data['SMA_20'] > current_data['SMA_50']:
                buy_signals += 2
                signals['reasons'].append("Price above SMAs (Bullish Alignment)")
            elif current_data['price'] < current_data['SMA_20'] < current_data['SMA_50']:
                sell_signals += 2
                signals['reasons'].append("Price below SMAs (Bearish Alignment)")
        
        # Bollinger Bands
        if not pd.isna(current_data['BB_lower']) and not pd.isna(current_data['BB_upper']):
            if current_data['price'] <= current_data['BB_lower']:
                buy_signals += 1
                signals['reasons'].append("Price at Lower Bollinger Band")
            elif current_data['price'] >= current_data['BB_upper']:
                sell_signals += 1
                signals['reasons'].append("Price at Upper Bollinger Band")
        
        # Stochastic
        if not pd.isna(current_data['Stoch_K']) and not pd.isna(current_data['Stoch_D']):
            if current_data['Stoch_K'] < 20 and current_data['Stoch_D'] < 20:
                buy_signals += 1
                signals['reasons'].append("Stochastic Oversold")
            elif current_data['Stoch_K'] > 80 and current_data['Stoch_D'] > 80:
                sell_signals += 1
                signals['reasons'].append("Stochastic Overbought")
        
        # Volume confirmation
        if not pd.isna(current_data['Volume_SMA']):
            if current_data['volume'] > current_data['Volume_SMA'] * 1.5:
                signal_strength += 1
                signals['reasons'].append("High Volume Confirmation")
        
        # Determine final signal
        net_signal = buy_signals - sell_signals
        total_signals = buy_signals + sell_signals
        
        if net_signal >= 3:
            signals['action'] = 'BUY'
            signals['confidence'] = min(90, (net_signal / max(total_signals, 1)) * 100 + signal_strength * 10)
        elif net_signal <= -3:
            signals['action'] = 'SELL'
            signals['confidence'] = min(90, abs(net_signal / max(total_signals, 1)) * 100 + signal_strength * 10)
        else:
            signals['action'] = 'HOLD'
            signals['confidence'] = 50 + abs(net_signal) * 5
        
        return signals
    
    def calculate_position_sizing(self, current_price, account_balance, risk_per_trade=0.02):
        """Calculate position size based on risk management"""
        risk_amount = account_balance * risk_per_trade
        
        # Calculate stop loss (2% below current price for long, 2% above for short)
        stop_loss_long = current_price * 0.98
        stop_loss_short = current_price * 1.02
        
        # Calculate position size
        position_size_long = risk_amount / (current_price - stop_loss_long)
        position_size_short = risk_amount / (stop_loss_short - current_price)
        
        return {
            'long_position_size': position_size_long,
            'short_position_size': position_size_short,
            'stop_loss_long': stop_loss_long,
            'stop_loss_short': stop_loss_short,
            'take_profit_long': current_price * 1.06,  # 6% profit target
            'take_profit_short': current_price * 0.94,   # 6% profit target
            'risk_amount': risk_amount
        }

def create_simple_chart(df, title):
    """Create a simple chart using Streamlit's built-in charting"""
    chart_data = pd.DataFrame({
        'Price': df['price'],
        'SMA 20': df['SMA_20'],
        'SMA 50': df['SMA_50']
    })
    return chart_data

def main():
    st.markdown('<h1 class="main-header">🚀 Crypto Trading Analyzer India</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = CryptoAnalyzer()
    
    # Sidebar
    st.sidebar.title("⚙️ Trading Configuration")
    
    # Crypto selection
    crypto_options = {
        'Bitcoin': 'bitcoin',
        'Ethereum': 'ethereum',
        'Polygon': 'matic-network',
        'Solana': 'solana',
        'Cardano': 'cardano',
        'Dogecoin': 'dogecoin',
        'Shiba Inu': 'shiba-inu',
        'BNB': 'binancecoin',
        'XRP': 'ripple',
        'Avalanche': 'avalanche-2'
    }
    
    selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", list(crypto_options.keys()))
    crypto_id = crypto_options[selected_crypto]
    
    # Trading parameters
    st.sidebar.header("💰 Trading Parameters")
    account_balance = st.sidebar.number_input("Account Balance (₹)", min_value=1000, max_value=10000000, value=50000, step=1000)
    risk_per_trade = st.sidebar.slider("Risk per Trade (%)", min_value=1, max_value=5, value=2) / 100
    target_daily_profit = st.sidebar.number_input("Daily Profit Target (₹)", min_value=100, max_value=50000, value=5000, step=100)
    
    # Time frame
    time_frame = st.sidebar.selectbox("Analysis Time Frame", ['7 days', '30 days', '90 days'])
    days = int(time_frame.split()[0])
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh (60s)", value=False)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        st.rerun()
    
    # Main content
    try:
        # Fetch data
        with st.spinner(f"Fetching {selected_crypto} data from CoinGecko..."):
            df, current_data = analyzer.get_crypto_data(crypto_id, days)
        
        if df is not None and current_data is not None:
            # Calculate indicators
            df = analyzer.calculate_technical_indicators(df)
            
            # Current price display
            col1, col2, col3, col4 = st.columns(4)
            
            change_24h = current_data.get('inr_24h_change', 0)
            change_color = "normal" if change_24h >= 0 else "inverse"
            
            with col1:
                st.metric(
                    "💰 Current Price",
                    f"₹{current_data['inr']:,.2f}",
                    f"{change_24h:.2f}%",
                    delta_color=change_color
                )
            
            with col2:
                volume_24h = current_data.get('inr_24h_vol', 0)
                st.metric("📊 24h Volume", f"₹{volume_24h:,.0f}")
            
            with col3:
                market_cap = current_data.get('market_cap', {}).get('inr', 0)
                st.metric("🏦 Market Cap", f"₹{market_cap:,.0f}")
            
            with col4:
                # Calculate required trades per day
                risk_reward_ratio = 3  # Assuming 1:3 risk-reward
                profit_per_trade = account_balance * risk_per_trade * risk_reward_ratio
                required_trades = max(1, target_daily_profit / profit_per_trade) if profit_per_trade > 0 else float('inf')
                st.metric("🎯 Required Trades/Day", f"{required_trades:.1f}")
            
            # Trading signals
            signals = analyzer.generate_trading_signals(df)
            patterns = analyzer.detect_chart_patterns(df)
            
            # Signal display
            st.header("📊 Trading Signals & Analysis")
            
            signal_col1, signal_col2 = st.columns([1, 2])
            
            with signal_col1:
                if signals['action'] == 'BUY':
                    st.markdown(f'<div class="signal-buy">🟢 STRONG BUY SIGNAL<br>Confidence: {signals["confidence"]:.0f}%</div>', unsafe_allow_html=True)
                elif signals['action'] == 'SELL':
                    st.markdown(f'<div class="signal-sell">🔴 STRONG SELL SIGNAL<br>Confidence: {signals["confidence"]:.0f}%</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="signal-hold">🟡 HOLD POSITION<br>Confidence: {signals["confidence"]:.0f}%</div>', unsafe_allow_html=True)
            
            with signal_col2:
                st.write("**📋 Signal Analysis:**")
                if signals['reasons']:
                    for i, reason in enumerate(signals['reasons'], 1):
                        st.write(f"{i}. {reason}")
                else:
                    st.write("• Waiting for clear signals...")
            
            # Position sizing and risk management
            position_info = analyzer.calculate_position_sizing(current_data['inr'], account_balance, risk_per_trade)
            
            st.header("💼 Position Management & Risk Analysis")
            
            pos_col1, pos_col2, pos_col3 = st.columns(3)
            
            with pos_col1:
                st.markdown('<div class="trading-box">', unsafe_allow_html=True)
                if signals['action'] == 'BUY':
                    st.write("**🟢 LONG POSITION SETUP:**")
                    st.write(f"• **Entry Price:** ₹{current_data['inr']:,.2f}")
                    st.write(f"• **Position Size:** {position_info['long_position_size']:.6f} {selected_crypto}")
                    st.write(f"• **Investment:** ₹{position_info['long_position_size'] * current_data['inr']:,.2f}")
                    st.write(f"• **Stop Loss:** ₹{position_info['stop_loss_long']:,.2f} (-2%)")
                    st.write(f"• **Take Profit:** ₹{position_info['take_profit_long']:,.2f} (+6%)")
                else:
                    st.write("**💡 LONG POSITION (If Bullish):**")
                    st.write(f"• Entry: ₹{current_data['inr']:,.2f}")
                    st.write(f"• Size: {position_info['long_position_size']:.6f} {selected_crypto}")
                    st.write(f"• SL: ₹{position_info['stop_loss_long']:,.2f}")
                    st.write(f"• TP: ₹{position_info['take_profit_long']:,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with pos_col2:
                st.markdown('<div class="trading-box">', unsafe_allow_html=True)
                if signals['action'] == 'SELL':
                    st.write("**🔴 SHORT POSITION SETUP:**")
                    st.write(f"• **Entry Price:** ₹{current_data['inr']:,.2f}")
                    st.write(f"• **Position Size:** {position_info['short_position_size']:.6f} {selected_crypto}")
                    st.write(f"• **Investment:** ₹{position_info['short_position_size'] * current_data['inr']:,.2f}")
                    st.write(f"• **Stop Loss:** ₹{position_info['stop_loss_short']:,.2f} (+2%)")
                    st.write(f"• **Take Profit:** ₹{position_info['take_profit_short']:,.2f} (-6%)")
                else:
                    st.write("**💡 SHORT POSITION (If Bearish):**")
                    st.write(f"• Entry: ₹{current_data['inr']:,.2f}")
                    st.write(f"• Size: {position_info['short_position_size']:.6f} {selected_crypto}")
                    st.write(f"• SL: ₹{position_info['stop_loss_short']:,.2f}")
                    st.write(f"• TP: ₹{position_info['take_profit_short']:,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with pos_col3:
                st.markdown('<div class="trading-box">', unsafe_allow_html=True)
                st.write("**⚖️ RISK MANAGEMENT:**")
                st.write(f"• **Risk Amount:** ₹{position_info['risk_amount']:,.2f}")
                st.write(f"• **Risk Percentage:** {risk_per_trade*100:.1f}% of capital")
                st.write(f"• **Risk/Reward:** 1:3")
                potential_profit = position_info['risk_amount'] * 3
                st.write(f"• **Potential Profit:** ₹{potential_profit:,.2f}")
                st.write(f"• **Win Rate Needed:** 25% (1:3 RR)")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Technical Analysis Summary
            st.header("🔍 Technical Analysis Summary")
            
            tech_col1, tech_col2 = st.columns(2)
            
            with tech_col1:
                st.subheader("📈 Current Indicators")
                current_data_df = df.iloc[-1]
                
                # RSI Analysis
                rsi_val = current_data_df['RSI']
                if not pd.isna(rsi_val):
                    if rsi_val < 30:
                        rsi_status = "🟢 Oversold (Bullish)"
                    elif rsi_val > 70:
                        rsi_status = "🔴 Overbought (Bearish)"
                    else:
                        rsi_status = "🟡 Neutral"
                    st.write(f"**RSI (14):** {rsi_val:.1f} - {rsi_status}")
                
                # MACD Analysis
                macd_val = current_data_df['MACD']
                macd_signal_val = current_data_df['MACD_signal']
                if not pd.isna(macd_val) and not pd.isna(macd_signal_val):
                    macd_status = "🟢 Bullish" if macd_val > macd_signal_val else "🔴 Bearish"
                    st.write(f"**MACD:** {macd_status}")
                
                # Moving Averages
                price = current_data_df['price']
                sma20 = current_data_df['SMA_20']
                sma50 = current_data_df['SMA_50']
                
                if not pd.isna(sma20) and not pd.isna(sma50):
                    if price > sma20 > sma50:
                        ma_status = "🟢 Strong Bullish"
                    elif price > sma20 and sma20 < sma50:
                        ma_status = "🟡 Mixed Signals"
                    elif price < sma20 < sma50:
                        ma_status = "🔴 Strong Bearish"
                    else:
                        ma_status = "🟡 Consolidating"
                    st.write(f"**Moving Averages:** {ma_status}")
            
            with tech_col2:
                st.subheader("🎯 Pattern Analysis")
                st.write(f"**Trend:** {patterns['trend']}")
                st.write(f"**Breakout:** {patterns['breakout']}")
                st.write(f"**Volume:** {patterns['volume']}")
                
                # Support and Resistance
                current_support = df['Support'].iloc[-1]
                current_resistance = df['Resistance'].iloc[-1]
                if not pd.isna(current_support) and not pd.isna(current_resistance):
                    st.write(f"**Support Level:** ₹{current_support:,.2f}")
                    st.write(f"**Resistance Level:** ₹{current_resistance:,.2f}")
                    
                    # Distance from S&R
                    current_price = current_data['inr']
                    support_distance = ((current_price - current_support) / current_support) * 100
                    resistance_distance = ((current_resistance - current_price) / current_price) * 100
                    
                    st.write(f"**Distance from Support:** +{support_distance:.1f}%")
                    st.write(f"**Distance to Resistance:** +{resistance_distance:.1f}%")
            
            # Price Charts
            st.header("📊 Price Charts & Technical Analysis")
            
            # Create chart data
            chart_data = create_simple_chart(df, f"{selected_crypto} Price Analysis")
            
            # Display main price chart
            st.subheader(f"💹 {selected_crypto} Price with Moving Averages")
            st.line_chart(chart_data, height=400)
            
            # Volume chart
            st.subheader("📊 Volume Analysis")
            volume_data = pd.DataFrame({'Volume': df['volume']})
            st.bar_chart(volume_data, height=200)
            
            # RSI chart
            if 'RSI' in df.columns and not df['RSI'].isna().all():
                st.subheader("📈 RSI (Relative Strength Index)")
                rsi_data = pd.DataFrame({'RSI': df['RSI']})
                st.line_chart(rsi_data, height=200)
                st.caption("RSI > 70: Overbought (Sell Signal) | RSI < 30: Oversold (Buy Signal)")
            
                # Trading Strategy Suggestions
    st.header("💡 Trading Strategy Suggestions")
    
    strategy_col1, strategy_col2 = st.columns(2)
    
    with strategy_col1:
        st.subheader("🎯 Entry Strategy")
        if signals['action'] == 'BUY':
            st.success("**Recommended Action: BUY**")
            st.write("• Wait for volume confirmation")
            st.write("• Enter on any minor dip")
            st.write("• Use limit orders near support")
            st.write("• Scale in if strong momentum")
        elif signals['action'] == 'SELL':
            st.error("**Recommended Action: SELL**")
            st.write("• Short on any bounce to resistance")
            st.write("• Use volume spike to enter")
            st.write("• Consider multiple timeframes")
            st.write("• Tight stop loss required")
        else:
            st.warning("**Recommended Action: HOLD**")
            st.write("• Wait for clear breakout")
            st.write("• Monitor volume patterns")
            st.write("• Watch for trend confirmation")
            st.write("• Patience is key")
    
    with strategy_col2:
        st.subheader("🔄 Exit Strategy")
        st.write("**Take Profit Levels:**")
        st.write(f"• Primary: 6% gain")
        st.write(f"• Secondary: 12% gain")
        st.write(f"• Extended: 20% gain")
        st.write("")
        st.write("**Stop Loss Rules:**")
        st.write("• Never risk more than 2% per trade")
        st.write("• Trail stops after 5% profit")
        st.write("• Consider volatility levels")
        st.write("• Monitor market conditions")

except Exception as e:
    st.error(f"Something went wrong while running the analyzer: {e}")
if __name__ == "__main__":
    main()
