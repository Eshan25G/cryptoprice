import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime, timedelta
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
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
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .signal-sell {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .signal-hold {
        background-color: #ffaa00;
        color: black;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class CryptoAnalyzer:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.indian_exchanges = ['wazirx', 'coindcx', 'bitbns']
        
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
            current_data = requests.get(url, params=params).json()
            
            # Get historical data
            hist_url = f"{self.base_url}/coins/{symbol}/market_chart"
            hist_params = {
                'vs_currency': 'inr',
                'days': days,
                'interval': 'hourly' if days <= 7 else 'daily'
            }
            hist_data = requests.get(hist_url, params=hist_params).json()
            
            # Convert to DataFrame
            df = pd.DataFrame(hist_data['prices'], columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add volume data
            volume_df = pd.DataFrame(hist_data['total_volumes'], columns=['timestamp', 'volume'])
            volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
            volume_df.set_index('timestamp', inplace=True)
            df['volume'] = volume_df['volume']
            
            return df, current_data[symbol]
            
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None, None
    
    def calculate_technical_indicators(self, df):
        """Calculate various technical indicators"""
        # Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['price'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['price'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['price'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['price'], window=26)
        
        # MACD
        df['MACD'] = ta.trend.macd_diff(df['price'])
        df['MACD_signal'] = ta.trend.macd_signal(df['price'])
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['price'], window=14)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['price'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        
        # Stochastic
        df['Stoch_K'] = ta.momentum.stoch(df['price'], df['price'], df['price'])
        df['Stoch_D'] = ta.momentum.stoch_signal(df['price'], df['price'], df['price'])
        
        # Volume indicators
        df['Volume_SMA'] = ta.volume.volume_sma(df['price'], df['volume'])
        
        # Support and Resistance levels
        df['Support'] = df['price'].rolling(window=20).min()
        df['Resistance'] = df['price'].rolling(window=20).max()
        
        return df
    
    def detect_chart_patterns(self, df):
        """Detect common chart patterns"""
        patterns = {}
        
        # Double Top/Bottom detection
        peaks = df['price'].rolling(window=5).max() == df['price']
        troughs = df['price'].rolling(window=5).min() == df['price']
        
        # Trend detection
        short_trend = df['SMA_20'].iloc[-1] > df['SMA_20'].iloc[-5]
        long_trend = df['SMA_50'].iloc[-1] > df['SMA_50'].iloc[-10]
        
        patterns['trend'] = 'Bullish' if short_trend and long_trend else 'Bearish' if not short_trend and not long_trend else 'Sideways'
        
        # Breakout detection
        current_price = df['price'].iloc[-1]
        resistance_level = df['Resistance'].iloc[-5:].mean()
        support_level = df['Support'].iloc[-5:].mean()
        
        if current_price > resistance_level * 1.02:
            patterns['breakout'] = 'Resistance Breakout'
        elif current_price < support_level * 0.98:
            patterns['breakout'] = 'Support Breakdown'
        else:
            patterns['breakout'] = 'No significant breakout'
        
        return patterns
    
    def generate_trading_signals(self, df):
        """Generate comprehensive trading signals"""
        current_data = df.iloc[-1]
        signals = {'action': 'HOLD', 'confidence': 0, 'reasons': []}
        
        buy_signals = 0
        sell_signals = 0
        signal_strength = 0
        
        # RSI Signals
        if current_data['RSI'] < 30:
            buy_signals += 2
            signals['reasons'].append("RSI Oversold (<30)")
        elif current_data['RSI'] > 70:
            sell_signals += 2
            signals['reasons'].append("RSI Overbought (>70)")
        
        # MACD Signals
        if current_data['MACD'] > current_data['MACD_signal'] and df['MACD'].iloc[-2] <= df['MACD_signal'].iloc[-2]:
            buy_signals += 3
            signals['reasons'].append("MACD Bullish Crossover")
        elif current_data['MACD'] < current_data['MACD_signal'] and df['MACD'].iloc[-2] >= df['MACD_signal'].iloc[-2]:
            sell_signals += 3
            signals['reasons'].append("MACD Bearish Crossover")
        
        # Moving Average Signals
        if current_data['price'] > current_data['SMA_20'] > current_data['SMA_50']:
            buy_signals += 2
            signals['reasons'].append("Price above SMAs (Bullish)")
        elif current_data['price'] < current_data['SMA_20'] < current_data['SMA_50']:
            sell_signals += 2
            signals['reasons'].append("Price below SMAs (Bearish)")
        
        # Bollinger Bands
        if current_data['price'] <= current_data['BB_lower']:
            buy_signals += 1
            signals['reasons'].append("Price at lower Bollinger Band")
        elif current_data['price'] >= current_data['BB_upper']:
            sell_signals += 1
            signals['reasons'].append("Price at upper Bollinger Band")
        
        # Stochastic
        if current_data['Stoch_K'] < 20 and current_data['Stoch_D'] < 20:
            buy_signals += 1
            signals['reasons'].append("Stochastic Oversold")
        elif current_data['Stoch_K'] > 80 and current_data['Stoch_D'] > 80:
            sell_signals += 1
            signals['reasons'].append("Stochastic Overbought")
        
        # Volume confirmation
        avg_volume = df['volume'].rolling(window=10).mean().iloc[-1]
        if current_data['volume'] > avg_volume * 1.5:
            signal_strength += 1
            signals['reasons'].append("High volume confirmation")
        
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
            signals['confidence'] = 50
        
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
            'take_profit_short': current_price * 0.94   # 6% profit target
        }

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
        'Shiba Inu': 'shiba-inu'
    }
    
    selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", list(crypto_options.keys()))
    crypto_id = crypto_options[selected_crypto]
    
    # Trading parameters
    st.sidebar.header("💰 Trading Parameters")
    account_balance = st.sidebar.number_input("Account Balance (₹)", min_value=1000, max_value=1000000, value=50000, step=1000)
    risk_per_trade = st.sidebar.slider("Risk per Trade (%)", min_value=1, max_value=5, value=2) / 100
    target_daily_profit = st.sidebar.number_input("Daily Profit Target (₹)", min_value=100, max_value=10000, value=5000, step=100)
    
    # Time frame
    time_frame = st.sidebar.selectbox("Analysis Time Frame", ['7 days', '30 days', '90 days'])
    days = int(time_frame.split()[0])
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Main content
    try:
        # Fetch data
        with st.spinner(f"Fetching {selected_crypto} data..."):
            df, current_data = analyzer.get_crypto_data(crypto_id, days)
        
        if df is not None and current_data is not None:
            # Calculate indicators
            df = analyzer.calculate_technical_indicators(df)
            
            # Current price display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"₹{current_data['inr']:,.2f}",
                    f"{current_data.get('inr_24h_change', 0):.2f}%"
                )
            
            with col2:
                st.metric("24h Volume", f"₹{current_data.get('inr_24h_vol', 0):,.0f}")
            
            with col3:
                st.metric("Market Cap", f"₹{current_data.get('market_cap', {}).get('inr', 0):,.0f}")
            
            with col4:
                required_trades = target_daily_profit / (account_balance * risk_per_trade * 3)  # Assuming 3:1 RR
                st.metric("Required Trades/Day", f"{required_trades:.1f}")
            
            # Trading signals
            signals = analyzer.generate_trading_signals(df)
            patterns = analyzer.detect_chart_patterns(df)
            
            # Signal display
            st.header("📊 Trading Signals")
            
            signal_col1, signal_col2 = st.columns(2)
            
            with signal_col1:
                if signals['action'] == 'BUY':
                    st.markdown(f'<div class="signal-buy">🟢 BUY SIGNAL - Confidence: {signals["confidence"]:.0f}%</div>', unsafe_allow_html=True)
                elif signals['action'] == 'SELL':
                    st.markdown(f'<div class="signal-sell">🔴 SELL SIGNAL - Confidence: {signals["confidence"]:.0f}%</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="signal-hold">🟡 HOLD - Confidence: {signals["confidence"]:.0f}%</div>', unsafe_allow_html=True)
            
            with signal_col2:
                st.write("**Signal Reasons:**")
                for reason in signals['reasons']:
                    st.write(f"• {reason}")
            
            # Position sizing
            position_info = analyzer.calculate_position_sizing(current_data['inr'], account_balance, risk_per_trade)
            
            st.header("💼 Position Management")
            
            pos_col1, pos_col2, pos_col3 = st.columns(3)
            
            with pos_col1:
                if signals['action'] == 'BUY':
                    st.write("**Long Position:**")
                    st.write(f"Position Size: {position_info['long_position_size']:.4f} {selected_crypto}")
                    st.write(f"Entry: ₹{current_data['inr']:,.2f}")
                    st.write(f"Stop Loss: ₹{position_info['stop_loss_long']:,.2f}")
                    st.write(f"Take Profit: ₹{position_info['take_profit_long']:,.2f}")
            
            with pos_col2:
                if signals['action'] == 'SELL':
                    st.write("**Short Position:**")
                    st.write(f"Position Size: {position_info['short_position_size']:.4f} {selected_crypto}")
                    st.write(f"Entry: ₹{current_data['inr']:,.2f}")
                    st.write(f"Stop Loss: ₹{position_info['stop_loss_short']:,.2f}")
                    st.write(f"Take Profit: ₹{position_info['take_profit_short']:,.2f}")
            
            with pos_col3:
                st.write("**Risk Management:**")
                st.write(f"Risk Amount: ₹{account_balance * risk_per_trade:,.2f}")
                st.write(f"Risk/Reward Ratio: 1:3")
                potential_profit = account_balance * risk_per_trade * 3
                st.write(f"Potential Profit: ₹{potential_profit:,.2f}")
            
            # Charts
            st.header("📈 Technical Analysis Charts")
            
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price & Moving Averages', 'Volume', 'RSI', 'MACD'),
                row_width=[0.3, 0.2, 0.2, 0.3]
            )
            
            # Price chart with indicators
            fig.add_trace(go.Scatter(x=df.index, y=df['price'], name='Price', line=dict(color='white', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='red')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
            
            # Volume
            fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='lightblue'), row=2, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            # MACD
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal', line=dict(color='red')), row=4, col=1)
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                title=f"{selected_crypto} Technical Analysis",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Pattern analysis
            st.header("🔍 Pattern Analysis")
            pattern_col1, pattern_col2 = st.columns(2)
            
            with pattern_col1:
                st.write(f"**Trend:** {patterns['trend']}")
                st.write(f"**Breakout:** {patterns['breakout']}")
            
            with pattern_col2:
                # Support and resistance levels
                current_support = df['Support'].iloc[-1]
                current_resistance = df['Resistance'].iloc[-1]
                st.write(f"**Support Level:** ₹{current_support:,.2f}")
                st.write(f"**Resistance Level:** ₹{current_resistance:,.2f}")
            
            # Risk disclaimer
            st.header("⚠️ Important Disclaimer")
            st.warning("""
            **Risk Warning:** Cryptocurrency trading involves substantial risk and is not suitable for everyone. 
            Past performance does not guarantee future results. This tool is for educational purposes only and 
            should not be considered as financial advice. Always do your own research and consider consulting 
            with a financial professional before making investment decisions.
            
            **Daily Profit Expectations:** Generating ₹5,000 daily profit requires significant capital and involves 
            high risk. Market conditions can change rapidly, and losses are possible.
            """)
            
            # Trading tips
            st.header("💡 Trading Tips")
            st.info("""
            **Risk Management:**
            - Never risk more than 2-3% of your account per trade
            - Always use stop losses
            - Maintain a risk-reward ratio of at least 1:2
            
            **Entry Strategies:**
            - Wait for multiple confirmations before entering
            - Consider market sentiment and volume
            - Use proper position sizing
            
            **Exit Strategies:**
            - Stick to your planned stop loss and take profit levels
            - Consider partial profit-taking at resistance levels
            - Trail your stop loss in profitable trades
            """)
            
        else:
            st.error("Failed to fetch cryptocurrency data. Please try again later.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
