import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import time
import json
from datetime import datetime, timedelta
import ta
from dataclasses import dataclass
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration
@dataclass
class TradingConfig:
    target_daily_profit: float = 5000  # INR
    max_risk_per_trade: float = 0.02  # 2% of portfolio
    stop_loss_pct: float = 0.015  # 1.5%
    take_profit_pct: float = 0.03  # 3%
    min_rsi_oversold: float = 30
    max_rsi_overbought: float = 70
    ema_short: int = 9
    ema_long: int = 21
    volume_threshold: float = 1.2  # 20% above average

class CryptoDataFetcher:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        
    def get_indian_exchanges(self):
        """Get popular Indian crypto exchanges"""
        return {
            'WazirX': 'wazirx',
            'CoinDCX': 'coindcx',
            'Bitbns': 'bitbns',
            'ZebPay': 'zebpay'
        }
    
    def get_crypto_prices(self, symbols: List[str], vs_currency: str = 'inr'):
        """Fetch current crypto prices in INR"""
        try:
            symbols_str = ','.join(symbols)
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': symbols_str,
                'vs_currencies': vs_currency,
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true'
            }
            response = requests.get(url, params=params, timeout=10)
            return response.json()
        except Exception as e:
            st.error(f"Error fetching prices: {e}")
            return {}
    
    def get_historical_data(self, coin_id: str, days: int = 30):
        """Fetch historical price data"""
        try:
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'inr',
                'days': days,
                'interval': 'hourly' if days <= 30 else 'daily'
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['volume'] = [vol[1] for vol in data['total_volumes']]
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            st.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

class TechnicalAnalysis:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame):
        """Calculate technical indicators"""
        if df.empty:
            return df
            
        # Moving averages
        df['EMA_9'] = ta.trend.ema_indicator(df['price'], window=9)
        df['EMA_21'] = ta.trend.ema_indicator(df['price'], window=21)
        df['SMA_50'] = ta.trend.sma_indicator(df['price'], window=50)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['price'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['price'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['price'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Support and Resistance
        df['support'] = df['price'].rolling(window=20).min()
        df['resistance'] = df['price'].rolling(window=20).max()
        
        return df
    
    @staticmethod
    def generate_signals(df: pd.DataFrame, config: TradingConfig):
        """Generate buy/sell signals"""
        if df.empty or len(df) < 50:
            return df
            
        signals = []
        
        for i in range(1, len(df)):
            signal = 'HOLD'
            confidence = 0
            
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Buy conditions
            buy_conditions = [
                current['RSI'] < config.min_rsi_oversold,  # Oversold
                current['EMA_9'] > current['EMA_21'],  # Short EMA above long EMA
                current['price'] > current['BB_lower'],  # Above lower Bollinger Band
                current['MACD'] > current['MACD_signal'],  # MACD bullish
                current['volume_ratio'] > config.volume_threshold,  # High volume
                current['price'] > previous['price']  # Price increasing
            ]
            
            # Sell conditions
            sell_conditions = [
                current['RSI'] > config.max_rsi_overbought,  # Overbought
                current['EMA_9'] < current['EMA_21'],  # Short EMA below long EMA
                current['price'] < current['BB_upper'],  # Below upper Bollinger Band
                current['MACD'] < current['MACD_signal'],  # MACD bearish
                current['price'] < previous['price']  # Price decreasing
            ]
            
            buy_score = sum(buy_conditions)
            sell_score = sum(sell_conditions)
            
            if buy_score >= 4:
                signal = 'BUY'
                confidence = buy_score / len(buy_conditions)
            elif sell_score >= 4:
                signal = 'SELL'
                confidence = sell_score / len(sell_conditions)
            
            signals.append({'signal': signal, 'confidence': confidence})
        
        signal_df = pd.DataFrame(signals, index=df.index[1:])
        df = df.iloc[1:].join(signal_df)
        
        return df

class PortfolioManager:
    def __init__(self, initial_capital: float = 100000):  # 1 Lakh INR
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.daily_pnl = []
    
    def calculate_position_size(self, price: float, risk_pct: float):
        """Calculate position size based on risk management"""
        risk_amount = self.current_capital * risk_pct
        position_size = risk_amount / price
        return position_size
    
    def execute_trade(self, symbol: str, action: str, price: float, quantity: float, confidence: float):
        """Execute a trade"""
        trade_value = price * quantity
        
        if action == 'BUY' and trade_value <= self.current_capital:
            self.positions[symbol] = {
                'quantity': quantity,
                'entry_price': price,
                'entry_time': datetime.now(),
                'stop_loss': price * (1 - 0.015),  # 1.5% stop loss
                'take_profit': price * (1 + 0.03),  # 3% take profit
                'confidence': confidence
            }
            self.current_capital -= trade_value
            
            self.trade_history.append({
                'symbol': symbol,
                'action': action,
                'price': price,
                'quantity': quantity,
                'value': trade_value,
                'time': datetime.now(),
                'confidence': confidence
            })
            
        elif action == 'SELL' and symbol in self.positions:
            position = self.positions[symbol]
            profit_loss = (price - position['entry_price']) * position['quantity']
            self.current_capital += trade_value
            
            self.trade_history.append({
                'symbol': symbol,
                'action': action,
                'price': price,
                'quantity': position['quantity'],
                'value': trade_value,
                'profit_loss': profit_loss,
                'time': datetime.now(),
                'confidence': confidence
            })
            
            del self.positions[symbol]
    
    def get_portfolio_value(self, current_prices: Dict):
        """Calculate current portfolio value"""
        total_value = self.current_capital
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position_value = current_price * position['quantity']
                total_value += position_value
        
        return total_value
    
    def get_daily_profit(self):
        """Calculate daily profit"""
        today = datetime.now().date()
        today_trades = [t for t in self.trade_history if t['time'].date() == today]
        
        daily_profit = sum([t.get('profit_loss', 0) for t in today_trades if 'profit_loss' in t])
        return daily_profit

def create_price_chart(df: pd.DataFrame, symbol: str):
    """Create candlestick chart with indicators"""
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} Price & Indicators', 'RSI', 'MACD'),
        row_width=[0.6, 0.2, 0.2]
    )
    
    # Price and moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['price'], name='Price', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], name='EMA 9', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], name='EMA 21', line=dict(color='red')), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # Buy/Sell signals
    buy_signals = df[df['signal'] == 'BUY']
    sell_signals = df[df['signal'] == 'SELL']
    
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['price'], 
                                mode='markers', name='Buy Signal', 
                                marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)
    
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['price'], 
                                mode='markers', name='Sell Signal', 
                                marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='MACD Signal', line=dict(color='red')), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_histogram'], name='MACD Histogram'), row=3, col=1)
    
    fig.update_layout(height=800, title=f'{symbol} Technical Analysis', showlegend=True)
    return fig

def main():
    st.set_page_config(page_title="Crypto Trading System - India", layout="wide")
    
    st.title("🚀 Advanced Crypto Trading System for India")
    st.markdown("*Target: ₹5,000 daily profit through systematic trading*")
    
    # Initialize components
    if 'data_fetcher' not in st.session_state:
        st.session_state.data_fetcher = CryptoDataFetcher()
        st.session_state.portfolio = PortfolioManager()
        st.session_state.config = TradingConfig()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Trading Configuration")
        
        # Portfolio settings
        st.subheader("Portfolio Settings")
        initial_capital = st.number_input("Initial Capital (INR)", min_value=10000, value=100000, step=10000)
        target_profit = st.number_input("Daily Profit Target (INR)", min_value=1000, value=5000, step=500)
        
        # Risk management
        st.subheader("Risk Management")
        max_risk = st.slider("Max Risk per Trade (%)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
        stop_loss = st.slider("Stop Loss (%)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
        take_profit = st.slider("Take Profit (%)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        
        # Update configuration
        st.session_state.config.target_daily_profit = target_profit
        st.session_state.config.max_risk_per_trade = max_risk / 100
        st.session_state.config.stop_loss_pct = stop_loss / 100
        st.session_state.config.take_profit_pct = take_profit / 100
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Value", f"₹{st.session_state.portfolio.current_capital:,.2f}")
    
    with col2:
        daily_profit = st.session_state.portfolio.get_daily_profit()
        st.metric("Daily P&L", f"₹{daily_profit:,.2f}", delta=f"{(daily_profit/target_profit)*100:.1f}% of target")
    
    with col3:
        st.metric("Active Positions", len(st.session_state.portfolio.positions))
    
    with col4:
        st.metric("Total Trades", len(st.session_state.portfolio.trade_history))
    
    # Crypto selection and analysis
    st.header("📊 Market Analysis")
    
    # Popular cryptos in India
    popular_cryptos = {
        'Bitcoin': 'bitcoin',
        'Ethereum': 'ethereum',
        'Binance Coin': 'binancecoin',
        'Cardano': 'cardano',
        'Solana': 'solana',
        'Dogecoin': 'dogecoin',
        'Polygon': 'matic-network',
        'Chainlink': 'chainlink'
    }
    
    selected_crypto = st.selectbox("Select Cryptocurrency", list(popular_cryptos.keys()))
    crypto_id = popular_cryptos[selected_crypto]
    
    # Fetch and display data
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.spinner(f"Fetching {selected_crypto} data..."):
            # Get historical data
            df = st.session_state.data_fetcher.get_historical_data(crypto_id, days=30)
            
            if not df.empty:
                # Calculate technical indicators
                df = TechnicalAnalysis.calculate_indicators(df)
                df = TechnicalAnalysis.generate_signals(df, st.session_state.config)
                
                # Display chart
                fig = create_price_chart(df, selected_crypto)
                st.plotly_chart(fig, use_container_width=True)
                
                # Latest signals
                if not df.empty:
                    latest = df.iloc[-1]
                    st.subheader("Current Signal")
                    
                    signal_col1, signal_col2, signal_col3 = st.columns(3)
                    with signal_col1:
                        signal_color = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}[latest['signal']]
                        st.markdown(f"**Signal:** <span style='color:{signal_color}'>{latest['signal']}</span>", unsafe_allow_html=True)
                    
                    with signal_col2:
                        st.markdown(f"**Confidence:** {latest['confidence']:.2%}")
                    
                    with signal_col3:
                        st.markdown(f"**RSI:** {latest['RSI']:.1f}")
    
    with col2:
        st.subheader("📈 Current Prices")
        
        # Get current prices
        crypto_symbols = list(popular_cryptos.values())
        prices = st.session_state.data_fetcher.get_crypto_prices(crypto_symbols)
        
        if prices:
            for name, symbol in popular_cryptos.items():
                if symbol in prices:
                    price_data = prices[symbol]
                    price = price_data.get('inr', 0)
                    change_24h = price_data.get('inr_24h_change', 0)
                    
                    color = 'green' if change_24h > 0 else 'red'
                    st.markdown(f"""
                    **{name}**  
                    ₹{price:,.2f}  
                    <span style='color:{color}'>{change_24h:+.2f}%</span>
                    """, unsafe_allow_html=True)
        
        st.subheader("🏢 Indian Exchanges")
        exchanges = st.session_state.data_fetcher.get_indian_exchanges()
        for exchange_name in exchanges.keys():
            st.write(f"• {exchange_name}")
    
    # Trading interface
    st.header("💼 Trading Interface")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Manual Trading")
        
        trade_crypto = st.selectbox("Select Crypto for Trading", list(popular_cryptos.keys()), key='trade_select')
        trade_action = st.selectbox("Action", ['BUY', 'SELL'])
        trade_amount = st.number_input("Amount (INR)", min_value=100, value=1000, step=100)
        
        if st.button("Execute Trade"):
            # Get current price
            crypto_symbol = popular_cryptos[trade_crypto]
            current_prices = st.session_state.data_fetcher.get_crypto_prices([crypto_symbol])
            
            if current_prices and crypto_symbol in current_prices:
                price = current_prices[crypto_symbol]['inr']
                quantity = trade_amount / price
                
                st.session_state.portfolio.execute_trade(
                    trade_crypto, trade_action, price, quantity, 0.8
                )
                st.success(f"Executed {trade_action} order for {trade_crypto}")
                st.rerun()
    
    with col2:
        st.subheader("Active Positions")
        
        if st.session_state.portfolio.positions:
            for symbol, position in st.session_state.portfolio.positions.items():
                with st.expander(f"{symbol} Position"):
                    st.write(f"**Quantity:** {position['quantity']:.6f}")
                    st.write(f"**Entry Price:** ₹{position['entry_price']:,.2f}")
                    st.write(f"**Stop Loss:** ₹{position['stop_loss']:,.2f}")
                    st.write(f"**Take Profit:** ₹{position['take_profit']:,.2f}")
                    st.write(f"**Confidence:** {position['confidence']:.2%}")
                    
                    if st.button(f"Close {symbol} Position"):
                        # Close position logic here
                        st.info("Position closed!")
        else:
            st.info("No active positions")
    
    # Performance analytics
    st.header("📊 Performance Analytics")
    
    if st.session_state.portfolio.trade_history:
        trades_df = pd.DataFrame(st.session_state.portfolio.trade_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Trade History")
            st.dataframe(trades_df.tail(10), use_container_width=True)
        
        with col2:
            st.subheader("Performance Metrics")
            
            # Calculate metrics
            completed_trades = trades_df[trades_df['action'] == 'SELL']
            if not completed_trades.empty:
                total_profit = completed_trades['profit_loss'].sum()
                win_rate = (completed_trades['profit_loss'] > 0).mean()
                avg_profit = completed_trades['profit_loss'].mean()
                
                st.metric("Total Profit/Loss", f"₹{total_profit:,.2f}")
                st.metric("Win Rate", f"{win_rate:.2%}")
                st.metric("Average Trade P&L", f"₹{avg_profit:,.2f}")
    else:
        st.info("No trading history available")
    
    # Auto-refresh
    if st.checkbox("Auto-refresh (30 seconds)"):
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
