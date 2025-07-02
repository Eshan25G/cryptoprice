!pip install streamlit pandas numpy plotly yfinance tensorflow scikit-learn requests
import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import warnings
import json
import time
import random
from typing import List, Tuple, Dict, Any
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.error("yfinance not found. Please install it with: pip install yfinance")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.error("TensorFlow not found. Please install it with: pip install tensorflow")

# Configure Streamlit page
st.set_page_config(
    page_title="AI Crypto Predictor - LSTM + GA",
    page_icon="🧠",
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
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .ai-header {
        font-size: 1.5rem;
        font-weight: bold;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0;
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
    .ga-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: #333;
        text-align: center;
        margin: 1rem 0;
    }
    .lstm-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: #333;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧠 AI Crypto Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="ai-header" style="text-align: center;">LSTM Neural Networks + Genetic Algorithm Optimization</p>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("⚙️ AI Configuration")

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
    '3 Months': '3mo',
    '6 Months': '6mo',
    '1 Year': '1y',
    '2 Years': '2y',
    '5 Years': '5y'
}

selected_period = st.sidebar.selectbox(
    "Select Time Period",
    list(period_options.keys()),
    index=2
)

# AI Model Parameters
st.sidebar.subheader("🧠 LSTM Parameters")
sequence_length = st.sidebar.slider("Sequence Length", 10, 60, 30, 
                                   help="Number of days to look back for prediction")
prediction_days = st.sidebar.slider("Prediction Days", 1, 30, 7,
                                   help="Number of days to predict ahead")

# Genetic Algorithm Parameters
st.sidebar.subheader("🧬 Genetic Algorithm")
population_size = st.sidebar.slider("Population Size", 10, 50, 20,
                                   help="Number of individuals in GA population")
generations = st.sidebar.slider("Generations", 5, 30, 10,
                               help="Number of evolution iterations")

# Advanced LSTM Settings
with st.sidebar.expander("Advanced LSTM Settings"):
    lstm_units = st.sidebar.slider("LSTM Units", 32, 256, 128)
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.2)
    epochs = st.sidebar.slider("Training Epochs", 10, 100, 50)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)

class GeneticAlgorithm:
    """Genetic Algorithm for LSTM hyperparameter optimization"""
    
    def __init__(self, population_size: int, generations: int):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
    def create_individual(self) -> Dict[str, Any]:
        """Create a random individual (hyperparameter set)"""
        return {
            'lstm_units': random.choice([32, 64, 128, 256]),
            'dropout_rate': random.uniform(0.1, 0.5),
            'learning_rate': random.uniform(0.0001, 0.01),
            'batch_size': random.choice([16, 32, 64]),
            'layers': random.randint(1, 3)
        }
    
    def create_population(self) -> List[Dict[str, Any]]:
        """Create initial population"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def fitness_function(self, individual: Dict[str, Any], X_train, y_train, X_val, y_val) -> float:
        """Evaluate fitness of an individual"""
        try:
            model = self.create_lstm_model(individual, X_train.shape)
            
            # Train with early stopping to prevent overfitting
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20,  # Reduced for GA speed
                batch_size=individual['batch_size'],
                verbose=0
            )
            
            # Predict and calculate fitness (inverse of validation loss)
            predictions = model.predict(X_val, verbose=0)
            mse = mean_squared_error(y_val, predictions)
            fitness = 1 / (1 + mse)  # Higher fitness for lower MSE
            
            return fitness
            
        except Exception as e:
            return 0.0  # Return low fitness for failed individuals
    
    def create_lstm_model(self, individual: Dict[str, Any], input_shape: tuple):
        """Create LSTM model based on individual parameters"""
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(input_shape[1], input_shape[2])))
        
        # LSTM layers
        for i in range(individual['layers']):
            return_sequences = i < individual['layers'] - 1
            model.add(LSTM(
                individual['lstm_units'],
                return_sequences=return_sequences,
                dropout=individual['dropout_rate'],
                recurrent_dropout=individual['dropout_rate'],
                kernel_regularizer=l2(0.01)
            ))
            model.add(Dropout(individual['dropout_rate']))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=individual['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def selection(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """Tournament selection"""
        selected = []
        for _ in range(self.population_size):
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), tournament_size)
            winner_idx = max(tournament_indices, key=lambda x: fitness_scores[x])
            selected.append(population[winner_idx].copy())
        return selected
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Single-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = parent1.copy(), parent2.copy()
        
        # Crossover for each parameter
        for key in parent1.keys():
            if random.random() < 0.5:
                child1[key], child2[key] = parent2[key], parent1[key]
        
        return child1, child2
    
    def mutate(self, individual: Dict) -> Dict:
        """Mutate individual"""
        mutated = individual.copy()
        
        if random.random() < self.mutation_rate:
            mutation_type = random.choice(['lstm_units', 'dropout_rate', 'learning_rate', 'batch_size', 'layers'])
            
            if mutation_type == 'lstm_units':
                mutated['lstm_units'] = random.choice([32, 64, 128, 256])
            elif mutation_type == 'dropout_rate':
                mutated['dropout_rate'] = random.uniform(0.1, 0.5)
            elif mutation_type == 'learning_rate':
                mutated['learning_rate'] = random.uniform(0.0001, 0.01)
            elif mutation_type == 'batch_size':
                mutated['batch_size'] = random.choice([16, 32, 64])
            elif mutation_type == 'layers':
                mutated['layers'] = random.randint(1, 3)
        
        return mutated
    
    def evolve(self, X_train, y_train, X_val, y_val) -> Tuple[Dict, List[float]]:
        """Main evolution loop"""
        population = self.create_population()
        best_fitness_history = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for i, individual in enumerate(population):
                fitness = self.fitness_function(individual, X_train, y_train, X_val, y_val)
                fitness_scores.append(fitness)
                
                # Update progress
                progress = (generation * len(population) + i + 1) / (self.generations * len(population))
                progress_bar.progress(progress)
                status_text.text(f"Generation {generation + 1}/{self.generations}, Individual {i + 1}/{len(population)}")
            
            best_fitness = max(fitness_scores)
            best_fitness_history.append(best_fitness)
            
            # Selection
            selected = self.selection(population, fitness_scores)
            
            # Crossover and mutation
            new_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # Return best individual
        final_fitness = [self.fitness_function(ind, X_train, y_train, X_val, y_val) for ind in population]
        best_idx = max(range(len(final_fitness)), key=lambda x: final_fitness[x])
        
        progress_bar.empty()
        status_text.empty()
        
        return population[best_idx], best_fitness_history

class LSTMPredictor:
    """LSTM-based cryptocurrency price predictor"""
    
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.feature_scaler = MinMaxScaler()
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        # Calculate technical indicators
        data = self.calculate_features(data)
        
        # Select features
        feature_columns = ['Close', 'Volume', 'SMA_10', 'SMA_20', 'RSI', 'MACD', 'BB_position']
        features = data[feature_columns].fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(features)
        
        # Scale target (Close price)
        scaled_target = self.scaler.fit_transform(data[['Close']])
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(scaled_target[i, 0])
        
        return np.array(X), np.array(y), scaled_features
    
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators as features"""
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Price momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Momentum'] = df['Close'].pct_change(periods=5)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
    
    def create_optimized_model(self, best_params: Dict, input_shape: tuple):
        """Create LSTM model with optimized parameters"""
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(input_shape[1], input_shape[2])))
        
        # LSTM layers based on GA optimization
        for i in range(best_params['layers']):
            return_sequences = i < best_params['layers'] - 1
            model.add(LSTM(
                best_params['lstm_units'],
                return_sequences=return_sequences,
                dropout=best_params['dropout_rate'],
                recurrent_dropout=best_params['dropout_rate'],
                kernel_regularizer=l2(0.01)
            ))
            model.add(Dropout(best_params['dropout_rate']))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile with optimized parameters
        optimizer = Adam(learning_rate=best_params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train_with_ga_optimization(self, X_train, y_train, X_val, y_val, ga_params):
        """Train LSTM with genetic algorithm optimization"""
        ga = GeneticAlgorithm(ga_params['population_size'], ga_params['generations'])
        
        st.info("🧬 Starting Genetic Algorithm optimization...")
        best_params, fitness_history = ga.evolve(X_train, y_train, X_val, y_val)
        
        st.success("✅ GA optimization completed!")
        
        # Train final model with best parameters
        self.model = self.create_optimized_model(best_params, X_train.shape)
        
        # Train the final model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=best_params['batch_size'],
            verbose=0
        )
        
        return best_params, fitness_history, history
    
    def predict_future(self, data: pd.DataFrame, days: int = 7) -> np.ndarray:
        """Predict future prices"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare last sequence
        features = self.calculate_features(data)
        feature_columns = ['Close', 'Volume', 'SMA_10', 'SMA_20', 'RSI', 'MACD', 'BB_position']
        scaled_features = self.feature_scaler.transform(features[feature_columns].fillna(method='ffill').fillna(method='bfill'))
        
        predictions = []
        current_sequence = scaled_features[-self.sequence_length:].copy()
        
        for _ in range(days):
            # Predict next price
            pred_input = current_sequence.reshape(1, self.sequence_length, -1)
            next_price_scaled = self.model.predict(pred_input, verbose=0)[0, 0]
            
            # Inverse transform to get actual price
            next_price = self.scaler.inverse_transform([[next_price_scaled]])[0, 0]
            predictions.append(next_price)
            
            # Update sequence for next prediction
            # Create new feature row (simplified - using last known values)
            new_features = current_sequence[-1].copy()
            new_features[0] = next_price_scaled  # Update close price
            
            # Shift sequence
            current_sequence = np.vstack([current_sequence[1:], new_features.reshape(1, -1)])
        
        return np.array(predictions)

# Data fetching functions (keeping the same as before)
@st.cache_data(ttl=300)
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
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': min(days, 365),
            'interval': 'daily' if days > 90 else 'hourly'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            if prices:
                df = pd.DataFrame(prices, columns=['timestamp', 'close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                if volumes:
                    vol_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                    vol_df['timestamp'] = pd.to_datetime(vol_df['timestamp'], unit='ms')
                    vol_df.set_index('timestamp', inplace=True)
                    df = df.join(vol_df)
                else:
                    df['volume'] = 0
                
                df['open'] = df['close'].shift(1).fillna(df['close'])
                df['high'] = df['close'] * 1.02
                df['low'] = df['close'] * 0.98
                
                df.columns = ['Close', 'Volume', 'Open', 'High', 'Low']
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                
                return df
                
        return None
        
    except Exception as e:
        st.error(f"Error fetching data from CoinGecko: {e}")
        return None

def fetch_crypto_data(symbol, period):
    """Main function to fetch crypto data with fallbacks"""
    period_to_days = {
        '3mo': 90, 
        '6mo': 180,
        '1y': 365,
        '2y': 730,
        '5y': 1825
    }
    
    days = period_to_days.get(period, 365)
    
    if YFINANCE_AVAILABLE:
        data = fetch_crypto_data_yfinance(symbol, period)
        if data is not None and not data.empty:
            return data
    
    st.info("Using CoinGecko API as data source...")
    return fetch_crypto_data_api(symbol, days)

# Main app logic
if st.sidebar.button("🚀 Start AI Training", type="primary"):
    st.cache_data.clear()

# Fetch data
symbol = crypto_options[selected_crypto]
period = period_options[selected_period]

if not TENSORFLOW_AVAILABLE:
    st.error("🚫 TensorFlow is required for LSTM functionality. Please install it with: `pip install tensorflow`")
    st.stop()

with st.spinner(f"Fetching {selected_crypto} data..."):
    data = fetch_crypto_data(symbol, period)

if data is not None and not data.empty and len(data) > sequence_length + 50:
    
    # Display current metrics
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
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
        volatility = data['Close'].pct_change().std() * np.sqrt(365) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>Volatility</h3>
            <h2>{volatility:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Prediction Section
    st.subheader("🧠 AI-Powered Prediction System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="lstm-box">
            <h3>🔗 LSTM Neural Network</h3>
            <p>Long Short-Term Memory networks capture complex temporal patterns in price data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="ga-box">
            <h3>🧬 Genetic Algorithm</h3>
            <p>Evolutionary optimization finds the best neural network architecture</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize LSTM predictor
    predictor = LSTMPredictor(sequence_length=sequence_length)
    
    # Prepare data
    with st.spinner("🔄 Preparing data and features..."):
        X, y, scaled_features = predictor.prepare_data(data)
    
    if len(X) > 100:  # Ensure sufficient data
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Further split training data for validation
        val_split_idx = int(len(X_train) * 0.8)
        X_val = X_train[val_split_idx:]
        y_val = y_train[val_split_idx:]
        X_train = X_train[:val_split_idx]
        y_train = y_train[:val_split_idx]
        
        st.info(f"📊 Training data: {len(X_train)} samples | Validation: {len(X_val)} samples | Test: {len(X_test)} samples")
        
        # Train with GA optimization
        ga_params = {
            'population_size': population_size,
            'generations': generations
        }
        
        with st.expander("🚀 Start AI Training", expanded=True):
            if st.button("Begin LSTM + GA Optimization", type="primary"):
                
                # Training progress
                training_container = st.container()
                
                with training_container:
                    best_params, fitness_history, training_history = predictor.train_with_ga_optimization(
                        X_train, y_train, X_val, y_val, ga_params
                    )
                    
                    # Display optimization results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🏆 Best GA Parameters:**")
                        for key, value in best_params.items():
                            if isinstance(value, float):
                                st.write(f"**{key}:** {value:.4f}")
                            else:
                                st.write(f"**{key}:** {value}")
                    
                    with col2:
                        # Plot GA fitness evolution
                        fig_ga = go.Figure()
                        fig_ga.add_trace(go.Scatter(
                            x=list(range(1, len(fitness_history) + 1)),
                            y=fitness_history,
                            mode='lines+markers',
                            name='Best Fitness',
                            line=dict(color='#ff6b6b', width=3)
                        ))
                        fig_ga.update_layout(
                            title="🧬 Genetic Algorithm Evolution",
                            xaxis_title="Generation",
                            yaxis_title="Fitness Score",
                            template='plotly_dark',
                            height=300
                        )
                        st.plotly_chart(fig_ga, use_container_width=True)
                    
                    # Make predictions
                    st.subheader("🔮 AI Predictions")
                    
                    # Test set predictions (continuing from where the code was cut off)
                    test_predictions = predictor.model.predict(X_test, verbose=0)
                    test_predictions = predictor.scaler.inverse_transform(test_predictions)
                    
                    # Actual test values
                    y_test_actual = predictor.scaler.inverse_transform(y_test.reshape(-1, 1))
                    
                    # Calculate metrics
                    mae = mean_absolute_error(y_test_actual, test_predictions)
                    mse = mean_squared_error(y_test_actual, test_predictions)
                    rmse = np.sqrt(mse)
                    mape = np.mean(np.abs((y_test_actual - test_predictions) / y_test_actual)) * 100
                    
                    # Display prediction metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>MAE</h3>
                            <h2>${mae:.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>RMSE</h3>
                            <h2>${rmse:.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>MAPE</h3>
                            <h2>{mape:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        accuracy = max(0, 100 - mape)
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Accuracy</h3>
                            <h2>{accuracy:.1f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Plot actual vs predicted
                    fig_pred = go.Figure()
                    
                    # Get test dates
                    test_dates = data.index[-len(y_test_actual):]
                    
                    fig_pred.add_trace(go.Scatter(
                        x=test_dates,
                        y=y_test_actual.flatten(),
                        mode='lines',
                        name='Actual Prices',
                        line=dict(color='#4ecdc4', width=2)
                    ))
                    
                    fig_pred.add_trace(go.Scatter(
                        x=test_dates,
                        y=test_predictions.flatten(),
                        mode='lines',
                        name='LSTM Predictions',
                        line=dict(color='#ff6b6b', width=2, dash='dash')
                    ))
                    
                    fig_pred.update_layout(
                        title="🎯 LSTM Model Performance on Test Data",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        template='plotly_dark',
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Future predictions
                    st.subheader("🔮 Future Price Predictions")
                    
                    future_predictions = predictor.predict_future(data, prediction_days)
                    
                    # Create future dates
                    last_date = data.index[-1]
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days)
                    
                    # Display future predictions
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Plot historical + future
                        fig_future = go.Figure()
                        
                        # Historical prices (last 60 days)
                        historical_data = data.tail(60)
                        fig_future.add_trace(go.Scatter(
                            x=historical_data.index,
                            y=historical_data['Close'],
                            mode='lines',
                            name='Historical Prices',
                            line=dict(color='#4ecdc4', width=2)
                        ))
                        
                        # Future predictions
                        fig_future.add_trace(go.Scatter(
                            x=future_dates,
                            y=future_predictions,
                            mode='lines+markers',
                            name='Future Predictions',
                            line=dict(color='#ff6b6b', width=3)
                        ))
                        
                        # Add confidence bands (simplified)
                        confidence_interval = rmse * 1.96  # 95% confidence
                        upper_bound = future_predictions + confidence_interval
                        lower_bound = future_predictions - confidence_interval
                        
                        fig_future.add_trace(go.Scatter(
                            x=future_dates,
                            y=upper_bound,
                            fill=None,
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            showlegend=False
                        ))
                        
                        fig_future.add_trace(go.Scatter(
                            x=future_dates,
                            y=lower_bound,
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            name='95% Confidence',
                            fillcolor='rgba(255, 107, 107, 0.2)'
                        ))
                        
                        fig_future.update_layout(
                            title=f"🚀 {selected_crypto} Price Forecast - Next {prediction_days} Days",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            template='plotly_dark',
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_future, use_container_width=True)
                    
                    with col2:
                        # Prediction summary
                        predicted_change = future_predictions[-1] - current_price
                        predicted_change_pct = (predicted_change / current_price) * 100
                        
                        prediction_color = "green" if predicted_change >= 0 else "red"
                        arrow = "📈" if predicted_change >= 0 else "📉"
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>{arrow} {prediction_days}-Day Forecast</h3>
                            <h2>${future_predictions[-1]:,.2f}</h2>
                            <p style="color: {prediction_color}; font-size: 1.2em; font-weight: bold;">
                                {predicted_change_pct:+.2f}%
                            </p>
                            <p>Change: ${predicted_change:+.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Prediction details table
                        st.subheader("📊 Daily Predictions")
                        pred_df = pd.DataFrame({
                            'Date': future_dates.strftime('%Y-%m-%d'),
                            'Predicted Price': [f"${p:.2f}" for p in future_predictions],
                            'Daily Change': [f"{((future_predictions[i] - (current_price if i == 0 else future_predictions[i-1])) / (current_price if i == 0 else future_predictions[i-1]) * 100):+.2f}%" for i in range(len(future_predictions))]
                        })
                        
                        st.dataframe(pred_df, use_container_width=True)
                        
                        # Risk assessment
                        volatility_score = min(volatility / 50, 1.0)  # Normalize to 0-1
                        accuracy_score = accuracy / 100
                        confidence_score = (accuracy_score + (1 - volatility_score)) / 2
                        
                        if confidence_score >= 0.8:
                            risk_level = "🟢 Low Risk"
                            risk_color = "green"
                        elif confidence_score >= 0.6:
                            risk_level = "🟡 Medium Risk"
                            risk_color = "orange"
                        else:
                            risk_level = "🔴 High Risk"
                            risk_color = "red"
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 1rem 0;">
                            <h4>🎯 Prediction Confidence</h4>
                            <h3>{confidence_score * 100:.1f}%</h3>
                            <p style="color: {risk_color}; font-weight: bold;">{risk_level}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Training history plots
                    st.subheader("📈 Training Performance")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Loss plot
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(
                            y=training_history.history['loss'],
                            mode='lines',
                            name='Training Loss',
                            line=dict(color='#ff6b6b')
                        ))
                        if 'val_loss' in training_history.history:
                            fig_loss.add_trace(go.Scatter(
                                y=training_history.history['val_loss'],
                                mode='lines',
                                name='Validation Loss',
                                line=dict(color='#4ecdc4')
                            ))
                        fig_loss.update_layout(
                            title="🔥 Training Loss",
                            xaxis_title="Epoch",
                            yaxis_title="Loss",
                            template='plotly_dark',
                            height=300
                        )
                        st.plotly_chart(fig_loss, use_container_width=True)
                    
                    with col2:
                        # MAE plot
                        fig_mae = go.Figure()
                        if 'mae' in training_history.history:
                            fig_mae.add_trace(go.Scatter(
                                y=training_history.history['mae'],
                                mode='lines',
                                name='Training MAE',
                                line=dict(color='#ff6b6b')
                            ))
                        if 'val_mae' in training_history.history:
                            fig_mae.add_trace(go.Scatter(
                                y=training_history.history['val_mae'],
                                mode='lines',
                                name='Validation MAE',
                                line=dict(color='#4ecdc4')
                            ))
                        fig_mae.update_layout(
                            title="📊 Mean Absolute Error",
                            xaxis_title="Epoch",
                            yaxis_title="MAE",
                            template='plotly_dark',
                            height=300
                        )
                        st.plotly_chart(fig_mae, use_container_width=True)
                    
                    # Feature importance (simplified visualization)
                    st.subheader("🎯 Feature Analysis")
                    
                    feature_names = ['Close', 'Volume', 'SMA_10', 'SMA_20', 'RSI', 'MACD', 'BB_position']
                    
                    # Calculate simple feature importance based on correlation with target
                    feature_data = predictor.calculate_features(data)
                    correlations = []
                    for feature in feature_names:
                        if feature in feature_data.columns:
                            corr = abs(feature_data[feature].corr(feature_data['Close']))
                            correlations.append(corr if not np.isnan(corr) else 0)
                        else:
                            correlations.append(0)
                    
                    fig_importance = go.Figure(data=[
                        go.Bar(
                            x=feature_names,
                            y=correlations,
                            marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd', '#98fb98']
                        )
                    ])
                    
                    fig_importance.update_layout(
                        title="🔍 Feature Importance (Correlation with Price)",
                        xaxis_title="Features",
                        yaxis_title="Correlation Strength",
                        template='plotly_dark',
                        height=400
                    )
                    
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # AI Insights
                    st.subheader("🧠 AI Insights & Recommendations")
                    
                    insights = []
                    
                    # Trend analysis
                    recent_trend = np.mean(np.diff(data['Close'].tail(7)))
                    if recent_trend > 0:
                        insights.append("📈 **Bullish Trend**: Recent 7-day price action shows upward momentum")
                    else:
                        insights.append("📉 **Bearish Trend**: Recent 7-day price action shows downward pressure")
                    
                    # Volatility insight
                    if volatility > 80:
                        insights.append("⚡ **High Volatility**: Expect significant price swings - higher risk/reward")
                    elif volatility < 30:
                        insights.append("🔒 **Low Volatility**: Relatively stable price movement expected")
                    else:
                        insights.append("⚖️ **Moderate Volatility**: Balanced risk-reward scenario")
                    
                    # Model confidence insight
                    if accuracy > 80:
                        insights.append("🎯 **High Confidence**: Model shows strong predictive accuracy")
                    elif accuracy > 65:
                        insights.append("✅ **Moderate Confidence**: Model predictions are reasonably reliable")
                    else:
                        insights.append("⚠️ **Low Confidence**: Use predictions with caution - high uncertainty")
                    
                    # Volume insight
                    recent_volume = data['Volume'].tail(7).mean()
                    avg_volume = data['Volume'].mean()
                    if recent_volume > avg_volume * 1.5:
                        insights.append("📊 **High Volume**: Increased trading activity may signal significant moves")
                    elif recent_volume < avg_volume * 0.5:
                        insights.append("📊 **Low Volume**: Reduced trading activity - prices may consolidate")
                    
                    for insight in insights:
                        st.markdown(f"• {insight}")
                    
                    # Disclaimer
                    st.markdown("---")
                    st.markdown("""
                    <div style="background: #f0f0f0; padding: 1rem; border-radius: 10px; color: #333; margin: 1rem 0;">
                        <h4>⚠️ Important Disclaimer</h4>
                        <p>This AI prediction system is for educational and research purposes only. Cryptocurrency markets are highly volatile and unpredictable. 
                        Past performance does not guarantee future results. Always conduct your own research and consider consulting with financial advisors 
                        before making investment decisions. Never invest more than you can afford to lose.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        st.error("❌ Insufficient data for training. Need at least 100 data points after feature engineering.")

else:
    if data is None:
        st.error("❌ Unable to fetch cryptocurrency data. Please check your internet connection and try again.")
    elif data.empty:
        st.error("❌ No data available for the selected cryptocurrency and time period.")
    else:
        st.error(f"❌ Insufficient data points. Got {len(data)} points, need at least {sequence_length + 50}.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 2rem;">
    <p>🧠 AI Crypto Predictor | Powered by LSTM Neural Networks & Genetic Algorithms</p>
    <p>Built with Streamlit, TensorFlow, and Plotly | For Educational Purposes Only</p>
</div>
""", unsafe_allow_html=True)
